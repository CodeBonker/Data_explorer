import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import json
import joblib
from datetime import datetime
import hashlib

META_MODEL_PATH = "data/meta_model.pkl"
META_LOG_PATH = "data/meta_logs.csv"
META_INFO_PATH = "data/meta_info.json"
class MetaFeatureExtractor:
    """ Extracts dataset level meta-features for meta-learning """

    @staticmethod
    def extract(df: pd.DataFrame, target: str):
        num_features = df.shape[1] - 1
        num_rows = df.shape[0]
        num_numeric = len(df.select_dtypes(include=np.number).columns)
        num_categorical = num_features - num_numeric
        imbalance_ratio = MetaFeatureExtractor._compute_imbalance(df, target)
        missing_ratio = df.isna().sum().sum() / (num_rows * num_features)

        return {
            "num_features": num_features,
            "num_rows": num_rows,
            "num_numeric": num_numeric,
            "num_categorical": num_categorical,
            "imbalance_ratio": imbalance_ratio,
            "missing_ratio": round(missing_ratio, 3)
        }

    @staticmethod
    def _compute_imbalance(df, target):
        try:
            value_counts = df[target].value_counts(normalize=True)
            return round(value_counts.min() / value_counts.max(), 3)
        except Exception:
            return 1.0


class MetaLogger:
    """Stores and retrieves past model performances"""
    
    @staticmethod
    def log_run(meta_features: dict, model_results: pd.DataFrame):
        """Append run data to meta log file with dataset+model uniqueness check"""

        os.makedirs(os.path.dirname(META_LOG_PATH), exist_ok=True)
        meta_records = []

        for _, row in model_results.iterrows():
            record = meta_features.copy()
        
            unique_string = json.dumps(meta_features, sort_keys=True) + row["model"]
            record["dataset_model_id"] = hashlib.md5(unique_string.encode()).hexdigest()
            record["model_name"] = row["model"]
            record["accuracy"] = row["accuracy"]
            meta_records.append(record)

        df_new = pd.DataFrame(meta_records)

        # If file exists load and check integrity
        if os.path.exists(META_LOG_PATH):
            df_old = pd.read_csv(META_LOG_PATH)
            if "dataset_model_id" not in df_old.columns:
                print("[MetaLogger] Old log missing dataset_model_id — reinitializing it.")
                df_old = pd.DataFrame(columns=df_new.columns)
            else:
                # Remove duplicates based on dataset_model_id
                df_old = df_old[~df_old["dataset_model_id"].isin(df_new["dataset_model_id"])]
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            print("[MetaLogger] Creating new meta log file.")
            df_all = df_new

        df_all.to_csv(META_LOG_PATH, index=False)
        print(f"Meta logs updated: {META_LOG_PATH} ({len(df_new)} new unique entries)")


    @staticmethod
    def load_logs():
        if not os.path.exists(META_LOG_PATH):
            print("No meta logs found yet")
            return pd.DataFrame()
        return pd.read_csv(META_LOG_PATH)
        

        
class MetaRecommender:
    """
    Learns from past runs, saves the result for future use, 
    and continuously improves as new data is logged
    """
    def __init__(self):
        self.meta_model = None
        self.last_trained_rows = 0
        self._load_model_if_exists()

    def _load_model_if_exists(self):
        """Load saved meta-model and its info if available"""
        if os.path.exists(META_MODEL_PATH):
            self.meta_model = joblib.load(META_MODEL_PATH)
            print(f"Loaded existing meta-model from {META_MODEL_PATH}")

            if os.path.exists(META_INFO_PATH):
                with open(META_INFO_PATH, "r") as f:
                    info = json.load(f)
                    self.last_trained_rows = info.get("last_trained_rows", 0)
                    print(f"Last trained on {self.last_trained_rows} meta-records.")
        else:
            print("No previous meta-model found. Starting from scratch.")

    def train_or_update(self, logs: pd.DataFrame):
        """Train or update the meta-model only when new logs appear"""
        if logs.empty:
            print("No logs available to train meta-model")
            return

        if len(logs) <= self.last_trained_rows:
            print("Meta-model already up to date. No new logs to learn from.")
            return

        print("Training or updating meta-model with new logs...")

        non_numeric_cols = [col for col in logs.columns if not np.issubdtype(logs[col].dtype, np.number)]
        if non_numeric_cols:
            print(f"Dropping non-numeric columns: {non_numeric_cols}")
            logs = logs.drop(columns=non_numeric_cols)

        X = logs.drop(columns=["accuracy"], errors="ignore")
        y = logs["accuracy"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        print(f"Meta-model trained. R² = {round(score, 3)}")

        # Save model and metadata
        joblib.dump(model, META_MODEL_PATH)
        self.meta_model = model
        self.last_trained_rows = len(logs)
        self._save_meta_info()
        print(f"Saved updated meta-model to {META_MODEL_PATH}")

    def _save_meta_info(self):
        info = {
            "last_trained_rows": self.last_trained_rows,
            "last_trained_at": datetime.now().isoformat()
        }
        with open(META_INFO_PATH, "w") as f:
            json.dump(info, f, indent=2)

    def predict_best_models(self, df: pd.DataFrame, target: str, model_list: list, extractor):
        """Use the trained meta-model to predict likely best models"""
        if self.meta_model is None:
            print("No trained meta-model found. Train it first using train_or_update()")
            return

        meta_features = extractor.extract(df, target)
        feature_df = pd.DataFrame([meta_features] * len(model_list))
        feature_df["model_name"] = model_list

        feature_df = feature_df.select_dtypes(include=[np.number])

        preds = self.meta_model.predict(feature_df)
        results = pd.DataFrame({
            "model_name": model_list,
            "predicted_accuracy": np.round(preds, 3)
        }).sort_values(by="predicted_accuracy", ascending=False)

        print("Predicted Best Models:")
        print(results)
        return results
