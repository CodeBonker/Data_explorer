import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import json

META_LOG_PATH = "data/meta_logs.csv"

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

        """Append run data to meta log file""" 
        meta_records = []
        for _, row in model_results.iterrows():
            record = meta_features.copy()
            record["model_name"] = row["model"]
            record["accuracy"] = row["accuracy"]
            meta_records.append(record)

        df_new = pd.DataFrame(meta_records)

        if os.path.exists(META_LOG_PATH):
            df_old = pd.read_csv(META_LOG_PATH)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new

        df_all.to_csv(META_LOG_PATH, index=False)
        print(f" Meta logs updated: {META_LOG_PATH}")


        @staticmethod
        def load_logs():
            if not os.path.exists(META_LOG_PATH):
                print("No meta logs found yet")
                return pd.DataFrame()
            return pd.read_csv(META_LOG_PATH)
        

        
class MetaRecommender:
    """Learns from past runs and predicts which models might perform best."""
    
    def __init__(self):
        self.meta_model = RandomForestRegressor(random_state=42)


    def train_meta_model(self, logs: pd.DataFrame):
        if logs.empty:
            print("Not enough data to train meta-model yet")
            return
        
        X = logs.drop(columns=["model_name", "accuracy"])
        y = logs["accuracy"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.meta_model.fit(X_train, y_train)
        preds = self.meta_model.predict(X_test)
        score = r2_score(y_test, preds)
        print(f"Meta-model trained. R² Score: {round(score, 3)}")


    def predict_best_models(self, df: pd.DataFrame, target: str, model_list: list):
        meta_features = MetaFeatureExtractor.extract(df, target)
        feature_df = pd.DataFrame([meta_features] * len(model_list))
        feature_df["model_name"] = model_list
        preds = self.meta_model.predict(feature_df.drop(columns=["model_name"]))
        
        results = pd.DataFrame({
            "model_name": model_list,
            "predicted_accuracy": np.round(preds, 3)
        }).sort_values(by="predicted_accuracy", ascending=False)
        
        print("Predicted Best Models Based on Meta-Learning:")
        print(results)
        return results