import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_squared_error
)
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.preprocessing import LabelEncoder

MODEL_MAP = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "SVC": SVC,
    "KNN": KNeighborsClassifier,
    "NeuralNetwork (MLP)": MLPClassifier,
    "XGBoostClassifier": XGBClassifier,
    "XGBoost": XGBClassifier,  

    "LinearRegression": LinearRegression,
    "RandomForestRegressor": RandomForestRegressor,
    "SVR": SVR,
    "KNeighborsRegressor": KNeighborsRegressor,
    "NeuralNetworkRegressor (MLP)": MLPRegressor,
    "XGBoostRegressor": XGBRegressor,
    "RandomForestClassifier/Regressor": RandomForestClassifier 
}

class AutoTrainer:
    """
    Automatically trains and evaluates multiple ML models based on the problem type
    """

    def __init__(self, df: pd.DataFrame, target: str, problem_type: str, candidates: list):
        self.df = df.copy()
        self.target = target
        self.problem_type = problem_type
        self.candidates = candidates
        self.results = []

    def train_and_evaluate(self, test_size: float = 0.2, random_state: int = 42):
        X = self.df.drop(self.target, axis=1)
        y = self.df[self.target]

        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        # detect categorical columns
        if len(categorical_cols) > 0:
            print(f"Encoding {len(categorical_cols)} categorical columns...")
            encoder = LabelEncoder()
            for col in categorical_cols:
                try:
                    X[col] = encoder.fit_transform(X[col].astype(str))
                except Exception as e:
                    print(f"Warning: could not encode column '{col}' — {e}")
                    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"Starting AutoTrainer for {self.problem_type.upper()} models...")
        for model_entry in tqdm(self.candidates, desc="Training Models"):
            model_name = model_entry["name"]
            try:
                model_class = MODEL_MAP.get(model_name)
                if model_class is None:
                    print(f"Model {model_name} not found in MODEL_MAP. Skipping...")
                    continue

                model = model_class()
                start_time = time.time()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                end_time = time.time()

                metrics = self.evaluate_model(y_test, y_pred)
                metrics["model"] = model_name
                metrics["train_time_sec"] = round(end_time - start_time, 2)
                metrics["notes"] = model_entry.get("notes", "")
                self.results.append(metrics)

            except Exception as e:
                print(f" {model_name} failed: {e}")
                continue

        return self.leaderboard()
    

    def evaluate_model(self, y_test, y_pred):

        """Compute evaluation metrics for a model"""
        if self.problem_type == "classification":
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            return {
                "accuracy": round(acc, 3),
                "f1_score": round(f1, 3),
                "precision": round(prec, 3),
                "recall": round(rec, 3)
            }

        elif self.problem_type == "regression":
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            return {
                "mse": round(mse, 3),
                "r2_score": round(r2, 3)
            }
        
    
    def leaderboard(self):
        """Generate ranked leaderboard of all results"""
        if not self.results:
            print("No successful model results found")
            return None

        df = pd.DataFrame(self.results)
        if self.problem_type == "classification":
            df = df.sort_values(by="accuracy", ascending=False)
        else:
            df = df.sort_values(by="r2_score", ascending=False)

        df.reset_index(drop=True, inplace=True)
        print("Leaderboard:")
        print(df)
        return df
