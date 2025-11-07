import pandas as pd
import numpy as np

class ModelSelector:
    """
    Automatically determines the ML problem type and suggests models based on dataset characteristics
    """
    
    def __init__(self, df: pd.DataFrame, target: str = None):
        self.df = df.copy()
        self.target = target
        self.analysis = {}
        self.problem_type = None

    
    def analyze(self):
        """Analyze dataset structure, target type, and problem category"""

        n_rows, n_cols = self.df.shape
        info = {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "features": list(self.df.columns)
        }

        if self.target is None or self.target not in self.df.columns:
            info["problem_type"] = "unsupervised"
            info["reason"] = "No target column provided."
            self.analysis = info
            self.problem_type = "unsupervised"
            return info
        
        target_col = self.df[self.target]
        dtype = target_col.dtype
        unique_count = target_col.nunique()

        unique_ratio = unique_count / n_rows

        # Detect classification vs regression
        if pd.api.types.is_numeric_dtype(target_col):
            if unique_count <= 20 or unique_ratio < 0.05:
                problem_type = "classification"
                reason = f"Numeric target with only {unique_count} unique values -- likely classification"
            else:
                problem_type = "regression"
                reason = f"Numeric target with {unique_count} unique values -- likely regression"
        else:
            problem_type = "classification"
            reason = f"Non-numeric target dtype ({dtype}) -- classification task"


        info["target_dtype"] = str(dtype)
        info["unique_values"] = unique_count
        info["problem_type"] = problem_type
        info["reason"] = reason

        if problem_type == "classification":
            counts = target_col.value_counts().to_dict()
            imbalance_ratio = min(counts.values()) / max(counts.values())
            info["class_balance"] = imbalance_ratio
            info["is_imbalanced"] = imbalance_ratio < 0.2

        self.analysis = info
        self.problem_type = problem_type
        return info


    def suggest_models(self):
        """ suggest the ML models based on the detected problem type"""

        if self.problem_type is None:
            self.analyze()

        if self.problem_type == "unsupervised":
            return ['KMeans', 'DBSCAN', "AgglomerativeClustering" ]
        
        elif self.problem_type == "regression":
            return ["LinearRegression", "Ridge", "RandomForestRegressor", "XGBoostRegressor"]

        elif self.problem_type == "classification":
            return ["LogisticRegression", "RandomForestClassifier", "XGBoostClassifier", "SVC"]
        


    def summary(self):
        """Readable summary of the analysis"""
        if not self.analysis:
            self.analyze()

        print(f"\n Dataset Summary:")
        print(f"Rows: {self.analysis['n_rows']}, Columns: {self.analysis['n_cols']}")
        print(f"Target: {self.target}")
        print(f"Target dtype: {self.analysis.get('target_dtype', 'N/A')}")
        print(f"Detected Problem Type: {self.analysis['problem_type'].upper()}")
        print(f"Reason: {self.analysis['reason']}")

        if self.analysis["problem_type"] == "classification":
            print(f"Class Balance Ratio: {self.analysis.get('class_balance', 'N/A'):.3f}")
            print("Imbalanced Dataset:", self.analysis.get("is_imbalanced"))