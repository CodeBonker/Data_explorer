import pandas as pd
import numpy as np
from scipy import stats

class DataQualityAnalyzer:
    """Evaluates dataset health and returns an interpretable report with a score (0-100)"""

    def __init__(self, df: pd.DataFrame, target: str = None):
        self.df = df
        self.target = target
        self.report = {}


    def compute_missing_score(self):
        total_cells = self.df.size
        total_missing = self.df.isnull().sum().sum()
        missing_ratio = total_missing / total_cells
        score = max(0, 100 - (missing_ratio * 100 * 2))
        self.report["missing_ratio"] = round(missing_ratio * 100, 2)
        return np.clip(score, 0, 100)
    

    def compute_dtype_score(self):
        inconsitent_cols =[]

        for col in self.df.columns:
            types = self.df[col].dropna().map(type).value_counts()
            if len(types) > 1:
                inconsitent_cols.append(col)
            
        ratio = len(inconsitent_cols) / len(self.df.columns)
        score = max(0, 100 - (ratio * 100 * 3))
        self.report["incosistent_columns"] = inconsitent_cols
        return np.clip(score, 0, 100)
    

    def compute_balance_score(self):
        if self.target is None or self.target not in self.df.columns:
            return 100
        
        vc = self.df[self.target].value_counts(normalize=True)
        if len(vc) <= 1:
            return 30
        imbalance = vc.max()
        score = max(0, 100 - (imbalance - 0.5) * 200)
        self.report["imbalance_ratio"] = round(imbalance, 2)
        return np.clip(score, 0, 100)
    

    def compute_outlier_score(self):
        numeric = self.df.select_dtypes(include = np.number)
        if numeric.empty:
            return 100
        
        z_scores= np.abs(stats.zscore(numeric, nan_policy= "omit"))
        outlier_ratio = (z_scores > 3).sum().sum() / numeric.size
        score = max(0, 100 - outlier_ratio * 400)
        self.report["outlier_ratio"] = round(outlier_ratio * 100, 2)
        return np.clip(score, 0, 100)
    

    def compute_variance_score(self):
        numeric = self.df.select_dtypes(include = np.number)
        if numeric.empty:
            return 100
        low_var_cols = [c for c in numeric.columns if numeric[c].nunique() <= 1]
        ratio = len(low_var_cols) / len(numeric.columns)
        score = max(0, 100 - ratio * 200)
        self.report["low_variance_cols"] = low_var_cols
        return np.clip(score, 0, 100)

 
    # combining all criterias:
    def calculate_health_score(self):
        s_missing = self.compute_missing_score()
        s_dtype = self.compute_dtype_score()
        s_balance = self.compute_balance_score()
        s_outlier = self.compute_outlier_score()
        s_var = self.compute_variance_score()

        weights = {
            "missing": 0.25,
            "dtype": 0.2,
            "balance": 0.2,
            "outlier": 0.2,
            "variance": 0.15,
        }

        final_score = (
            s_missing * weights["missing"]
            + s_dtype * weights["dtype"]
            + s_balance * weights["balance"]
            + s_outlier * weights["outlier"]
            + s_var * weights["variance"]
        )

        self.report["scores"] = {
            "missing": round(s_missing, 2),
            "dtype": round(s_dtype, 2),
            "balance": round(s_balance, 2),
            "outlier": round(s_outlier, 2),
            "variance": round(s_var, 2),
            "final_score": round(final_score, 2),
        }
        
        return round(final_score, 2), self.report