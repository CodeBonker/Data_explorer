import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any

from ml_model.model_registry import get_candidates

HIGH_CARDINALITY_THRESHOLD = 50
IMBALANCE_RATIO = 0.2

class ModelSelector:
    """
    Automatically determines the ML problem type and suggests models based on dataset characteristics
    """
    
    def __init__(self, df: pd.DataFrame, target: Optional[str] = None):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        self.df = df.copy()
        self.target = target
        self.analysis_results: Dict[str, Any] = {}
        self._analyzed = False

    
    def analyze(self) -> Dict[str, Any]:
        n_rows, n_cols = self.df.shape
        features = {}
        n_numeric = 0
        n_categorical = 0
        high_cardinality_cols = []

        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            n_unique = int(self.df[col].nunique(dropna=True))
            n_missing = int(self.df[col].isnull().sum())
            features[col] = {
                "dtype": dtype,
                "n_unique": n_unique,
                "n_missing": n_missing,
                "percent_missing": float(n_missing / max(1, n_rows))
            }
            if dtype.startswith("int") or dtype.startswith("float"):
                n_numeric += 1
            else:
                n_categorical += 1
            
            if n_unique >= HIGH_CARDINALITY_THRESHOLD:
                high_cardinality_cols.append(col)

        summary = {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "features": features,
            "n_numeric": n_numeric,
            "n_categorical": n_categorical,
            "high_cardinality_cols": high_cardinality_cols
        }

        # target ko analyze karenge
        if self.target is None or self.target not in self.df.columns:
            summary["problem_type_hint"] = "unsupervised"
            summary["reason"] = "No target column provided"
            self.analysis_results = summary
            self._analyzed = True
            return summary

        y = self.df[self.target]
        y_nunique = int(y.nunique(dropna=True))
        y_dtype = str(y.dtype)
        y_missing = int(y.isnull().sum())
        y_unique_ratio = y_nunique / max(1, n_rows)
        
        target_info = {
            "dtype": y_dtype,
            "n_unique": y_nunique,
            "n_missing": y_missing,
            "unique_ratio": y_unique_ratio
        }


        # Detect classification vs regression
        if pd.api.types.is_numeric_dtype(y):
            if y_nunique >= 20 or y_unique_ratio > 0.1:
                problem_type = "regression"
                reason = f"Numeric target with {y_nunique} unique values suggests regression."
            else:
                problem_type = "classification"
                reason = f"Numeric target with {y_nunique} unique values suggests classification."
        else:
            problem_type = "classification"
            reason = f"Non-numeric target dtype '{y_dtype}' suggests classification."

        summary["target"] = target_info
        summary["problem_type_hint"] = problem_type
        summary["problem_reason"] = reason

        if problem_type == "classification":
            value_counts = y.value_counts(dropna=True)
            class_counts = value_counts.to_dict()
            majority = max(class_counts.values()) if class_counts else 0
            minority = min(class_counts.values()) if class_counts else 0
            imbalance_ratio = (minority / majority) if majority > 0 else 0.0
            summary["class_info"] = {
                "n_classes": y_nunique,
                "class_counts": class_counts,
                "imbalance_ratio": float(imbalance_ratio),
                "is_binary": (y_nunique == 2),
                "is_imbalanced": (imbalance_ratio < IMBALANCE_RATIO)
            }

        # dataset signals for registry scoring
        dataset_signals = {
            "n_numeric": summary["n_numeric"],
            "n_categorical": summary["n_categorical"],
            "has_high_cardinality": len(summary["high_cardinality_cols"]) > 0,
            "is_imbalanced": summary.get("class_info", {}).get("is_imbalanced", False)
        }

        summary["dataset_signals"] = dataset_signals

        self.analysis_results = summary
        self._analyzed = True
        return summary
    

    def suggest_models(self, top_k: int = 5, speed_constraint: Optional[str] = None, prefer_interpretable: bool = False):
        """
        Use model registry to get candidate models based on analysis results and dataset signals
        """
        if not self._analyzed:
            self.analyze()

        problem_type = self.analysis_results.get("problem_type_hint", "unsupervised")
        dataset_signals = self.analysis_results.get("dataset_signals", {})

        candidates = get_candidates(
            problem_type = problem_type,
            dataset_signals = dataset_signals,
            top_k = top_k,
            speed_constraint = speed_constraint,
            prefer_interpretable = prefer_interpretable
        )
        return {
            "problem_type": problem_type,
            "candidates": candidates,
            "dataset_signals": dataset_signals
        }
        


    def summary(self) -> str:
        if not self._analyzed:
            self.analyze()
        s = self.analysis_results
        lines = []
        lines.append(f"Rows: {s.get('n_rows')}, Columns: {s.get('n_cols')}")
        if s.get("problem_type_hint") == "unsupervised":
            lines.append("Problem type: Unsupervised (clustering) — no target column provided.")
        else:
            lines.append(f"Problem type: {s.get('problem_type_hint')}")
            if s.get("target"):
                t = s["target"]
                lines.append(f"Target dtype: {t.get('dtype')}, unique values: {t.get('n_unique')}, missing: {t.get('n_missing')}")
            if s.get("class_info"):
                ci = s["class_info"]
                lines.append(f"Classes: {ci.get('n_classes')}, Imbalance ratio: {ci.get('imbalance_ratio'):.3f}")
            if s.get("problem_reason"):
                lines.append(f"Reason: {s.get('problem_reason')}")
        return "\n".join(lines)