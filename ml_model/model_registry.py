"""
Model Registry & Capability Engine

Provides:
 - MODEL_REGISTRY: dict of model metadata
 - get_candidates(): queryable function to fetch candidate models for a problem
 - register_model(): add custom models at runtime
 - get_model(): lookup by id
    
 author- @shikhar_navdeep
"""

from typing import Dict, Any, List, Optional
import copy

# each model has an id, display name, problem types it applies to,
# and capability flags. We can tweak/add metadata as needed

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "id": "logistic_regression",
        "name": "LogisticRegression",
        "problem_types": ["classification"],
        "handles_categorical": False,
        "needs_scaling": True,
        "supports_imbalance": True,
        "speed": "fast",
        "interpretability": "high",
        "notes": "Linear baseline; good for small/clean datasets; supports class_weight."
    },
    "random_forest": {
        "id": "random_forest",
        "name": "RandomForestClassifier/Regressor",
        "problem_types": ["classification", "regression"],
        "handles_categorical": True,
        "needs_scaling": False,
        "supports_imbalance": True,
        "speed": "medium",
        "interpretability": "medium",
        "notes": "Robust default; insensitive to scaling; good baseline."
    },
    "xgboost": {
        "id": "xgboost",
        "name": "XGBoost",
        "problem_types": ["classification", "regression"],
        "handles_categorical": False,
        "needs_scaling": False,
        "supports_imbalance": True,
        "speed": "medium",
        "interpretability": "low",
        "notes": "High performance boosted trees; fast training with GPU; handle missing values."
    },
    "lightgbm": {
        "id": "lightgbm",
        "name": "LightGBM",
        "problem_types": ["classification", "regression"],
        "handles_categorical": False,
        "needs_scaling": False,
        "supports_imbalance": True,
        "speed": "fast",
        "interpretability": "low",
        "notes": "Efficient for large datasets; supports categorical handling with special API."
    },
    "catboost": {
        "id": "catboost",
        "name": "CatBoost",
        "problem_types": ["classification", "regression"],
        "handles_categorical": True,
        "needs_scaling": False,
        "supports_imbalance": True,
        "speed": "medium",
        "interpretability": "low",
        "notes": "Handles categorical features natively; robust defaults."
    },
    "svc": {
        "id": "svc",
        "name": "SVC",
        "problem_types": ["classification"],
        "handles_categorical": False,
        "needs_scaling": True,
        "supports_imbalance": False,
        "speed": "slow",
        "interpretability": "low",
        "notes": "Kernel SVMs for complex boundaries; slow on large datasets."
    },
    "knn": {
        "id": "knn",
        "name": "KNN",
        "problem_types": ["classification", "regression"],
        "handles_categorical": False,
        "needs_scaling": True,
        "supports_imbalance": False,
        "speed": "slow",
        "interpretability": "low",
        "notes": "Instance-based; useful for small datasets and multimodal distributions."
    },
    "linear_regression": {
        "id": "linear_regression",
        "name": "LinearRegression",
        "problem_types": ["regression"],
        "handles_categorical": False,
        "needs_scaling": True,
        "supports_imbalance": False,
        "speed": "fast",
        "interpretability": "high",
        "notes": "Simple linear baseline; check assumptions before using."
    },
    "ridge_lasso": {
        "id": "ridge_lasso",
        "name": "Ridge/Lasso",
        "problem_types": ["regression"],
        "handles_categorical": False,
        "needs_scaling": True,
        "supports_imbalance": False,
        "speed": "fast",
        "interpretability": "medium",
        "notes": "Regularized linear models reduce overfitting."
    },
    "mlp": {
        "id": "mlp",
        "name": "NeuralNetwork (MLP)",
        "problem_types": ["classification", "regression"],
        "handles_categorical": False,
        "needs_scaling": True,
        "supports_imbalance": False,
        "speed": "slow",
        "interpretability": "low",
        "notes": "Flexible but needs hyperparameter tuning and scaling."
    },
    "kmeans": {
        "id": "kmeans",
        "name": "KMeans",
        "problem_types": ["clustering"],
        "handles_categorical": False,
        "needs_scaling": True,
        "supports_imbalance": False,
        "speed": "fast",
        "interpretability": "medium",
        "notes": "Simple clustering; requires numeric features and scaling."
    },
    "dbscan": {
        "id": "dbscan",
        "name": "DBSCAN",
        "problem_types": ["clustering"],
        "handles_categorical": False,
        "needs_scaling": True,
        "supports_imbalance": False,
        "speed": "medium",
        "interpretability": "medium",
        "notes": "Density-based clustering; finds noise and arbitrary shapes."
    },
    "agglomerative": {
        "id": "agglomerative",
        "name": "AgglomerativeClustering",
        "problem_types": ["clustering"],
        "handles_categorical": False,
        "needs_scaling": True,
        "supports_imbalance": False,
        "speed": "slow",
        "interpretability": "medium",
        "notes": "Hierarchical clustering; better for small datasets."
    }
}


# gonna act as an API which will help us to call without calling internal variables
def get_model(model_id: str) -> Optional[Dict[str, Any]]:
    """Return a deep copy of model metadata by id or None if it is missing/ not exist at all"""
    entry = MODEL_REGISTRY.get(model_id)
    return copy.deepcopy(entry) if entry is not None else None


def register_model(entry: Dict[str, Any]) -> None:
    """
    It registers a new model metadata entry
    Entry must contain at least 'id' and 'problem_types' otherwise raise an error
    """
    if "id" not in entry or "problem_types" not in entry:
        raise ValueError("Model entry must include 'id' and 'problem_types'.")
    MODEL_REGISTRY[entry["id"]] = copy.deepcopy(entry)


def _score_model_for_dataset(meta: Dict[str, Any], dataset_signals: Dict[str, Any]) -> float:
    """
    Give +1 for matching desirable capabilities, -1 for mismatches.
    dataset_signals example:
      {
        "n_numeric": 5,
        "n_categorical": 2,
        "has_high_cardinality": False,
        "is_imbalanced": True
      }
    """
    score = 0.0
    # prefer models that handle categorical if dataset has many categorical features
    n_cat = dataset_signals.get("n_categorical", 0)
    n_num = dataset_signals.get("n_numeric", 0)
    is_imbalanced = dataset_signals.get("is_imbalanced", False)
    has_high_cardinality = dataset_signals.get("has_high_cardinality", False)

    # categorical handling
    if n_cat > n_num and meta.get("handles_categorical", False):
        score += 2.0
    elif n_cat > n_num and not meta.get("handles_categorical", False):
        score -= 1.0

    # imbalance
    if is_imbalanced and meta.get("supports_imbalance", False):
        score += 1.5
    elif is_imbalanced and not meta.get("supports_imbalance", False):
        score -= 0.5

    # scaling requirement
    if meta.get("needs_scaling", False) and n_num == 0:
        score -= 1.0  # penalize models that need scaling if no numeric features

    # high cardinality sensitivity
    if has_high_cardinality and meta.get("handles_categorical", False) is False:
        # models that do not handle categorical well may struggle with high-cardinality after one-hot
        score -= 0.5


    return score

def get_candidates(     
    problem_type: str,
    dataset_signals: Dict[str, Any],
    top_k: int = 5,
    speed_constraint: Optional[str] = None,
    prefer_interpretable: bool = False
    ) -> List[Dict[str, Any]]:

    # It takes the information and based on that returns the best fit models

    """
    it is a query registry for candidate models
    - problem_type: 'classification'|'regression'|'clustering'
    - dataset_signals: dictionary of dataset characteristics (see _score_model_for_dataset)
    - speed_constraint: optional filter ('fast','medium','slow')
    - prefer_interpretable: if True, boost interpretable models
    Returns top_k entries ordered by score (plus tie-breakers)
    """
    candidates = []
    for meta in MODEL_REGISTRY.values():
        if problem_type in meta.get("problem_types", []):
            meta_copy = copy.deepcopy(meta)
            
            base_score = _score_model_for_dataset(meta_copy, dataset_signals)
          
            interpretability = meta_copy.get("interpretability", "low")
            if prefer_interpretable and interpretability in ("high", "medium"):
                base_score += 0.5
         
            if speed_constraint and meta_copy.get("speed") != speed_constraint:
                continue

            candidates.append((base_score, meta_copy))

    candidates.sort(key=lambda x: (x[0], x[1].get("speed", "medium"), x[1].get("interpretability", "low")), reverse=True)

    results = []
    for score, meta in candidates[:top_k]:
        entry = copy.deepcopy(meta)
        entry["_score"] = float(score)
        results.append(entry)

    return results
