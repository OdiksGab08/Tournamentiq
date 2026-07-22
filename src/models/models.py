"""Construct the candidate classifiers evaluated by the training workflow.

Purpose:
    Provide a consistent, reproducible set of model families for comparison.
Responsibility:
    Instantiate unfitted scikit-learn, XGBoost, LightGBM, and CatBoost
    classifiers using shared configuration values.
Inputs:
    Hyperparameter constants from ``src.models.model_config``.
Outputs:
    A name-to-estimator mapping consumed by the model trainer.
Interactions:
    ``trainer.train_all_models`` fits, evaluates, and persists each returned
    estimator before selecting a production model.
"""

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.models.model_config import (
    RANDOM_STATE,
    N_ESTIMATORS,
    MAX_DEPTH,
    MAX_ITER,
    N_JOBS,
)


def get_models() -> dict[str, object]:
    """Return fresh, unfitted candidate classifiers for model comparison.

    Args:
        None.

    Returns:
        A mapping from stable display names to configured estimator instances.

    Notes:
        New instances are created on every call so training runs cannot reuse
        fitted state from an earlier candidate evaluation.
    """

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE,
            n_estimators=300,
            eval_metric="mlogloss",
            tree_method="hist",
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            random_state=RANDOM_STATE,
            n_estimators=300,
            verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            random_state=RANDOM_STATE,
            verbose=False,
        ),
    }
