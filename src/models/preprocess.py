"""Fit and persist the feature preprocessor used by training and inference.

Purpose:
    Transform mixed numerical and categorical match features into the matrix
    consumed by the candidate classifiers.
Responsibility:
    Load chronological split artifacts, fit transformations on training data
    only, transform validation/test data, and save the fitted preprocessor.
Inputs:
    ``train.parquet``, ``validation.parquet``, and ``test.parquet`` under
    ``data/processed``.
Outputs:
    Transformed matrices and ``models/preprocessor.pkl``.
Interactions:
    ``trainer`` uses the returned matrices; the runtime ``Predictor`` loads the
    same saved preprocessor before executing model inference.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"

MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the established chronological train, validation, and test artifacts.

    Args:
        None.

    Returns:
        A tuple in ``(train, validation, test)`` order.

    Notes:
        The split order is preserved because the trainer fits preprocessing only
        on the first table and evaluates on the later chronological periods.
    """

    train = pd.read_parquet(DATA_DIR / "train.parquet")

    validation = pd.read_parquet(DATA_DIR / "validation.parquet")

    test = pd.read_parquet(DATA_DIR / "test.parquet")

    return train, validation, test


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictor columns from the canonical training target.

    Args:
        df: Prepared split containing a ``target`` column.

    Returns:
        A tuple of feature DataFrame and target Series.

    Notes:
        The target name is shared with ``prepare_data.create_target`` and the
        model-class mapping used by dashboard inference.
    """

    X = df.drop(columns=["target"])

    y = df["target"]

    return X, y


def preprocess(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
):
    """Fit preprocessing on training data and transform every chronological split.

    Args:
        train: Earliest chronological split used to fit transformations.
        validation: Middle split transformed for model selection.
        test: Latest split transformed for downstream held-out diagnostics.

    Returns:
        A tuple of transformed train/validation/test matrices followed by their
        target vectors.

    Notes:
        Fitting only on ``train`` prevents validation and test distributions from
        leaking into scaling or categorical encoding decisions.
    """

    X_train, y_train = split_features(train)
    X_validation, y_validation = split_features(validation)
    X_test, y_test = split_features(test)

    categorical_columns = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_columns = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    # Fit once on the earliest split; later periods must only be transformed to
    # preserve the same temporal boundary used by model evaluation.
    X_train = preprocessor.fit_transform(X_train)

    X_validation = preprocessor.transform(X_validation)

    X_test = preprocessor.transform(X_test)

    joblib.dump(
        preprocessor,
        MODEL_DIR / "preprocessor.pkl",
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


def main() -> None:
    """Run preprocessing and report transformed split dimensions.

    Args:
        None.

    Returns:
        None. The fitted preprocessor is persisted to :data:`MODEL_DIR`.

    Notes:
        This script validates artifact availability and is also invoked by the
        model trainer before candidate estimators are fitted.
    """

    train, validation, test = load_datasets()

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = preprocess(
        train,
        validation,
        test,
    )

    print()

    print("=" * 60)

    print("PREPROCESSING COMPLETE")

    print("=" * 60)

    print()

    print("Training :", X_train.shape)

    print("Validation :", X_validation.shape)

    print("Testing :", X_test.shape)

    print()

    print("Preprocessor Saved")


if __name__ == "__main__":
    main()
