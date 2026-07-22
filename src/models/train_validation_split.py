"""Partition the model-ready dataset into chronological train/validation/test sets.

Purpose:
    Create temporally ordered datasets for fitting, model selection, and held-out
    evaluation after the target and leakage exclusions have been applied.
Responsibility:
    Split ``ml_dataset.parquet`` in 70/15/15 order and persist each partition.
Inputs:
    ``data/processed/ml_dataset.parquet``.
Outputs:
    ``train.parquet``, ``validation.parquet``, and ``test.parquet`` under
    ``data/processed``.
Interactions:
    ``src.models.preprocess`` loads these exact filenames to fit the shared
    inference preprocessor and train model candidates.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data" / "processed" / "ml_dataset.parquet"

OUTPUT = ROOT / "data" / "processed"


def split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ordered model-ready records into 70/15/15 contiguous partitions.

    Args:
        df: Chronologically ordered prepared dataset.

    Returns:
        Train, validation, and test DataFrames in source order.

    Notes:
        No shuffle occurs here: source order is the established temporal order
        produced by the preceding feature-engineering pipeline.
    """

    n = len(df)

    train_end = int(n * 0.70)

    validation_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()

    validation = df.iloc[train_end:validation_end].copy()

    test = df.iloc[validation_end:].copy()

    return train, validation, test


def save_dataset(df: pd.DataFrame, filename: str) -> None:
    """Persist one chronological split using the established output directory.

    Args:
        df: Split DataFrame to write.
        filename: Destination filename relative to :data:`OUTPUT`.

    Returns:
        None.

    Notes:
        Filenames form the contract consumed by ``preprocess.load_datasets``.
    """

    df.to_parquet(
        OUTPUT / filename,
        index=False,
    )


def main() -> None:
    """Create persisted chronological model-development partitions.

    Args:
        None.

    Returns:
        None.

    Notes:
        Distribution reporting is diagnostic only; it does not rebalance rows
        because temporal ordering has priority over random sampling here.
    """

    print("=" * 60)
    print("TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 60)

    df = pd.read_parquet(INPUT)

    print(f"\nLoaded {len(df):,} matches")

    train, validation, test = split_dataset(df)

    save_dataset(train, "train.parquet")
    save_dataset(validation, "validation.parquet")
    save_dataset(test, "test.parquet")

    print("\nDatasets Saved")

    print(f"\nTrain      : {len(train):,}")
    print(f"Validation : {len(validation):,}")
    print(f"Test       : {len(test):,}")

    print("\nTrain Target Distribution")
    print(train["target"].value_counts())

    print("\nValidation Target Distribution")
    print(validation["target"].value_counts())

    print("\nTest Target Distribution")
    print(test["target"].value_counts())


if __name__ == "__main__":
    main()
