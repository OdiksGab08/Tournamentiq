"""Split an earlier historical training dataset into chronological partitions.

Purpose:
    Preserve a legacy time-based split workflow for a dataset that retains its
    date column.
Responsibility:
    Load the configured dataset, order it by date, create 80/10/10 partitions,
    and write the resulting parquet files.
Inputs:
    ``data/processed/training_dataset.parquet``.
Outputs:
    Train, validation, and test parquet artifacts under ``data/processed``.
Interactions:
    This legacy script is distinct from ``train_validation_split.py``, which
    partitions the current model-ready dataset after target preparation.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATASET = ROOT / "data" / "processed" / "training_dataset.parquet"


def load_dataset() -> pd.DataFrame:
    """Load the configured legacy training dataset.

    Args:
        None.

    Returns:
        The parquet table located at :data:`DATASET`.

    Notes:
        The date column is intentionally retained here because this workflow
        performs its chronological ordering internally.
    """

    return pd.read_parquet(DATASET)


def chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition a date-sorted dataset into 80/10/10 chronological slices.

    Args:
        df: Historical records containing a parseable ``date`` column.

    Returns:
        Train, validation, and test DataFrames in chronological order.

    Notes:
        Rows are never randomly shuffled because future matches must not inform
        model selection or evaluation on earlier periods.
    """

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    n = len(df)

    train_end = int(n * 0.80)

    valid_end = int(n * 0.90)

    train = df.iloc[:train_end]

    validation = df.iloc[train_end:valid_end]

    test = df.iloc[valid_end:]

    return train, validation, test


def save_split(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Persist the legacy chronological partitions to processed-data paths.

    Args:
        train: Earliest chronological partition.
        validation: Middle chronological partition.
        test: Latest chronological partition.

    Returns:
        None.

    Notes:
        Output filenames match the names consumed by the preprocessing module.
    """

    output = ROOT / "data" / "processed"

    train.to_parquet(
        output / "train.parquet",
        index=False,
    )

    validation.to_parquet(
        output / "validation.parquet",
        index=False,
    )

    test.to_parquet(
        output / "test.parquet",
        index=False,
    )


def main() -> None:
    """Run the legacy chronological-splitting workflow and print date ranges.

    Args:
        None.

    Returns:
        None.

    Notes:
        This command is retained for the legacy source-dataset contract; the
        current preparation path uses ``train_validation_split.py``.
    """

    print()

    print("Loading Dataset...")

    df = load_dataset()

    train, validation, test = chronological_split(df)

    save_split(
        train,
        validation,
        test,
    )

    print()

    print(f"Training     : {train.shape}")

    print(f"Validation   : {validation.shape}")

    print(f"Testing      : {test.shape}")

    print()

    print("Date Range")

    print("----------------------------")

    print(
        "Train:",
        train.date.min(),
        "->",
        train.date.max(),
    )

    print(
        "Validation:",
        validation.date.min(),
        "->",
        validation.date.max(),
    )

    print(
        "Test:",
        test.date.min(),
        "->",
        test.date.max(),
    )


if __name__ == "__main__":
    main()
