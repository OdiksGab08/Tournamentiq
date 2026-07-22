"""Inspect the raw international-results file before the cleaning stage.

Purpose:
    Surface schema, completeness, duplication, and date-range problems in the
    raw source before it enters the persisted data pipeline.
Responsibility:
    Read and report diagnostics without changing or writing the input data.
Inputs:
    ``data/raw/international_results/results.csv``.
Outputs:
    Console diagnostics and a :class:`ValueError` for missing required columns.
Interactions:
    This is a manual preflight companion to ``clean_results.py``; it does not
    replace the cleaning or integrity-validation stages.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATASET = ROOT / "data" / "raw" / "international_results" / "results.csv"


REQUIRED_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]


def validate_results() -> None:
    """Print structural diagnostics for the configured raw-results source.

    Args:
        None.

    Returns:
        None.

    Raises:
        ValueError: If a column required by downstream processing is absent.

    Notes:
        The function remains read-only so operators can inspect source quality
        before deciding whether to run the cleaning pipeline.
    """

    print("=" * 60)
    print("VALIDATING DATASET")
    print("=" * 60)

    df = pd.read_csv(DATASET)

    print(f"Rows : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    print()

    # -----------------------------
    # Required Columns
    # -----------------------------

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    print("✓ Required columns present")

    # -----------------------------
    # Missing Values
    # -----------------------------

    print()

    print("Missing Values")

    print(df.isna().sum())

    # -----------------------------
    # Duplicate Matches
    # -----------------------------

    duplicates = df.duplicated().sum()

    print()

    print(f"Duplicate Rows : {duplicates}")

    # -----------------------------
    # Date Conversion
    # -----------------------------

    df["date"] = pd.to_datetime(df["date"])

    print()

    print(
        "Date Range:",
        df["date"].min(),
        "→",
        df["date"].max(),
    )

    print()

    print("Validation Complete")


if __name__ == "__main__":
    validate_results()
