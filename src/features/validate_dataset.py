"""Print validation diagnostics for the final engineered training dataset.

Purpose:
    Provide a lightweight human-readable preflight check for the feature output.
Responsibility:
    Read the final training parquet and report shape, duplication, missingness,
    data types, and score-derived outcome counts without modifying data.
Inputs:
    ``data/processed/final_training_dataset.parquet``.
Outputs:
    Console diagnostics only.
Interactions:
    Operators can run this script before preparing targets and chronological
    splits in ``src.models``.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data" / "processed" / "final_training_dataset.parquet"


def main() -> None:
    """Print structural and outcome diagnostics for the engineered dataset.

    Args:
        None.

    Returns:
        None.

    Notes:
        This script is deliberately read-only so it can be used as a safe
        checkpoint between feature generation and model preparation.
    """

    df = pd.read_parquet(DATA)

    print("=" * 60)
    print("FINAL DATASET REPORT")
    print("=" * 60)

    print("\nShape")
    print(df.shape)

    print("\nDuplicate Rows")
    print(df.duplicated().sum())

    print("\nDuplicate Columns")
    print(df.columns.duplicated().sum())

    print("\nMissing Values")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    print("\nData Types")
    print(df.dtypes.value_counts())

    print("\nTarget Distribution")

    home = (df.home_score > df.away_score).sum()
    draw = (df.home_score == df.away_score).sum()
    away = (df.home_score < df.away_score).sum()

    print(f"Home Wins : {home:,}")
    print(f"Draws     : {draw:,}")
    print(f"Away Wins : {away:,}")

    print("\nColumns")

    for c in df.columns:
        print(c)


if __name__ == "__main__":
    main()
