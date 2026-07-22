"""Clean raw international-result records for downstream feature engineering.

Purpose:
    Produce a chronological, structurally valid match table from the raw
    international-results export.
Responsibility:
    Remove unusable records, normalize basic text fields, and persist the
    cleaned table without deriving modelling features.
Inputs:
    ``data/raw/international_results/international_results.csv``.
Outputs:
    ``data/processed/matches_clean.parquet``.
Interactions:
    The country and tournament standardization stages consume this artifact
    before warehouse construction and feature engineering.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    ROOT / "data" / "raw" / "international_results" / "international_results.csv"
)

OUTPUT_FILE = ROOT / "data" / "processed" / "matches_clean.parquet"


def clean_results() -> None:
    """Create the cleaned historical-results artifact.

    Args:
        None.

    Returns:
        None. The cleaned match table is written to :data:`OUTPUT_FILE`.

    Notes:
        Validation happens before text normalization so missing or impossible
        values cannot be converted into apparently valid display strings.
    """

    print("=" * 60)
    print("CLEANING INTERNATIONAL RESULTS")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    print(f"Original rows: {len(df):,}")

    # -----------------------------------------------------
    # Remove duplicates before applying transformations so exact source repeats
    # cannot become distinct rows after text normalization.
    # -----------------------------------------------------

    df = df.drop_duplicates()

    # -----------------------------------------------------
    # Keep only fields required by every later data and feature stage.
    # -----------------------------------------------------

    required = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "country",
        "neutral",
    ]

    df = df.dropna(subset=required)

    # -----------------------------------------------------
    # Remove impossible scores
    # -----------------------------------------------------

    df = df[(df.home_score >= 0) & (df.away_score >= 0)]

    # -----------------------------------------------------
    # Remove same-team matches
    # -----------------------------------------------------

    df = df[df.home_team != df.away_team]

    # -----------------------------------------------------
    # Clean text columns
    # -----------------------------------------------------

    text_columns = [
        "home_team",
        "away_team",
        "tournament",
        "city",
        "country",
    ]

    for column in text_columns:
        df[column] = df[column].astype(str).str.strip().str.title()

    # -----------------------------------------------------
    # Convert date
    # -----------------------------------------------------

    df["date"] = pd.to_datetime(df["date"])

    # -----------------------------------------------------
    # Sort
    # -----------------------------------------------------

    df = df.sort_values("date")

    df = df.reset_index(drop=True)

    print(f"Clean rows: {len(df):,}")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(f"Saved to\n{OUTPUT_FILE}")


if __name__ == "__main__":
    clean_results()
