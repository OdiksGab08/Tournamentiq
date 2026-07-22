"""Standardize country and team names in the cleaned match dataset.

Purpose:
    Ensure a single national-team identity is used across historical records.
Responsibility:
    Apply :data:`COUNTRY_MAPPING` to team and host-country columns, then save a
    canonical intermediate artifact.
Inputs:
    ``data/processed/matches_clean.parquet``.
Outputs:
    ``data/processed/matches_standardized.parquet``.
Interactions:
    Tournament standardization reads this output before integrity validation and
    feature construction.
"""

from pathlib import Path

import pandas as pd

if __package__:
    # Package execution is the supported pipeline path. Direct script execution
    # retains the local import used by the existing data-maintenance workflow.
    from .country_mapping import COUNTRY_MAPPING
else:  # pragma: no cover - exercised only by direct script runs.
    from country_mapping import COUNTRY_MAPPING

ROOT = Path(__file__).resolve().parents[3]

INPUT = ROOT / "data" / "processed" / "matches_clean.parquet"

OUTPUT = ROOT / "data" / "processed" / "matches_standardized.parquet"


COUNTRY_COLUMNS = [
    "home_team",
    "away_team",
    "country",
]


def standardize() -> None:
    """Write a country-standardized version of the cleaned match table.

    Args:
        None.

    Returns:
        None. The standardized parquet artifact is written to :data:`OUTPUT`.

    Notes:
        Only identity columns are changed so score, date, and venue facts remain
        exactly as supplied by the cleaning stage.
    """

    print("=" * 60)
    print("STANDARDIZING COUNTRY NAMES")
    print("=" * 60)

    df = pd.read_parquet(INPUT)

    for column in COUNTRY_COLUMNS:
        df[column] = df[column].replace(COUNTRY_MAPPING).str.strip()

    df.to_parquet(
        OUTPUT,
        index=False,
    )

    print()

    print("Countries standardized successfully.")

    print()

    print(f"Saved -> {OUTPUT}")


if __name__ == "__main__":
    standardize()
