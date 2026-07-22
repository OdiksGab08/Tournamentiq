"""Promote verified standardized matches into the central warehouse dataset.

Purpose:
    Create the chronological match table from which all feature-engineering
    stages begin.
Responsibility:
    Sort verified records, assign a stable sequential match identifier, and
    persist the warehouse artifact.
Inputs:
    ``data/processed/matches_verified.parquet``.
Outputs:
    ``data/warehouse/master_matches.parquet``.
Interactions:
    ``src.features.feature_pipeline`` consumes this output to build the final
    training dataset, and standalone feature scripts use the same source.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data" / "processed" / "matches_verified.parquet"

OUTPUT = ROOT / "data" / "warehouse" / "master_matches.parquet"


def build_master_dataset() -> None:
    """Build and save the canonical chronologically ordered warehouse table.

    Args:
        None.

    Returns:
        None. The master match table is written to :data:`OUTPUT`.

    Notes:
        The sequential identifier is assigned after chronological ordering so it
        remains a stable ordering aid rather than a source-data identifier.
    """

    print("=" * 60)
    print("BUILDING MASTER DATASET")
    print("=" * 60)

    df = pd.read_parquet(INPUT)

    # ------------------------------------------
    # Feature builders iterate statefully, so the warehouse establishes one
    # canonical chronological order before any derived values are calculated.
    # ------------------------------------------

    df = df.sort_values("date")

    # ------------------------------------------
    # Reset Index
    # ------------------------------------------

    df = df.reset_index(drop=True)

    # ------------------------------------------
    # Assign after sorting so identifiers reflect pipeline order consistently.
    # ------------------------------------------

    df.insert(
        0,
        "match_id",
        range(1, len(df) + 1),
    )

    # ------------------------------------------
    # Save
    # ------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT,
        index=False,
    )

    print()

    print("Master dataset created successfully.")

    print()

    print(f"Rows : {len(df):,}")

    print(f"Columns : {len(df.columns)}")

    print()

    print(f"Saved -> {OUTPUT}")


if __name__ == "__main__":
    build_master_dataset()
