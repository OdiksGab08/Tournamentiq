"""Canonicalize tournament labels after country standardization.

Purpose:
    Collapse equivalent competition names into the labels expected by feature
    engineering and tournament-history calculations.
Responsibility:
    Apply :data:`TOURNAMENT_MAPPING` to the tournament field and persist the
    resulting intermediate dataset.
Inputs:
    ``data/processed/matches_standardized.parquet``.
Outputs:
    ``data/processed/matches_standardized_v2.parquet``.
Interactions:
    The data-integrity stage validates this artifact before it becomes the
    warehouse master dataset.
"""

from pathlib import Path

import pandas as pd

if __package__:
    # Package execution is the supported pipeline path. Direct script execution
    # retains the local import used by the existing data-maintenance workflow.
    from .tournament_mapping import TOURNAMENT_MAPPING
else:  # pragma: no cover - exercised only by direct script runs.
    from tournament_mapping import TOURNAMENT_MAPPING

ROOT = Path(__file__).resolve().parents[3]

INPUT = ROOT / "data" / "processed" / "matches_standardized.parquet"

OUTPUT = ROOT / "data" / "processed" / "matches_standardized_v2.parquet"


def standardize_tournaments() -> None:
    """Write a tournament-standardized version of the match table.

    Args:
        None.

    Returns:
        None. The canonicalized parquet artifact is written to :data:`OUTPUT`.

    Notes:
        The operation intentionally changes only the tournament label; match
        outcomes and other source facts remain untouched.
    """

    print("=" * 60)
    print("STANDARDIZING TOURNAMENTS")
    print("=" * 60)

    df = pd.read_parquet(INPUT)

    df["tournament"] = df["tournament"].str.strip().replace(TOURNAMENT_MAPPING)

    df.to_parquet(
        OUTPUT,
        index=False,
    )

    print()

    print("Tournament names standardized.")

    print()

    print(f"Saved -> {OUTPUT}")


if __name__ == "__main__":
    standardize_tournaments()
