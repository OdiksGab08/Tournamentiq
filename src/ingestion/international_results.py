"""Load the retained raw international-results source into a DataFrame.

Purpose:
    Provide a small reusable adapter for the primary historical match CSV.
Responsibility:
    Verify source availability and parse the date column without cleaning or
    changing source values.
Inputs:
    ``data/raw/international_results/international_results.csv``.
Outputs:
    A pandas DataFrame with parsed match dates.
Interactions:
    Cleaning, live head-to-head, and exploratory workflows can reuse this raw
    data contract before downstream standardization.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "international_results"
    / "international_results.csv"
)


def load_international_results() -> pd.DataFrame:
    """Load raw international football results with parsed dates.

    Args:
        None.

    Returns:
        The source CSV as a pandas DataFrame with a datetime ``date`` column.

    Raises:
        FileNotFoundError: If the configured raw-results CSV is unavailable.

    Notes:
        This function intentionally performs no normalization so downstream
        cleaning can retain a clear boundary between source and derived data.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found:\n{DATA_PATH}")

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
    )

    return df
