"""Validate standardized matches and publish a transparent data-quality report.

Purpose:
    Record key integrity signals for the standardized historical match dataset
    before it is promoted into the warehouse.
Responsibility:
    Calculate non-destructive diagnostics, write the report, and persist the
    verified dataset used by the warehouse builder.
Inputs:
    ``data/processed/matches_standardized_v2.parquet``.
Outputs:
    ``data/processed/matches_verified.parquet`` and
    ``reports/data_quality_report.txt``.
Interactions:
    ``src.warehouse.build_master_dataset`` consumes the verified artifact;
    report consumers can review quality without altering model inputs.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

INPUT = ROOT / "data" / "processed" / "matches_standardized_v2.parquet"

OUTPUT = ROOT / "data" / "processed" / "matches_verified.parquet"

REPORT = ROOT / "reports" / "data_quality_report.txt"


def validate_dataset() -> None:
    """Write integrity diagnostics and the verified dataset artifact.

    Args:
        None.

    Returns:
        None. Diagnostics are written to :data:`REPORT` and the input records
        are persisted to :data:`OUTPUT`.

    Notes:
        The routine reports quality signals instead of silently dropping rows.
        That preserves traceability for later data-quality investigation.
    """

    print("=" * 60)
    print("DATA INTEGRITY CHECK")
    print("=" * 60)

    df = pd.read_parquet(INPUT)

    report = []

    report.append("=" * 60)
    report.append("WORLD CUP PREDICTOR")
    report.append("DATA QUALITY REPORT")
    report.append("=" * 60)
    report.append("")

    # -------------------------------------------------
    # Keep diagnostics in a durable text report so quality review does not
    # depend on a transient console session.
    # -------------------------------------------------

    report.append(f"Rows: {len(df):,}")
    report.append(f"Columns: {len(df.columns)}")
    report.append("")

    # -------------------------------------------------
    # Missing Values
    # -------------------------------------------------

    missing = df.isna().sum()

    report.append("Missing Values")

    if missing.sum() == 0:
        report.append("None")

    else:
        report.append(str(missing[missing > 0]))

    report.append("")

    # -------------------------------------------------
    # Duplicate Rows
    # -------------------------------------------------

    duplicates = df.duplicated().sum()

    report.append(f"Duplicate Rows: {duplicates}")

    # -------------------------------------------------
    # Negative Scores
    # -------------------------------------------------

    negative_scores = ((df.home_score < 0) | (df.away_score < 0)).sum()

    report.append(f"Negative Scores: {negative_scores}")

    # -------------------------------------------------
    # Same Team
    # -------------------------------------------------

    same_team = (df.home_team == df.away_team).sum()

    report.append(f"Home = Away Team: {same_team}")

    # -------------------------------------------------
    # Future Dates
    # -------------------------------------------------

    future_dates = (df.date > pd.Timestamp.today()).sum()

    report.append(f"Future Dates: {future_dates}")

    # -------------------------------------------------
    # Empty Team Names
    # -------------------------------------------------

    empty_home = (df.home_team.str.strip() == "").sum()

    empty_away = (df.away_team.str.strip() == "").sum()

    report.append(f"Empty Home Team: {empty_home}")
    report.append(f"Empty Away Team: {empty_away}")

    # -------------------------------------------------
    # Neutral Field
    # -------------------------------------------------

    invalid_neutral = (~df["neutral"].isin([True, False])).sum()

    report.append(f"Invalid Neutral Values: {invalid_neutral}")

    report.append("")
    report.append("=" * 60)

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    df.to_parquet(
        OUTPUT,
        index=False,
    )

    print()

    print("Validation complete.")

    print()

    print(f"Verified dataset saved -> {OUTPUT}")

    print(f"Report saved -> {REPORT}")


if __name__ == "__main__":
    validate_dataset()
