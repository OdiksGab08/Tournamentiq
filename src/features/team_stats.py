"""Build rolling per-team statistics before each historical fixture.

Purpose:
    Capture a team's cumulative match record, results, and goals available at
    kickoff for every historical match.
Responsibility:
    Maintain mutable team records, emit pre-match home/away snapshots, and
    optionally persist the standalone statistics table.
Inputs:
    Chronological warehouse matches with team names and final scores.
Outputs:
    An aligned statistics DataFrame and, when run as a script,
    ``data/features/team_statistics.parquet``.
Interactions:
    ``feature_pipeline`` uses :func:`build_team_statistics` as one of the core
    sources for the final model-training dataset.
"""

from pathlib import Path

import pandas as pd


# =========================================================
# Paths
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "warehouse" / "master_matches.parquet"

OUTPUT_FILE = ROOT / "data" / "features" / "team_statistics.parquet"


# =========================================================
# Load Matches
# =========================================================


def load_matches() -> pd.DataFrame:
    """Load and sort the warehouse matches used by the standalone script.

    Args:
        None.

    Returns:
        Historical matches with a parsed, ascending ``date`` column.

    Notes:
        Chronological order is required because team records are updated in
        sequence and represent only information known before each fixture.
    """

    df = pd.read_parquet(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    return df


# =========================================================
# Initialize Team Record
# =========================================================


def initialize_team() -> dict[str, int]:
    """Create the mutable zero-state record for one national team.

    Args:
        None.

    Returns:
        A dictionary containing cumulative results and goal counters.

    Notes:
        The shape deliberately mirrors the fields emitted for both home and away
        teams in :func:`build_team_statistics`.
    """

    return {
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
    }


# =========================================================
# Update Team Record
# =========================================================


def update_team(team: dict[str, int], goals_for: int, goals_against: int) -> None:
    """Apply one completed fixture to a team's cumulative record.

    Args:
        team: Mutable cumulative record returned by :func:`initialize_team`.
        goals_for: Goals scored by the team in the completed fixture.
        goals_against: Goals conceded by the team in the completed fixture.

    Returns:
        None. ``team`` is updated in place.

    Notes:
        Mutation occurs only after the fixture's feature snapshot has been
        saved, which maintains the training pipeline's no-leakage boundary.
    """

    team["played"] += 1

    team["goals_for"] += goals_for

    team["goals_against"] += goals_against

    if goals_for > goals_against:
        team["wins"] += 1

    elif goals_for < goals_against:
        team["losses"] += 1

    else:
        team["draws"] += 1


# =========================================================
# Build Rolling Statistics
# =========================================================


def build_team_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Create pre-match cumulative team statistics for every input fixture.

    Args:
        df: Chronologically ordered matches with home/away teams and scores.

    Returns:
        A row-aligned DataFrame of home and away cumulative statistics.

    Notes:
        Team records are captured before their current match is applied, so a
        result never contributes to its own prediction features.
    """

    teams = {}

    records = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        if home not in teams:
            teams[home] = initialize_team()

        if away not in teams:
            teams[away] = initialize_team()

        home_stats = teams[home]
        away_stats = teams[away]

        # -----------------------------------------
        # Snapshot before mutation so the output represents pre-kickoff facts,
        # not results that became known after the match finished.
        # -----------------------------------------

        record = {
            # -------------------------
            # HOME TEAM
            # -------------------------
            "home_matches_played": home_stats["played"],
            "home_wins": home_stats["wins"],
            "home_draws": home_stats["draws"],
            "home_losses": home_stats["losses"],
            "home_goals_for": home_stats["goals_for"],
            "home_goals_against": home_stats["goals_against"],
            # -------------------------
            # AWAY TEAM
            # -------------------------
            "away_matches_played": away_stats["played"],
            "away_wins": away_stats["wins"],
            "away_draws": away_stats["draws"],
            "away_losses": away_stats["losses"],
            "away_goals_for": away_stats["goals_for"],
            "away_goals_against": away_stats["goals_against"],
        }
        records.append(record)

        # -----------------------------------------
        # Update both team records only after their shared fixture snapshot.
        # -----------------------------------------

        update_team(
            home_stats,
            row["home_score"],
            row["away_score"],
        )

        update_team(
            away_stats,
            row["away_score"],
            row["home_score"],
        )

    return pd.DataFrame(records)


# =========================================================
# Save
# =========================================================


def save_statistics(df: pd.DataFrame) -> None:
    """Persist a standalone rolling-statistics table for inspection or reuse.

    Args:
        df: Row-aligned team-statistics output from
            :func:`build_team_statistics`.

    Returns:
        None. The parquet file is written to :data:`OUTPUT_FILE`.

    Notes:
        The main feature pipeline builds statistics in memory, while this helper
        supports an explicit artifact for diagnostics and offline analysis.
    """

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()

    print("Rolling Team Statistics Saved")

    print(OUTPUT_FILE)


# =========================================================
# Main
# =========================================================


def main() -> None:
    """Run the standalone rolling-team-statistics generation workflow.

    Args:
        None.

    Returns:
        None.

    Notes:
        This command-line entry point mirrors the statistics component used by
        the full feature pipeline without changing its calculation contract.
    """

    print()

    print("Loading Matches...")

    matches = load_matches()

    print(f"Loaded {len(matches):,} matches")

    print()

    print("Building Rolling Statistics...")

    stats = build_team_statistics(matches)

    save_statistics(stats)

    print()

    print(stats.head())

    print()

    print(f"Rows: {len(stats):,}")

    print(f"Columns: {stats.shape[1]}")


if __name__ == "__main__":
    main()
