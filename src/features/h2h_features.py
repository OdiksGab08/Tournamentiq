"""Derive chronological head-to-head features for each team pairing.

Purpose:
    Capture prior meetings, wins, goals, and draws from the perspective of the
    current home and away teams.
Responsibility:
    Maintain one normalized history record per unordered team pair and emit a
    leakage-safe snapshot before each fixture.
Inputs:
    Chronologically ordered DataFrame containing team names and final scores.
Outputs:
    A row-aligned DataFrame of pair-history and orientation-aware rate features.
Interactions:
    ``feature_pipeline`` merges this output into the final training dataset;
    ``h2h_live`` calculates analogous runtime statistics.
"""

from collections import defaultdict
import pandas as pd


def build_head_to_head_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Create pre-match head-to-head features for each historical fixture.

    Args:
        matches: Chronologically ordered matches with team names and scores.

    Returns:
        A DataFrame aligned to ``matches`` with home- and away-oriented pair
        history features.

    Notes:
        Pair histories use an alphabetically sorted key, while emitted values
        are reoriented to the fixture's home and away teams.
    """

    history = defaultdict(
        lambda: {
            "matches": 0,
            "team1_wins": 0,
            "team2_wins": 0,
            "draws": 0,
            "team1_goals": 0,
            "team2_goals": 0,
        }
    )

    records = []

    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        # One normalized key prevents France–Spain and Spain–France from
        # accumulating separate histories while preserving fixture orientation.
        pair = tuple(sorted([home, away]))

        stats = history[pair]

        # Determine which team is team1/team2
        if pair[0] == home:
            home_wins = stats["team1_wins"]
            away_wins = stats["team2_wins"]

            home_goals = stats["team1_goals"]
            away_goals = stats["team2_goals"]

        else:
            home_wins = stats["team2_wins"]
            away_wins = stats["team1_wins"]

            home_goals = stats["team2_goals"]
            away_goals = stats["team1_goals"]

        records.append(
            {
                "h2h_matches": stats["matches"],
                "home_h2h_wins": home_wins,
                "away_h2h_wins": away_wins,
                "h2h_draws": stats["draws"],
                "home_h2h_goals": home_goals,
                "away_h2h_goals": away_goals,
                "home_h2h_win_rate": home_wins / stats["matches"]
                if stats["matches"] > 0
                else 0,
                "away_h2h_win_rate": away_wins / stats["matches"]
                if stats["matches"] > 0
                else 0,
            }
        )

        # ------------------------------------
        # Mutate history only after the snapshot to keep the feature row
        # available at the match's kickoff time.
        # ------------------------------------

        stats["matches"] += 1

        if pair[0] == home:
            stats["team1_goals"] += row["home_score"]
            stats["team2_goals"] += row["away_score"]

            if row["home_score"] > row["away_score"]:
                stats["team1_wins"] += 1

            elif row["home_score"] < row["away_score"]:
                stats["team2_wins"] += 1

            else:
                stats["draws"] += 1

        else:
            stats["team1_goals"] += row["away_score"]
            stats["team2_goals"] += row["home_score"]

            if row["away_score"] > row["home_score"]:
                stats["team1_wins"] += 1

            elif row["away_score"] < row["home_score"]:
                stats["team2_wins"] += 1

            else:
                stats["draws"] += 1

    return pd.DataFrame(records)
