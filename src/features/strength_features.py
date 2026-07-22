"""Build rolling attacking and defensive strength features for each team.

Purpose:
    Estimate recent scoring, conceding, clean-sheet, and scoring-failure rates
    before every historical match.
Responsibility:
    Maintain a bounded ten-match record for each team and emit aligned snapshots
    before updating that history.
Inputs:
    Chronologically ordered match DataFrame with home/away teams and scores.
Outputs:
    A DataFrame containing home and away strength measurements per input row.
Interactions:
    ``feature_pipeline`` merges these features with form and team statistics for
    model training.
"""

from collections import defaultdict, deque
import pandas as pd

WINDOW = 10


def build_strength_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Create ten-match rolling attack and defense features for each fixture.

    Args:
        matches: Chronologically ordered historical match records.

    Returns:
        A row-aligned DataFrame of rolling home and away strength features.

    Notes:
        Empty histories resolve to explicit zero values so early historical rows
        remain usable without relying on future fixtures.
    """

    history = defaultdict(lambda: deque(maxlen=WINDOW))

    records = []

    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        def summarize(games):
            """Summarize attack and defense measures for a recent game history.

            Args:
                games: Goal-for and goal-against dictionaries for one team.

            Returns:
                Average goals, goal difference, and defensive-rate metrics.

            Notes:
                Zero values for an empty history preserve feature-row alignment
                for teams appearing in the earliest chronological records.
            """

            if len(games) == 0:
                return {
                    "avg_gf": 0.0,
                    "avg_ga": 0.0,
                    "avg_gd": 0.0,
                    "clean_sheet_rate": 0.0,
                    "failed_to_score_rate": 0.0,
                }

            played = len(games)

            gf = sum(x["gf"] for x in games)

            ga = sum(x["ga"] for x in games)

            clean_sheets = sum(x["ga"] == 0 for x in games)

            failed_to_score = sum(x["gf"] == 0 for x in games)

            return {
                "avg_gf": gf / played,
                "avg_ga": ga / played,
                "avg_gd": (gf - ga) / played,
                "clean_sheet_rate": clean_sheets / played,
                "failed_to_score_rate": failed_to_score / played,
            }

        home_stats = summarize(history[home])

        away_stats = summarize(history[away])

        records.append(
            {
                "home_attack_strength": home_stats["avg_gf"],
                "home_defense_strength": home_stats["avg_ga"],
                "home_goal_difference": home_stats["avg_gd"],
                "home_clean_sheet_rate": home_stats["clean_sheet_rate"],
                "home_failed_to_score_rate": home_stats["failed_to_score_rate"],
                "away_attack_strength": away_stats["avg_gf"],
                "away_defense_strength": away_stats["avg_ga"],
                "away_goal_difference": away_stats["avg_gd"],
                "away_clean_sheet_rate": away_stats["clean_sheet_rate"],
                "away_failed_to_score_rate": away_stats["failed_to_score_rate"],
            }
        )

        history[home].append(
            {
                "gf": row["home_score"],
                "ga": row["away_score"],
            }
        )

        history[away].append(
            {
                "gf": row["away_score"],
                "ga": row["home_score"],
            }
        )

    return pd.DataFrame(records)
