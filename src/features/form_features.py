"""Derive rolling recent-form features from historical match results.

Purpose:
    Capture each team's results, points, and goal output across its latest
    fixtures before every match.
Responsibility:
    Maintain bounded per-team match histories and emit aligned pre-match form
    features.
Inputs:
    Chronologically ordered match DataFrame with team and score columns.
Outputs:
    A DataFrame with home and away form statistics for each input row.
Interactions:
    ``feature_pipeline`` combines these features with team, strength, and
    tournament signals for model training.
"""

from collections import defaultdict, deque

import pandas as pd


WINDOW = 5


def build_form_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Create last-five-match form features before each historical fixture.

    Args:
        matches: Chronologically ordered matches containing home/away teams and
            final scores.

    Returns:
        A row-aligned DataFrame of home and away form statistics.

    Notes:
        Histories are updated only after a record is emitted to avoid target
        leakage from the fixture currently being modelled.
    """

    history = defaultdict(lambda: deque(maxlen=WINDOW))

    records = []

    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        home_history = list(history[home])
        away_history = list(history[away])

        def summarize(games):
            """Summarize one team's bounded recent-match history.

            Args:
                games: Historical result dictionaries for a single team.

            Returns:
                Counts, points, goals, averages, and win rate for ``games``.

            Notes:
                An empty history returns zero-valued features so the first match
                for a team remains valid without introducing missing values.
            """

            if len(games) == 0:
                return {
                    "played": 0,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "points": 0,
                    "gf": 0,
                    "ga": 0,
                    "gd": 0,
                    "avg_gf": 0,
                    "avg_ga": 0,
                    "win_rate": 0,
                }

            played = len(games)

            wins = sum(g["result"] == "W" for g in games)

            draws = sum(g["result"] == "D" for g in games)

            losses = sum(g["result"] == "L" for g in games)

            gf = sum(g["gf"] for g in games)

            ga = sum(g["ga"] for g in games)

            points = wins * 3 + draws

            return {
                "played": played,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "points": points,
                "gf": gf,
                "ga": ga,
                "gd": gf - ga,
                "avg_gf": gf / played,
                "avg_ga": ga / played,
                "win_rate": wins / played,
            }

        h = summarize(home_history)
        a = summarize(away_history)

        records.append(
            {
                "home_form_played": h["played"],
                "home_form_wins": h["wins"],
                "home_form_draws": h["draws"],
                "home_form_losses": h["losses"],
                "home_form_points": h["points"],
                "home_form_gf": h["gf"],
                "home_form_ga": h["ga"],
                "home_form_gd": h["gd"],
                "home_form_avg_gf": h["avg_gf"],
                "home_form_avg_ga": h["avg_ga"],
                "home_form_win_rate": h["win_rate"],
                "away_form_played": a["played"],
                "away_form_wins": a["wins"],
                "away_form_draws": a["draws"],
                "away_form_losses": a["losses"],
                "away_form_points": a["points"],
                "away_form_gf": a["gf"],
                "away_form_ga": a["ga"],
                "away_form_gd": a["gd"],
                "away_form_avg_gf": a["avg_gf"],
                "away_form_avg_ga": a["avg_ga"],
                "away_form_win_rate": a["win_rate"],
            }
        )

        # Update only after recording the snapshot so current-match outcomes
        # cannot influence their own training features.

        if row["home_score"] > row["away_score"]:
            home_result = "W"
            away_result = "L"
        elif row["home_score"] < row["away_score"]:
            home_result = "L"
            away_result = "W"
        else:
            home_result = "D"
            away_result = "D"

        history[home].append(
            {
                "result": home_result,
                "gf": row["home_score"],
                "ga": row["away_score"],
            }
        )

        history[away].append(
            {
                "result": away_result,
                "gf": row["away_score"],
                "ga": row["home_score"],
            }
        )

    return pd.DataFrame(records)
