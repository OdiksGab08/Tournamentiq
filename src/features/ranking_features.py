"""Create leakage-safe historical team-strength features.

This module enriches chronological match records with each team's pre-match
record, win rate, and goal difference. It accepts a match-level dataframe and
returns an enriched copy for the feature pipeline. Statistics are updated only
after the current row is recorded, which keeps the current match outcome from
leaking into its own prediction features.
"""


def create_team_strength_features(df):
    """Add pre-match cumulative team-strength features to match records.

    Args:
        df: Match-level dataframe containing ``date``, home and away team names,
            and home and away scores.

    Returns:
        A date-sorted dataframe copy with cumulative match counts, results, win
        rates, goal differences, and a home-versus-away strength difference.

    Notes:
        Each feature is captured before updating team statistics with the
        current result, preserving a realistic historical prediction context.
    """

    df = df.sort_values("date").copy()

    stats = {}

    home_matches = []
    away_matches = []

    home_wins = []
    away_wins = []

    home_draws = []
    away_draws = []

    home_losses = []
    away_losses = []

    home_win_rate = []
    away_win_rate = []

    home_goal_diff = []
    away_goal_diff = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        if home not in stats:
            stats[home] = {
                "played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "gf": 0,
                "ga": 0,
            }

        if away not in stats:
            stats[away] = {
                "played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "gf": 0,
                "ga": 0,
            }

        h = stats[home]
        a = stats[away]

        home_matches.append(h["played"])
        away_matches.append(a["played"])

        home_wins.append(h["wins"])
        away_wins.append(a["wins"])

        home_draws.append(h["draws"])
        away_draws.append(a["draws"])

        home_losses.append(h["losses"])
        away_losses.append(a["losses"])

        home_goal_diff.append(h["gf"] - h["ga"])

        away_goal_diff.append(a["gf"] - a["ga"])

        if h["played"] == 0:
            home_win_rate.append(0)
        else:
            home_win_rate.append(h["wins"] / h["played"])

        if a["played"] == 0:
            away_win_rate.append(0)
        else:
            away_win_rate.append(a["wins"] / a["played"])

        hs = row["home_score"]
        aw = row["away_score"]

        h["played"] += 1
        a["played"] += 1

        h["gf"] += hs
        h["ga"] += aw

        a["gf"] += aw
        a["ga"] += hs

        if hs > aw:
            h["wins"] += 1
            a["losses"] += 1

        elif hs < aw:
            a["wins"] += 1
            h["losses"] += 1

        else:
            h["draws"] += 1
            a["draws"] += 1

    df["home_matches_played"] = home_matches
    df["away_matches_played"] = away_matches

    df["home_total_wins"] = home_wins
    df["away_total_wins"] = away_wins

    df["home_total_draws"] = home_draws
    df["away_total_draws"] = away_draws

    df["home_total_losses"] = home_losses
    df["away_total_losses"] = away_losses

    df["home_win_rate"] = home_win_rate
    df["away_win_rate"] = away_win_rate

    df["home_goal_difference"] = home_goal_diff
    df["away_goal_difference"] = away_goal_diff

    df["strength_difference"] = df["home_win_rate"] - df["away_win_rate"]

    return df
