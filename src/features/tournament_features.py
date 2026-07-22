"""Build leakage-safe FIFA World Cup experience features for national teams.

Purpose:
    Measure each team's prior World Cup appearances, matches, wins, and goal
    difference before every historical fixture.
Responsibility:
    Track per-team tournament history and count participation once per World Cup
    edition while preserving chronological feature snapshots.
Inputs:
    Chronologically ordered match DataFrame with dates, tournament labels, team
    names, and final scores.
Outputs:
    A row-aligned DataFrame of home and away World Cup experience features.
Interactions:
    ``feature_pipeline`` merges this output with other engineered features;
    runtime snapshots expose compatible fields to the predictor.
"""

from collections import defaultdict

import pandas as pd


WORLD_CUP = "FIFA World Cup"


def initialize() -> dict[str, int]:
    """Return the empty World Cup history record for one team.

    Args:
        None.

    Returns:
        A mutable dictionary of World Cup appearance, result, and goal counters.

    Notes:
        This factory is passed to ``defaultdict`` so teams are initialized only
        when first encountered in the chronological source data.
    """

    return {
        "appearances": 0,
        "world_cup_matches": 0,
        "world_cup_wins": 0,
        "world_cup_draws": 0,
        "world_cup_losses": 0,
        "world_cup_goals": 0,
        "world_cup_goals_against": 0,
    }


def build_tournament_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Create pre-match World Cup experience features for historical fixtures.

    Args:
        matches: Chronologically ordered matches containing dates, tournament
            labels, team names, and scores.

    Returns:
        A DataFrame of home and away World Cup experience features aligned to
        ``matches``.

    Notes:
        World Cup appearances increment once per team per edition, while match
        and result counters increment per fixture.
    """

    history = defaultdict(initialize)

    records = []

    current_world_cup = None

    appeared_this_world_cup = set()

    for _, row in matches.iterrows():
        home = row["home_team"]

        away = row["away_team"]

        tournament = row["tournament"]

        year = row["date"].year

        # ----------------------------
        # Count an appearance once per edition rather than once per fixture;
        # fixture-level counters below preserve the separate match history.
        # ----------------------------

        if tournament == WORLD_CUP:
            if current_world_cup != year:
                current_world_cup = year

                appeared_this_world_cup = set()

            records.append(
                {
                    "home_wc_appearances": history[home]["appearances"],
                    "away_wc_appearances": history[away]["appearances"],
                    "home_wc_matches": history[home]["world_cup_matches"],
                    "away_wc_matches": history[away]["world_cup_matches"],
                    "home_wc_win_rate": history[home]["world_cup_wins"]
                    / history[home]["world_cup_matches"]
                    if history[home]["world_cup_matches"] > 0
                    else 0,
                    "away_wc_win_rate": history[away]["world_cup_wins"]
                    / history[away]["world_cup_matches"]
                    if history[away]["world_cup_matches"] > 0
                    else 0,
                    "home_wc_goal_diff": history[home]["world_cup_goals"]
                    - history[home]["world_cup_goals_against"],
                    "away_wc_goal_diff": history[away]["world_cup_goals"]
                    - history[away]["world_cup_goals_against"],
                }
            )

            if home not in appeared_this_world_cup:
                history[home]["appearances"] += 1

                appeared_this_world_cup.add(home)

            if away not in appeared_this_world_cup:
                history[away]["appearances"] += 1

                appeared_this_world_cup.add(away)

            history[home]["world_cup_matches"] += 1
            history[away]["world_cup_matches"] += 1

            history[home]["world_cup_goals"] += row["home_score"]
            history[home]["world_cup_goals_against"] += row["away_score"]

            history[away]["world_cup_goals"] += row["away_score"]
            history[away]["world_cup_goals_against"] += row["home_score"]

            if row["home_score"] > row["away_score"]:
                history[home]["world_cup_wins"] += 1
                history[away]["world_cup_losses"] += 1

            elif row["home_score"] < row["away_score"]:
                history[away]["world_cup_wins"] += 1
                history[home]["world_cup_losses"] += 1

            else:
                history[home]["world_cup_draws"] += 1
                history[away]["world_cup_draws"] += 1

        else:
            records.append(
                {
                    "home_wc_appearances": history[home]["appearances"],
                    "away_wc_appearances": history[away]["appearances"],
                    "home_wc_matches": history[home]["world_cup_matches"],
                    "away_wc_matches": history[away]["world_cup_matches"],
                    "home_wc_win_rate": history[home]["world_cup_wins"]
                    / history[home]["world_cup_matches"]
                    if history[home]["world_cup_matches"] > 0
                    else 0,
                    "away_wc_win_rate": history[away]["world_cup_wins"]
                    / history[away]["world_cup_matches"]
                    if history[away]["world_cup_matches"] > 0
                    else 0,
                    "home_wc_goal_diff": history[home]["world_cup_goals"]
                    - history[home]["world_cup_goals_against"],
                    "away_wc_goal_diff": history[away]["world_cup_goals"]
                    - history[away]["world_cup_goals_against"],
                }
            )

    return pd.DataFrame(records)
