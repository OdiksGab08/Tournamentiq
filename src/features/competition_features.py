"""Build chronological competition-experience features for national teams.

Purpose:
    Quantify a team's prior exposure to competitions of different importance.
Responsibility:
    Maintain per-team cumulative competition scores before each match and return
    a feature row aligned with the supplied match table.
Inputs:
    Chronologically ordered DataFrame containing ``home_team``, ``away_team``,
    and ``tournament`` columns.
Outputs:
    A DataFrame with home/away competition scores, match counts, and score
    differences.
Interactions:
    ``feature_pipeline`` concatenates this output with other leakage-safe
    feature tables to create the training dataset.
"""

from collections import defaultdict

import pandas as pd


COMPETITION_WEIGHTS = {
    "Friendly": 1.0,
    "FIFA World Cup qualification": 4.0,
    "UEFA Nations League": 3.0,
    "UEFA Euro": 4.0,
    "Copa América": 4.0,
    "African Cup of Nations": 4.0,
    "AFC Asian Cup": 4.0,
    "CONCACAF Gold Cup": 4.0,
    "FIFA Confederations Cup": 3.5,
    "FIFA World Cup": 5.0,
}


def initialize() -> dict[str, float | int]:
    """Return the empty history record for one team's competition exposure.

    Args:
        None.

    Returns:
        A mutable dictionary containing score and match counters.

    Notes:
        ``defaultdict`` calls this factory only for teams first seen in the
        chronological source table.
    """

    return {
        "score": 0,
        "matches": 0,
        "competitive_matches": 0,
    }


def build_competition_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Create pre-match competition-strength features for every fixture.

    Args:
        matches: Chronologically ordered historical matches with team and
            tournament columns.

    Returns:
        A feature DataFrame whose row order matches ``matches``.

    Notes:
        Each row is emitted before its competition weight is added to history,
        preventing the fixture being predicted from leaking into its features.
    """

    history = defaultdict(initialize)

    records = []

    for _, row in matches.iterrows():
        home = row["home_team"]

        away = row["away_team"]

        # Snapshot history before the current fixture so model training only
        # receives information that would have been known at kickoff.
        records.append(
            {
                "home_competition_score": history[home]["score"],
                "away_competition_score": history[away]["score"],
                "competition_difference": history[home]["score"]
                - history[away]["score"],
                "home_competition_matches": history[home]["matches"],
                "away_competition_matches": history[away]["matches"],
            }
        )

        weight = COMPETITION_WEIGHTS.get(
            row["tournament"],
            2.0,
        )

        history[home]["score"] += weight
        history[away]["score"] += weight

        history[home]["matches"] += 1
        history[away]["matches"] += 1

        if weight >= 3:
            history[home]["competitive_matches"] += 1
            history[away]["competitive_matches"] += 1

    return pd.DataFrame(records)
