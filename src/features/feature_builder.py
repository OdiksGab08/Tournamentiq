"""Combine aligned feature tables into one model-training dataset.

Purpose:
    Join the raw match table with independently calculated feature outputs.
Responsibility:
    Preserve row alignment while concatenating the feature tables column-wise.
Inputs:
    A match DataFrame plus same-length team, form, strength, head-to-head,
    tournament, and competition feature DataFrames.
Outputs:
    A combined DataFrame suitable for subsequent target preparation.
Interactions:
    This helper represents the composition contract also used explicitly by
    ``feature_pipeline``.
"""

import pandas as pd


def build_feature_dataset(
    matches,
    team_stats,
    form,
    strength,
    h2h,
    tournament,
    competition,
):
    """Concatenate aligned raw-match and engineered-feature tables.

    Args:
        matches: Base historical match records.
        team_stats: Rolling team-statistics feature table.
        form: Recent-form feature table.
        strength: Attack and defensive strength feature table.
        h2h: Historical head-to-head feature table.
        tournament: World Cup experience feature table.
        competition: Competition-exposure feature table.

    Returns:
        A column-wise concatenation of all supplied tables.

    Notes:
        Callers are responsible for producing tables in the same chronological
        order; this helper intentionally does not reorder data.
    """

    dataset = matches.copy()

    dataset = pd.concat(
        [
            dataset,
            team_stats,
            form,
            strength,
            h2h,
            tournament,
            competition,
        ],
        axis=1,
    )

    return dataset
