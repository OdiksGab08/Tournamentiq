"""Retrieve the latest engineered historical snapshot for one national team.

Purpose:
    Expose the most recent pre-match feature values needed for live model
    inference.
Responsibility:
    Load the final feature dataset once and select the newest home or away row
    for a requested team, preserving its feature orientation.
Inputs:
    ``data/processed/final_training_dataset.parquet`` and a canonical team name.
Outputs:
    A dictionary of the latest team-specific engineered statistics.
Interactions:
    ``FeatureBuilder`` uses the snapshots to build live model inputs for
    ``Predictor`` and dashboard prediction services.
"""

# Load and filter the engineered historical feature table.
import pandas as pd

# Resolve and diagnose the persisted snapshot artifact before reading it.
from src.config.deployment import find_project_root, log_artifact

# Build the absolute input path so prediction does not depend on CWD.
ROOT = find_project_root(__file__)

DATA = ROOT / "data" / "processed" / "final_training_dataset.parquet"


class LiveSnapshot:
    """Provide latest feature snapshots from the processed historical dataset.

    Args:
        None.

    Notes:
        The full feature table is loaded during construction because one live
        prediction requires two snapshots and repeated calls share the dataset.
    """

    def __init__(self):

        # Load the persisted snapshot once for efficient repeated team lookups.
        self.df = pd.read_parquet(log_artifact(DATA, label="team snapshot dataset"))

    def get_snapshot(self, team: str) -> dict[str, object]:
        """Return the latest available engineered feature values for a team.

        Args:
            team: Canonical national-team name present in the processed dataset.

        Returns:
            A team-labelled dictionary of feature values expected by
            :class:`FeatureBuilder`.

        Raises:
            ValueError: If the team has no home or away records in the dataset.

        Notes:
            Home and away rows use different column prefixes, so the method
            selects the newest row first and then reads fields using its prefix.
        """

        # A team can appear in either orientation, so collect and date-sort both views.
        home = self.df[self.df.home_team == team].sort_values("date")

        away = self.df[self.df.away_team == team].sort_values("date")

        if home.empty and away.empty:
            raise ValueError(f"{team} not found.")

        # The latest record may place the same team on either side; select the
        # newest row before removing the side-specific feature prefix.
        if away.empty:
            latest = home.iloc[-1]
            prefix = "home"

        elif home.empty:
            latest = away.iloc[-1]
            prefix = "away"

        else:
            if home.iloc[-1]["date"] >= away.iloc[-1]["date"]:
                latest = home.iloc[-1]
                prefix = "home"

            else:
                latest = away.iloc[-1]
                prefix = "away"

        snapshot = {}

        snapshot["team"] = team

        # These feature suffixes match the data-engineering schema used during training.
        columns = [
            "matches_played",
            "wins",
            "draws",
            "losses",
            "goals_for",
            "goals_against",
            "form_played",
            "form_wins",
            "form_draws",
            "form_losses",
            "form_points",
            "form_gf",
            "form_ga",
            "form_gd",
            "form_avg_gf",
            "form_avg_ga",
            "form_win_rate",
            "attack_strength",
            "defense_strength",
            "goal_difference",
            "clean_sheet_rate",
            "failed_to_score_rate",
            "wc_appearances",
            "wc_matches",
            "wc_win_rate",
            "wc_goal_diff",
            "competition_score",
            "competition_matches",
        ]

        # Copy the latest side-specific values into a team-neutral snapshot dictionary.
        for col in columns:
            snapshot[col] = latest[f"{prefix}_{col}"]

        return snapshot
