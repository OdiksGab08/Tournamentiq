"""Assemble one model-compatible feature row for a live team matchup.

Purpose:
    Recreate the feature schema expected by the persisted preprocessor and
    trained match-outcome model.
Responsibility:
    Combine latest per-team snapshots, live head-to-head data, fixed tournament
    context, and explicit team-difference features into a single DataFrame.
Inputs:
    Home and away team names plus processed and raw historical data loaded by
    dependent snapshot providers.
Outputs:
    One-row pandas DataFrame ready for the saved preprocessor.
Interactions:
    ``Predictor`` and dashboard match services use this builder for inference;
    its field names must remain compatible with the training pipeline.
"""

# Use pandas to return the one-row feature table expected by the preprocessor.
import pandas as pd

# Read latest engineered team values from the persisted snapshot dataset.
from src.simulator.live_snapshot import LiveSnapshot
# Calculate matchup-specific historical head-to-head values.
from src.features.h2h_live import LiveH2H


class FeatureBuilder:
    """Build exact inference features from persisted team and H2H snapshots.

    Args:
        None.

    Notes:
        Providers are created once per builder so repeated predictions reuse
        their in-memory source data rather than rereading artifacts per match.
    """

    def __init__(self):

        # Create reusable data providers so every prediction does not reread source files.
        self.snapshot = LiveSnapshot()
        self.h2h = LiveH2H()

    def build(self, home_team: str, away_team: str) -> pd.DataFrame:
        """Create the model-ready feature row for one neutral World Cup matchup.

        Args:
            home_team: Team positioned as the home side in the model schema.
            away_team: Team positioned as the away side in the model schema.

        Returns:
            A single-row DataFrame whose columns match the persisted model's
            feature-generation contract.

        Notes:
            Difference features are calculated explicitly to match the training
            pipeline rather than relying on a model to infer them from paired
            home and away columns.
        """

        # Retrieve the latest engineered values for both selected teams.
        home = self.snapshot.get_snapshot(home_team)
        away = self.snapshot.get_snapshot(away_team)

        h2h = self.h2h.get_stats(home_team, away_team)

        # Assemble raw fields in the exact schema expected by the saved pipeline.
        row = {}

        # -------------------------------------------------
        # Match Info
        # -------------------------------------------------

        row["home_team"] = home_team
        row["away_team"] = away_team
        row["tournament"] = "FIFA World Cup"
        row["neutral"] = True

        # -------------------------------------------------
        # Copy every Home Feature
        # -------------------------------------------------

        # Prefix values to preserve each team's model feature orientation.
        for key, value in home.items():
            if key == "team":
                continue

            row[f"home_{key}"] = value

        # -------------------------------------------------
        # Copy every Away Feature
        # -------------------------------------------------

        for key, value in away.items():
            if key == "team":
                continue

            row[f"away_{key}"] = value

        # -------------------------------------------------
        # Head-to-Head
        # -------------------------------------------------

        # Add shared matchup history fields after team-specific fields are present.
        row.update(h2h)

        # -------------------------------------------------
        # Keep explicit differences aligned with the engineered training schema;
        # they are not recalculated by the persisted preprocessor.
        # -------------------------------------------------

        row["competition_difference"] = (
            home["competition_score"] - away["competition_score"]
        )

        row["goal_difference"] = home["goal_difference"] - away["goal_difference"]

        row["form_difference"] = home["form_points"] - away["form_points"]

        row["experience_difference"] = home["wc_appearances"] - away["wc_appearances"]

        row["competition_score_difference"] = (
            home["competition_score"] - away["competition_score"]
        )

        row["attack_difference"] = home["attack_strength"] - away["attack_strength"]

        row["defense_difference"] = home["defense_strength"] - away["defense_strength"]

        # Return one record because the classifier predicts one requested fixture at a time.
        return pd.DataFrame([row])
