"""Build knockout-match feature rows from persisted simulator team ratings.

Purpose:
    Translate compact simulator rating records into the matchup fields required
    by the knockout-oriented feature contract.
Responsibility:
    Load team ratings, resolve requested teams, and emit home/away strength and
    difference values for a neutral World Cup fixture.
Inputs:
    ``data/simulator/team_ratings.parquet`` and two team names.
Outputs:
    A single-row pandas DataFrame of rating-based matchup features.
Interactions:
    This rating-based generator supports the standalone simulator rating path;
    the primary live predictor uses ``FeatureBuilder`` with richer snapshots.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


RATING_FILE = ROOT / "data" / "simulator" / "team_ratings.parquet"


class MatchFeatureGenerator:
    """Generate rating-based feature rows for configured knockout teams.

    Args:
        None.

    Notes:
        Ratings are loaded once when the generator is constructed so paired team
        lookups do not repeatedly read the parquet artifact.
    """

    def __init__(self):

        self.ratings = pd.read_parquet(RATING_FILE)

    def get_team(self, name: str) -> pd.Series:
        """Return the persisted rating record for one configured team.

        Args:
            name: National-team name stored in the ratings artifact.

        Returns:
            The matching team-rating row as a pandas Series.

        Raises:
            ValueError: If ``name`` is not present in the rating artifact.

        Notes:
            Failing early prevents feature construction with fabricated or
            partially populated team ratings.
        """

        team = self.ratings[self.ratings["team"] == name]

        if team.empty:
            raise ValueError(f"{name} not found")

        return team.iloc[0]

    def create_features(self, home_team, away_team):
        """Create a neutral World Cup feature row from two team rating records.

        Args:
            home_team: Team assigned to home-oriented model columns.
            away_team: Team assigned to away-oriented model columns.

        Returns:
            A one-row DataFrame with paired rating fields and explicit
            comparative features.

        Notes:
            The selected field names are intentionally compatible with the
            simulator rating artifact and should not be substituted for the
            richer live ``FeatureBuilder`` schema without retraining.
        """

        home = self.get_team(home_team)

        away = self.get_team(away_team)

        features = {}

        # -----------------------------
        # Team Identity
        # -----------------------------

        features["home_team"] = home_team

        features["away_team"] = away_team

        features["neutral"] = True

        features["tournament"] = "FIFA World Cup"

        # -----------------------------
        # Attack
        # -----------------------------

        features["home_attack_strength"] = home.attack_strength

        features["away_attack_strength"] = away.attack_strength

        features["attack_difference"] = home.attack_strength - away.attack_strength

        # -----------------------------
        # Defense
        # -----------------------------

        features["home_defense_strength"] = home.defense_strength

        features["away_defense_strength"] = away.defense_strength

        features["defense_difference"] = home.defense_strength - away.defense_strength

        # -----------------------------
        # Form
        # -----------------------------

        features["home_form_points"] = home.form

        features["away_form_points"] = away.form

        features["form_difference"] = home.form - away.form

        # -----------------------------
        # Experience
        # -----------------------------

        features["home_wc_appearances"] = home.world_cup_experience

        features["away_wc_appearances"] = away.world_cup_experience

        features["experience_difference"] = (
            home.world_cup_experience - away.world_cup_experience
        )

        # -----------------------------
        # Competition Strength
        # -----------------------------

        features["home_competition_score"] = home.competition_strength

        features["away_competition_score"] = away.competition_strength

        features["competition_score_difference"] = (
            home.competition_strength - away.competition_strength
        )

        return pd.DataFrame([features])
