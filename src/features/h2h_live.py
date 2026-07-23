"""Calculate runtime head-to-head statistics from raw historical results.

Purpose:
    Supply match-specific historical meeting statistics to live prediction
    feature construction.
Responsibility:
    Load the raw results source once per instance and orient matching fixtures
    from a caller-specified home-team perspective.
Inputs:
    ``data/raw/international_results/international_results.csv`` and two team
    names passed to :meth:`LiveH2H.get_stats`.
Outputs:
    A dictionary of meeting counts, goals, win counts, and win rates.
Interactions:
    ``src.simulator.feature_builder.FeatureBuilder`` calls this class when it
    constructs the exact feature row used by the saved model.
"""

import pandas as pd

from src.config.deployment import find_project_root, log_artifact

ROOT = find_project_root(__file__)

MATCHES = ROOT / "data" / "raw" / "international_results" / "international_results.csv"


class LiveH2H:
    """Provide orientation-aware historical H2H statistics for live inference.

    Args:
        None.

    Notes:
        The source table is loaded during construction so repeated match
        predictions can reuse one in-memory historical dataset.
    """

    def __init__(self):

        self.df = pd.read_csv(
            log_artifact(MATCHES, label="head-to-head history dataset"),
            parse_dates=["date"],
        )

    def get_stats(self, home_team: str, away_team: str) -> dict[str, float | int]:
        """Return historical H2H statistics from the requested home perspective.

        Args:
            home_team: Team to treat as the home side in returned statistics.
            away_team: Opponent to treat as the away side.

        Returns:
            Counts and rates for all historical meetings between the two teams.

        Notes:
            Existing fixtures are reoriented when necessary, so a past away
            match still contributes correctly to the caller's requested view.
        """

        matches = self.df[
            ((self.df.home_team == home_team) & (self.df.away_team == away_team))
            | ((self.df.home_team == away_team) & (self.df.away_team == home_team))
        ]

        total = len(matches)

        if total == 0:
            return {
                "h2h_matches": 0,
                "home_h2h_wins": 0,
                "away_h2h_wins": 0,
                "h2h_draws": 0,
                "home_h2h_goals": 0,
                "away_h2h_goals": 0,
                "home_h2h_win_rate": 0,
                "away_h2h_win_rate": 0,
            }

        home_wins = 0
        away_wins = 0
        draws = 0

        home_goals = 0
        away_goals = 0

        for _, row in matches.iterrows():
            if row.home_team == home_team:
                hg = row.home_score
                ag = row.away_score

            else:
                hg = row.away_score
                ag = row.home_score

            home_goals += hg
            away_goals += ag

            if hg > ag:
                home_wins += 1

            elif hg < ag:
                away_wins += 1

            else:
                draws += 1

        return {
            "h2h_matches": total,
            "home_h2h_wins": home_wins,
            "away_h2h_wins": away_wins,
            "h2h_draws": draws,
            "home_h2h_goals": home_goals,
            "away_h2h_goals": away_goals,
            "home_h2h_win_rate": round(home_wins / total, 3),
            "away_h2h_win_rate": round(away_wins / total, 3),
        }
