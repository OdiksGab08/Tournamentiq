"""Create compact simulator ratings from the engineered historical dataset.

Purpose:
    Derive current attack, defense, form, experience, and competition values
    for the teams supported by the rating-based knockout feature generator.
Responsibility:
    Aggregate each configured team's latest available historical feature values
    and persist the compact rating table.
Inputs:
    ``data/processed/final_training_dataset.parquet`` and
    :data:`WORLD_CUP_TEAMS`.
Outputs:
    ``data/simulator/team_ratings.parquet``.
Interactions:
    ``MatchFeatureGenerator`` loads this artifact for the legacy rating-based
    simulator path; the primary predictor uses richer live snapshots.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


INPUT_FILE = ROOT / "data" / "processed" / "final_training_dataset.parquet"


OUTPUT_FILE = ROOT / "data" / "simulator" / "team_ratings.parquet"


# Teams remaining in World Cup

WORLD_CUP_TEAMS = [
    "France",
    "Morocco",
    "Spain",
    "Belgium",
    "Norway",
    "England",
    "Argentina",
    "Switzerland",
]


def calculate_team_rating(df: pd.DataFrame, team: str) -> dict[str, object]:
    """Calculate compact recent ratings for one configured national team.

    Args:
        df: Final engineered historical dataset containing home and away feature
            columns.
        team: Canonical team name to aggregate.

    Returns:
        A dictionary with attack, defense, form, World Cup experience, and
        competition-strength values.

    Notes:
        Home and away records are combined so a team's rating does not depend on
        which side of historical fixtures it happened to occupy.
    """

    home = df[df["home_team"] == team].copy()

    away = df[df["away_team"] == team].copy()

    ratings = {}

    # -----------------------------
    # Combine both orientations because home/away feature columns describe the
    # same team history from different fixture perspectives.
    # -----------------------------

    home_gf = home["home_goals_for"]

    away_gf = away["away_goals_for"]

    ratings["attack_strength"] = pd.concat([home_gf, away_gf]).tail(10).mean()

    # -----------------------------
    # Defense
    # -----------------------------

    home_ga = home["home_goals_against"]

    away_ga = away["away_goals_against"]

    ratings["defense_strength"] = pd.concat([home_ga, away_ga]).tail(10).mean()

    # -----------------------------
    # Form
    # -----------------------------

    home_points = home["home_form_points"]

    away_points = away["away_form_points"]

    ratings["form"] = pd.concat([home_points, away_points]).tail(10).mean()

    # -----------------------------
    # World Cup Experience
    # -----------------------------

    wc = pd.concat(
        [
            home["home_wc_appearances"],
            away["away_wc_appearances"],
        ]
    )

    ratings["world_cup_experience"] = wc.tail(1).values[0] if len(wc) else 0

    # -----------------------------
    # Competition Strength
    # -----------------------------

    comp = pd.concat(
        [
            home["home_competition_score"],
            away["away_competition_score"],
        ]
    )

    ratings["competition_strength"] = comp.tail(10).mean()

    return ratings


def main() -> None:
    """Build and persist ratings for the configured knockout tournament field.

    Args:
        None.

    Returns:
        None. The compact rating table is written to :data:`OUTPUT_FILE`.

    Notes:
        The configured field is intentionally explicit because the associated
        rating-based knockout flow supports only those teams and fixtures.
    """

    print("=" * 60)

    print("BUILDING TEAM RATINGS")

    print("=" * 60)

    df = pd.read_parquet(INPUT_FILE)

    teams = []

    for team in WORLD_CUP_TEAMS:
        print("Processing:", team)

        rating = calculate_team_rating(df, team)

        rating["team"] = team

        teams.append(rating)

    ratings = pd.DataFrame(teams)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    ratings.to_parquet(OUTPUT_FILE, index=False)

    print()

    print(ratings)

    print()

    print("Saved:")

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
