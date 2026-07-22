"""Orchestrate feature generation for the model-training dataset.

Purpose:
    Transform the chronological warehouse match table into the final dataset
    used by target preparation, splitting, and model training.
Responsibility:
    Run every aligned feature builder, merge outputs, derive comparative
    home-versus-away signals, and persist the final parquet artifact.
Inputs:
    ``data/warehouse/master_matches.parquet``.
Outputs:
    ``data/processed/final_training_dataset.parquet``.
Interactions:
    Uses the feature modules in this package; ``src.models.prepare_data`` then
    adds the target and removes leakage-prone columns for model training.
"""

from pathlib import Path

import pandas as pd

from src.features.team_stats import build_team_statistics
from src.features.form_features import build_form_features
from src.features.strength_features import build_strength_features
from src.features.h2h_features import build_head_to_head_features
from src.features.tournament_features import build_tournament_features
from src.features.competition_features import build_competition_features

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "warehouse" / "master_matches.parquet"

OUTPUT_FILE = ROOT / "data" / "processed" / "final_training_dataset.parquet"


def load_matches() -> pd.DataFrame:
    """Load and chronologically order the warehouse match table.

    Args:
        None.

    Returns:
        Historical matches with a parsed, ascending ``date`` column.

    Notes:
        Every feature builder relies on the same deterministic ordering so its
        row output can be concatenated safely with the base matches.
    """

    df = pd.read_parquet(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    return df


def main() -> None:
    """Generate and save the complete engineered training dataset.

    Args:
        None.

    Returns:
        None. The final feature table is written to :data:`OUTPUT_FILE`.

    Notes:
        All feature builders receive the same chronologically ordered table to
        preserve row alignment and prevent future-match information leakage.
    """

    print("=" * 60)
    print("WORLD CUP FEATURE PIPELINE")
    print("=" * 60)

    print("\nLoading Matches...")

    matches = load_matches()

    print(f"{len(matches):,} matches loaded.")

    print("\nBuilding Team Statistics...")

    team_stats = build_team_statistics(matches)

    print("Done.")

    print("\nBuilding Form Features...")

    form = build_form_features(matches)

    print("Done.")

    print("\nBuilding Strength Features...")

    strength = build_strength_features(matches)

    print("Done.")

    print("\nBuilding Head-to-Head Features...")

    h2h = build_head_to_head_features(matches)

    print("Done.")

    print("\nBuilding Tournament Features...")

    tournament = build_tournament_features(matches)

    print("Done.")

    print("\nBuilding Competition Features...")

    competition = build_competition_features(matches)

    print("Done.")

    print("\nMerging Features...")

    final_df = pd.concat(
        [
            matches,
            team_stats,
            form,
            strength,
            h2h,
            tournament,
            competition,
        ],
        axis=1,
    )

    # Feature tables may carry shared identity columns; retain the first copy,
    # which comes from the authoritative warehouse match table.
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    duplicates = final_df.columns[final_df.columns.duplicated()]

    if len(duplicates) > 0:
        print("Duplicate columns found:")

        print(list(duplicates))

    else:
        print("✓ No duplicate columns")

    # ------------------------------------------
    # Explicit differences let the model compare teams without having to infer
    # every relationship from separate home and away feature columns.
    # ------------------------------------------

    print("Creating Difference Features...")

    final_df["goal_difference"] = (
        final_df["home_goal_difference"] - final_df["away_goal_difference"]
    )

    final_df["form_difference"] = (
        final_df["home_form_points"] - final_df["away_form_points"]
    )

    final_df["experience_difference"] = (
        final_df["home_wc_appearances"] - final_df["away_wc_appearances"]
    )

    final_df["competition_score_difference"] = (
        final_df["home_competition_score"] - final_df["away_competition_score"]
    )

    final_df["attack_difference"] = (
        final_df["home_attack_strength"] - final_df["away_attack_strength"]
    )

    final_df["defense_difference"] = (
        final_df["home_defense_strength"] - final_df["away_defense_strength"]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()

    print("=" * 60)

    print("FEATURE ENGINEERING COMPLETE")

    print("=" * 60)

    print()

    print("Dataset Shape")

    print(final_df.shape)

    print()

    print("Saved To")

    print(OUTPUT_FILE)

    print()

    print(final_df.head())


if __name__ == "__main__":
    main()
