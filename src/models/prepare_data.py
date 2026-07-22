"""Prepare engineered match data for supervised outcome-model training.

Purpose:
    Create the three-class outcome target and remove fields that would reveal
    the result or introduce non-predictive identifiers.
Responsibility:
    Convert the final feature dataset into a model-ready parquet artifact while
    preserving the existing feature-engineering contract.
Inputs:
    ``data/processed/final_training_dataset.parquet``.
Outputs:
    ``data/processed/ml_dataset.parquet`` with a ``target`` column.
Interactions:
    ``train_validation_split`` partitions this output; ``preprocess`` then
    transforms its features before candidate-model training.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data" / "processed" / "final_training_dataset.parquet"

OUTPUT = ROOT / "data" / "processed" / "ml_dataset.parquet"


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the canonical home-win, draw, away-win target to match records.

    Args:
        df: Engineered match DataFrame containing home and away final scores.

    Returns:
        The same DataFrame with integer ``target`` values: 0 for home wins, 1
        for draws, and 2 for away wins.

    Notes:
        The class mapping is consumed by the saved model and dashboard adapter,
        so it must remain stable across retraining runs.
    """

    target = []

    for home, away in zip(df["home_score"], df["away_score"]):
        if home > away:
            target.append(0)

        elif home == away:
            target.append(1)

        else:
            target.append(2)

    df["target"] = target

    return df


def remove_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Remove result and identity fields that should not enter model features.

    Args:
        df: Match DataFrame that already includes the training target.

    Returns:
        A DataFrame without present columns listed in ``leakage_columns``.

    Notes:
        Removing final scores prevents the classifier from observing its answer;
        date and location identifiers are also excluded from this established
        training contract.
    """

    leakage_columns = [
        "match_id",
        "home_score",
        "away_score",
        "city",
        "country",
        "date",
    ]

    existing = [c for c in leakage_columns if c in df.columns]

    df = df.drop(columns=existing)

    return df


def main() -> None:
    """Create and persist the model-ready dataset from engineered features.

    Args:
        None.

    Returns:
        None. The prepared dataset is written to :data:`OUTPUT`.

    Notes:
        Target creation occurs before leakage removal because final scores are
        needed to derive the label but must never remain in model inputs.
    """

    print("=" * 60)
    print("PREPARING MACHINE LEARNING DATASET")
    print("=" * 60)

    df = pd.read_parquet(INPUT)

    print(f"Loaded {len(df):,} matches")

    # Derive the label before removing final scores, then remove every feature
    # that could disclose the already-known match outcome to the classifier.
    df = create_target(df)

    df = remove_leakage(df)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT,
        index=False,
    )

    print()

    print("Dataset Saved")

    print(OUTPUT)

    print()

    print(df.head())

    print()

    print(df.shape)


if __name__ == "__main__":
    main()
