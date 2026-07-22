"""Train, evaluate, rank, and package candidate match-outcome models.

Purpose:
    Produce the persisted estimators and model metadata used by TournamentIQ
    inference and model-insight views.
Responsibility:
    Preprocess chronological data, fit every candidate model, evaluate on the
    validation split, save artifacts, rank candidates, and copy the winner to
    the stable production filename.
Inputs:
    Persisted train/validation/test parquet splits from ``src.models``.
Outputs:
    Individual ``.pkl`` models, ``model_results.csv``, ``model_ranking.csv``,
    and ``best_model.pkl`` in ``models/``.
Interactions:
    The dashboard's cached predictor loads ``best_model.pkl`` and the shared
    preprocessor; analytics services read the ranking/result CSV artifacts.
"""

from pathlib import Path
import shutil
import time

import joblib
import pandas as pd

from src.models.preprocess import (
    load_datasets,
    preprocess,
)

from src.models.models import get_models
from src.models.evaluation import evaluate_model

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models"


def train_all_models() -> pd.DataFrame:
    """Fit and rank every configured classifier, then publish the top artifact.

    Args:
        None.

    Returns:
        The ranked candidate-model DataFrame written to ``model_ranking.csv``.

    Notes:
        Candidate failures are reported and skipped so one unavailable optional
        library does not prevent other models from producing usable artifacts.
    """

    print("=" * 60)
    print("WORLD CUP MODEL TRAINER")
    print("=" * 60)

    print("\nLoading datasets...")

    train, validation, test = load_datasets()

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = preprocess(
        train,
        validation,
        test,
    )

    print("✓ Preprocessing Complete")

    print(f"Training Samples : {len(X_train)}")
    print(f"Validation Samples : {len(X_validation)}")
    print(f"Features : {X_train.shape[1]}")

    models = get_models()

    print(f"\nTraining {len(models)} models...\n")

    results = []

    for name, model in models.items():
        print("=" * 60)
        print(f"Training {name}")
        print("=" * 60)

        start = time.time()

        try:
            model.fit(
                X_train,
                y_train,
            )

            training_time = round(
                time.time() - start,
                2,
            )

            metrics = evaluate_model(
                model=model,
                X=X_validation,
                y=y_validation,
                model_name=name,
            )

            metrics["Training Time"] = training_time

            filename = name.lower().replace(" ", "_") + ".pkl"

            model_path = MODEL_DIR / filename

            joblib.dump(
                model,
                model_path,
            )

            print(f"✓ Saved {filename}")

            results.append(metrics)

        except Exception as e:
            print(f"✗ {name} Failed")

            print(e)

            continue

    results = pd.DataFrame(results)

    # Rank by the established validation accuracy contract before exposing the
    # production alias consumed by runtime inference.
    results = results.sort_values(
        by="Accuracy",
        ascending=False,
    )

    results.to_csv(
        MODEL_DIR / "model_results.csv",
        index=False,
    )

    ranking = results.copy()

    ranking.insert(
        0,
        "Rank",
        range(
            1,
            len(ranking) + 1,
        ),
    )

    ranking.to_csv(
        MODEL_DIR / "model_ranking.csv",
        index=False,
    )

    best_model_name = ranking.iloc[0]["Model"]

    print("\nBest Model:")

    print(best_model_name)

    best_filename = best_model_name.lower().replace(" ", "_") + ".pkl"

    # The stable alias lets the predictor load the selected model without
    # coupling production inference to a particular estimator filename.
    shutil.copy(
        MODEL_DIR / best_filename,
        MODEL_DIR / "best_model.pkl",
    )

    print("\nBest model saved.")

    print("=" * 60)

    return ranking


if __name__ == "__main__":
    ranking = train_all_models()

    print()

    print(ranking)
