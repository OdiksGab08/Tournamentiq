"""Re-rank saved model metrics using a normalized composite score.

Purpose:
    Support an alternative transparent selection analysis across accuracy, F1,
    log loss, and training time.
Responsibility:
    Normalize available result columns, calculate a weighted overall score,
    publish a ranking, and refresh the stable production-model alias.
Inputs:
    ``models/model_results.csv`` created by the training workflow.
Outputs:
    Updated ``models/model_ranking.csv`` and ``models/best_model.pkl``.
Interactions:
    This script operates on artifacts from ``trainer.py``; dashboard services
    use the resulting ranking and production alias for metadata and inference.
"""

from pathlib import Path
import shutil

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models"


def normalize(series: pd.Series) -> object:
    """Scale one numeric metric series to the inclusive zero-to-one range.

    Args:
        series: Numeric pandas Series representing one candidate-model metric.

    Returns:
        A NumPy-compatible one-dimensional array of normalized values.

    Notes:
        The comparison workflow inverts lower-is-better metrics after this
        transformation rather than changing the shared normalization routine.
    """

    scaler = MinMaxScaler()

    return scaler.fit_transform(series.values.reshape(-1, 1)).flatten()


def main() -> None:
    """Calculate a composite ranking and publish the selected model alias.

    Args:
        None.

    Returns:
        None. Ranking CSV and production-model alias are written to
        :data:`MODEL_DIR`.

    Notes:
        Optional metric columns fall back to neutral scores, allowing older
        result artifacts to remain comparable without fabricating measurements.
    """

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    results = pd.read_csv(MODEL_DIR / "model_results.csv")

    # Normalize every component to a common direction before weighting metrics
    # with otherwise incompatible scales and units.

    results["Accuracy_N"] = normalize(results["Accuracy"])
    results["F1_N"] = normalize(results["F1"])

    if "Log Loss" in results.columns:
        results["LogLoss_N"] = 1 - normalize(results["Log Loss"])
    else:
        results["LogLoss_N"] = 1.0

    if "Training Time" in results.columns:
        results["Time_N"] = 1 - normalize(results["Training Time"])
    else:
        results["Time_N"] = 1.0

    results["Overall Score"] = (
        0.40 * results["F1_N"]
        + 0.30 * results["Accuracy_N"]
        + 0.20 * results["LogLoss_N"]
        + 0.10 * results["Time_N"]
    )

    results = results.sort_values(
        by="Overall Score",
        ascending=False,
    )

    print(
        results[
            [
                "Model",
                "Accuracy",
                "F1",
                "Log Loss",
                "Training Time",
                "Overall Score",
            ]
        ]
    )

    best = results.iloc[0]

    print("\nBEST MODEL")

    print(best["Model"])

    filename = best["Model"].lower().replace(" ", "_") + ".pkl"

    shutil.copy(
        MODEL_DIR / filename,
        MODEL_DIR / "best_model.pkl",
    )

    results.to_csv(
        MODEL_DIR / "model_ranking.csv",
        index=False,
    )

    print("\nSaved")

    print("best_model.pkl")

    print("model_ranking.csv")


if __name__ == "__main__":
    main()
