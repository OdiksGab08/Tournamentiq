"""Evaluate fitted outcome classifiers using consistent validation metrics.

Purpose:
    Calculate the metrics used to compare candidate football-match classifiers.
Responsibility:
    Produce accuracy, weighted precision/recall/F1, and probabilistic log loss
    when an estimator exposes class probabilities.
Inputs:
    A fitted estimator, validation feature matrix, target vector, and model
    display name.
Outputs:
    A metrics dictionary used by the trainer plus console diagnostics.
Interactions:
    ``trainer.train_all_models`` consumes returned metrics to rank and persist
    model candidates.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    log_loss,
)


def evaluate_model(
    model,
    X,
    y,
    model_name,
):
    """Calculate validation metrics for one fitted candidate model.

    Args:
        model: Fitted classifier exposing ``predict`` and optionally
            ``predict_proba``.
        X: Validation feature matrix in the fitted preprocessor's column order.
        y: Validation target labels.
        model_name: Stable display name used in logs and returned metrics.

    Returns:
        A dictionary containing model identity and classification metric values.

    Notes:
        Weighted metrics preserve class representation in the validation set;
        log loss is omitted only for estimators without probability support.
    """

    predictions = model.predict(X)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)

        loss = log_loss(
            y,
            probabilities,
        )

    else:
        loss = None

    accuracy = accuracy_score(
        y,
        predictions,
    )

    precision = precision_score(
        y,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print()
    print("=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    if loss is not None:
        print(f"Log Loss : {loss:.4f}")

    print("\nConfusion Matrix")
    print(
        confusion_matrix(
            y,
            predictions,
        )
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Log Loss": loss,
    }
