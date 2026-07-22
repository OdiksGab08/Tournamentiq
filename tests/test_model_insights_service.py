"""Focused contracts for the Model Insights dashboard service.

These tests intentionally use only small synthetic inputs.  They must never
load the production model or any persisted evaluation split.
"""

import numpy as np
import pandas as pd
import pytest

from dashboard.services.model_insights_service import (
    ModelInsightsError,
    calculate_calibration_bins,
    dataframe_csv,
    normalize_model_comparison,
    normalize_rate_values,
    resolve_metric_columns,
    select_ranked_model,
    validate_class_mapping,
    validate_confusion_matrix,
    validate_feature_importance,
    validate_probability_rows,
)


def test_rate_normalization_accepts_consistent_decimal_and_percentage_series():
    """Verify equivalent decimal and percentage rates normalize identically."""
    decimals = normalize_rate_values([0.5652525252525252, 0.5], label="Accuracy")
    percentages = normalize_rate_values([56.52525252525252, 50.0], label="Accuracy")

    assert decimals.tolist() == pytest.approx([0.5652525252525252, 0.5])
    assert percentages.tolist() == pytest.approx(decimals.tolist())


@pytest.mark.parametrize("value", [-0.1, float("nan"), "not-a-rate"])
def test_rate_normalization_rejects_invalid_values(value):
    """Verify invalid rate values cannot enter model-insight metrics."""
    with pytest.raises(ModelInsightsError):
        normalize_rate_values([value], label="F1")


def test_class_mapping_accepts_the_verified_three_outcome_classes_only():
    """Verify only the canonical home, draw, and away outcome classes are valid."""
    assert validate_class_mapping([0, 1, 2]) == {
        0: "Home Win",
        1: "Draw",
        2: "Away Win",
    }
    assert validate_class_mapping(np.array([2, 0, 1])) == {
        2: "Away Win",
        0: "Home Win",
        1: "Draw",
    }

    with pytest.raises(ModelInsightsError):
        validate_class_mapping([0, 1, 3])
    with pytest.raises(ModelInsightsError):
        validate_class_mapping([0, 1, 1])


def test_metric_aliases_and_comparison_normalization_preserve_rank_one_selection():
    """Verify metric aliases normalize comparison data without changing ranking."""
    source = pd.DataFrame(
        {
            "Rank": [2, 1],
            "Model Name": ["Random Forest", "Extra Trees"],
            "Validation Accuracy": [0.5640, 0.5653],
            "Weighted F1": [0.4924, 0.4942],
            "Log Loss": [0.9329, 0.9282],
            "Training Time": [45.42, 116.13],
        }
    )

    assert resolve_metric_columns(source.columns) == {
        "rank": "Rank",
        "model": "Model Name",
        "accuracy": "Validation Accuracy",
        "f1": "Weighted F1",
        "log_loss": "Log Loss",
        "training_time": "Training Time",
    }

    comparison = normalize_model_comparison(source)
    assert comparison.columns.tolist() == [
        "model",
        "rank",
        "accuracy",
        "f1",
        "log_loss",
        "training_time",
    ]
    assert comparison.sort_values("rank").iloc[0]["model"] == "Extra Trees"
    assert comparison.loc[
        comparison["model"].eq("Extra Trees"), "accuracy"
    ].item() == pytest.approx(0.5653)

    with pytest.raises(ModelInsightsError, match="ambiguous"):
        resolve_metric_columns(["Model", "Accuracy", "Validation Accuracy"])


def test_select_ranked_model_uses_one_saved_rank_one_record():
    """Verify production-model selection requires exactly one saved rank-one row."""
    comparison = pd.DataFrame(
        {"model": ["Random Forest", "Extra Trees"], "rank": [2, 1]}
    )

    selected = select_ranked_model(comparison)
    assert selected["model"] == "Extra Trees"
    assert selected["rank"] == 1

    with pytest.raises(ModelInsightsError, match="without a saved ranking"):
        select_ranked_model(comparison.drop(columns="rank"))
    with pytest.raises(ModelInsightsError, match="does not identify one"):
        select_ranked_model(pd.DataFrame({"model": ["A", "B"], "rank": [1, 1]}))


def test_confusion_matrix_requires_a_three_by_three_numeric_matrix():
    """Verify confusion matrices match the expected outcome shape and total."""
    matrix = np.array([[5, 1, 0], [1, 4, 1], [0, 2, 6]])
    validate_confusion_matrix(matrix, class_count=3, expected_total=20)

    with pytest.raises(ModelInsightsError):
        validate_confusion_matrix(np.array([[1, 2], [3, 4]]), class_count=3)
    with pytest.raises(ModelInsightsError):
        validate_confusion_matrix(matrix, class_count=3, expected_total=21)


def test_probability_rows_require_three_non_negative_rows_that_sum_to_one():
    """Verify probability rows have one non-negative value per outcome class."""
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    validated = validate_probability_rows(probabilities, class_count=3)
    assert validated.shape == (2, 3)
    assert validated.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])

    with pytest.raises(ModelInsightsError):
        validate_probability_rows([[0.7, 0.2], [0.1, 0.9]], class_count=3)
    with pytest.raises(ModelInsightsError):
        validate_probability_rows([[0.7, 0.2, 0.2]], class_count=3)


def test_calibration_bins_use_validated_rows_and_real_one_vs_rest_brier_scores():
    """Verify calibration output uses validated probabilities and true Brier scores."""
    targets = [0, 1, 2, 0]
    probabilities = np.array(
        [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.6, 0.2, 0.2]]
    )

    calibration, brier = calculate_calibration_bins(
        targets,
        probabilities,
        [0, 1, 2],
        bins=2,
    )

    assert calibration.groupby("Class")["Count"].sum().to_dict() == {
        "Home Win": 4,
        "Draw": 4,
        "Away Win": 4,
    }
    assert brier.set_index("Class")["Brier score"].to_dict() == pytest.approx(
        {"Home Win": 0.0625, "Draw": 0.045, "Away Win": 0.0375}
    )

    with pytest.raises(ModelInsightsError, match="at least two bins"):
        calculate_calibration_bins(targets, probabilities, [0, 1, 2], bins=1)


def test_native_feature_importance_requires_aligned_real_feature_names():
    """Verify feature importances retain aligned, non-fabricated feature names."""
    importance = validate_feature_importance(
        ["numeric__home_attack_strength", "numeric__away_defense_strength"],
        [0.35, 0.65],
        expected_feature_count=2,
    )

    assert importance["feature"].tolist() == [
        "numeric__away_defense_strength",
        "numeric__home_attack_strength",
    ]
    assert importance["importance"].tolist() == pytest.approx([0.65, 0.35])
    assert importance["feature_group"].tolist() == [
        "Defence strength",
        "Attack strength",
    ]

    with pytest.raises(ModelInsightsError, match="counts do not match"):
        validate_feature_importance(["numeric__home_attack_strength"], [0.3, 0.7])
    with pytest.raises(ModelInsightsError, match="input count"):
        validate_feature_importance(
            ["feature_a", "feature_b"], [0.3, 0.7], expected_feature_count=3
        )


def test_csv_export_is_limited_to_the_current_tabular_artifact():
    """Verify CSV export serializes only a valid current tabular artifact."""
    exported = dataframe_csv(
        pd.DataFrame({"model": ["Extra Trees"], "accuracy": [0.5653]})
    )

    assert exported.decode("utf-8").splitlines() == [
        "model,accuracy",
        "Extra Trees,0.5653",
    ]
    with pytest.raises(ModelInsightsError):
        dataframe_csv({"model": "Extra Trees"})  # type: ignore[arg-type]
