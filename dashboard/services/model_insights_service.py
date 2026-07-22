"""Read and normalize saved model-evaluation artifacts for the Insights route.

Purpose:
    Provide transparent, read-only model comparison and optional held-out-test
    diagnostics from the project's persisted training artifacts.
Responsibility:
    Validate artifact schemas, normalize metrics and diagnostics, and reject
    malformed data without fitting models, modifying splits, or writing files.
Inputs:
    Saved CSV, pickle, preprocessor, and immutable held-out test artifacts.
Outputs:
    UI-safe model metadata, comparison tables, diagnostics, and CSV bytes or
    actionable ``ModelInsightsError`` exceptions.
Collaboration:
    Consumed by ``components.model_insights`` and uses the existing predictor
    metadata conventions without changing the training pipeline.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import streamlit as st

try:  # The app runs with ``dashboard`` on sys.path; tests import the package.
    from services.match_prediction_service import (
        BEST_MODEL_PATH,
        PREPROCESSOR_PATH,
        get_predictor,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by package imports.
    from dashboard.services.match_prediction_service import (
        BEST_MODEL_PATH,
        PREPROCESSOR_PATH,
        get_predictor,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_SPLIT_PATH = PROCESSED_DIR / "train.parquet"
VALIDATION_SPLIT_PATH = PROCESSED_DIR / "validation.parquet"
TEST_SPLIT_PATH = PROCESSED_DIR / "test.parquet"

CLASS_LABELS: dict[int, str] = {
    0: "Home Win",
    1: "Draw",
    2: "Away Win",
}
RATE_METRICS: tuple[str, ...] = ("accuracy", "precision", "recall", "f1")
METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "rank": ("rank", "position"),
    "model": ("model", "model_name", "estimator", "estimator_name"),
    "accuracy": ("accuracy", "test_accuracy", "validation_accuracy"),
    "precision": (
        "precision",
        "weighted_precision",
        "precision_weighted",
        "macro_precision",
        "precision_macro",
    ),
    "recall": (
        "recall",
        "weighted_recall",
        "recall_weighted",
        "macro_recall",
        "recall_macro",
    ),
    "f1": ("f1", "f1_score", "weighted_f1", "f1_weighted", "macro_f1", "f1_macro"),
    "log_loss": ("log_loss", "logloss", "cross_entropy"),
    "training_time": ("training_time", "training_seconds", "fit_time", "fit_seconds"),
}

_DISCOVERY_NAMES: dict[str, tuple[str, ...]] = {
    "best_model": ("best_model.pkl", "model.pkl", "best_model.joblib", "model.joblib"),
    "preprocessor": (
        "preprocessor.pkl",
        "pipeline.pkl",
        "preprocessor.joblib",
        "pipeline.joblib",
    ),
    "model_results": ("model_results.csv", "metrics.csv", "model_metrics.csv"),
    "model_ranking": ("model_ranking.csv", "ranking.csv", "model_rankings.csv"),
    "confusion_matrix": ("confusion_matrix.csv", "held_out_confusion_matrix.csv"),
    "classification_report": ("classification_report.csv", "class_report.csv"),
    "test_predictions": (
        "test_predictions.csv",
        "held_out_predictions.csv",
        "evaluation_predictions.csv",
    ),
    "calibration": ("calibration.csv", "calibration_bins.csv", "calibration_curve.csv"),
    "feature_importance": ("feature_importance.csv", "feature_importances.csv"),
    "feature_names": ("feature_names.json", "feature_names.csv"),
    "learning_curves": ("learning_curves.csv", "learning_curve.csv"),
    "shap": ("shap_values.csv", "shap_values.parquet", "shap_summary.csv"),
}


class ModelInsightsError(ValueError):
    """A safe, actionable error for unavailable or malformed ML artifacts."""


def _column_token(value: object) -> str:
    """Normalize a possible metric heading for stable alias matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _path_token(path: Path) -> tuple[str, int, int]:
    """Return a cache invalidation token without exposing it in the UI."""
    try:
        stat = path.stat()
    except OSError:
        return str(path), -1, -1
    return str(path), int(stat.st_mtime_ns), int(stat.st_size)


def _discover_file(key: str, directory: Path = MODELS_DIR) -> Path | None:
    """Find one canonical artifact, preferring repository-used exact names."""
    if not directory.exists() or not directory.is_dir():
        return None
    names = _DISCOVERY_NAMES[key]
    casefolded = {
        child.name.casefold(): child for child in directory.iterdir() if child.is_file()
    }
    for name in names:
        path = casefolded.get(name.casefold())
        if path is not None:
            return path
    return None


def discover_model_artifacts() -> dict[str, Path | None]:
    """Discover known artifact categories without combining experiments."""
    return {key: _discover_file(key) for key in _DISCOVERY_NAMES}


def resolve_metric_columns(columns: Iterable[object]) -> dict[str, str]:
    """Resolve one unambiguous source column for each known metric concept.

    More than one alias for a single metric is rejected rather than silently
    selecting a macro, weighted, or duplicate metric variant.
    """
    source = [str(column) for column in columns]
    resolved: dict[str, str] = {}
    for canonical, aliases in METRIC_ALIASES.items():
        alias_tokens = {_column_token(alias) for alias in aliases}
        matches = [column for column in source if _column_token(column) in alias_tokens]
        if len(matches) > 1:
            raise ModelInsightsError(
                f"The metric artifact has ambiguous columns for {canonical}: "
                f"{', '.join(matches)}."
            )
        if matches:
            resolved[canonical] = matches[0]
    return resolved


def normalize_rate_values(
    values: Sequence[object] | pd.Series, *, label: str
) -> pd.Series:
    """Normalize a consistently decimal or percentage rate series to 0-to-1.

    Values above 100, negatives, non-numeric entries, and non-finite values are
    structural errors.  A series with a maximum above one is treated as a
    percentage scale; saved comparison files are required to use one scale per
    metric column.
    """
    try:
        output = pd.to_numeric(pd.Series(values), errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ModelInsightsError(
            f"{label} contains a non-numeric metric value."
        ) from error
    if output.empty:
        raise ModelInsightsError(f"{label} contains no metric values.")
    if not np.isfinite(output.to_numpy(dtype=float)).all():
        raise ModelInsightsError(f"{label} contains a non-finite metric value.")
    if (output < 0).any() or (output > 100).any():
        raise ModelInsightsError(
            f"{label} must be between 0 and 1 or between 0 and 100."
        )
    return output / 100.0 if float(output.max()) > 1 else output


def _numeric_metric(values: Sequence[object] | pd.Series, *, label: str) -> pd.Series:
    """Validate a finite, non-negative scalar metric without rescaling it."""
    try:
        output = pd.to_numeric(pd.Series(values), errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ModelInsightsError(
            f"{label} contains a non-numeric metric value."
        ) from error
    if (
        output.empty
        or not np.isfinite(output.to_numpy(dtype=float)).all()
        or (output < 0).any()
    ):
        raise ModelInsightsError(f"{label} must contain finite non-negative values.")
    return output


def normalize_model_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize one saved model-comparison experiment.

    The result keeps canonical lowercase field names so rendering never depends
    on a CSV's capitalization or whitespace.  It does not invent missing
    metrics; only present fields are retained.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ModelInsightsError("The model-comparison artifact is empty.")
    columns = resolve_metric_columns(frame.columns)
    model_column = columns.get("model")
    if model_column is None:
        raise ModelInsightsError(
            "The model-comparison artifact has no model-name column."
        )
    output = pd.DataFrame()
    models = frame[model_column].astype("string").str.strip()
    if models.isna().any() or models.eq("").any():
        raise ModelInsightsError(
            "The model-comparison artifact contains a blank model name."
        )
    if models.duplicated().any():
        raise ModelInsightsError(
            "The model-comparison artifact contains duplicate model names."
        )
    output["model"] = models.astype(str)
    if "rank" in columns:
        rank = _numeric_metric(frame[columns["rank"]], label="Rank")
        if (
            not np.allclose(rank.to_numpy(), np.round(rank.to_numpy()))
            or (rank < 1).any()
        ):
            raise ModelInsightsError("Model ranks must be positive whole numbers.")
        if rank.duplicated().any():
            raise ModelInsightsError(
                "The model-ranking artifact contains duplicate ranks."
            )
        output["rank"] = rank.round().astype(int)
    for metric in RATE_METRICS:
        column = columns.get(metric)
        if column is not None:
            output[metric] = normalize_rate_values(
                frame[column], label=metric.replace("_", " ")
            )
    for metric in ("log_loss", "training_time"):
        column = columns.get(metric)
        if column is not None:
            output[metric] = _numeric_metric(
                frame[column], label=metric.replace("_", " ")
            )
    if not any(
        metric in output for metric in (*RATE_METRICS, "log_loss", "training_time")
    ):
        raise ModelInsightsError(
            "The model-comparison artifact has no recognized metric columns."
        )
    return output


def _compare_compatible_experiments(
    results: pd.DataFrame, ranking: pd.DataFrame
) -> None:
    """Reject separate files that cannot safely be treated as one experiment."""
    result_models = set(results["model"].str.casefold())
    ranking_models = set(ranking["model"].str.casefold())
    if result_models != ranking_models:
        raise ModelInsightsError(
            "The model-results and ranking artifacts do not contain the same models."
        )
    shared_metrics = set(results.columns) & set(ranking.columns) - {"model", "rank"}
    for metric in shared_metrics:
        result_values = results.set_index(results["model"].str.casefold())[
            metric
        ].sort_index()
        ranking_values = ranking.set_index(ranking["model"].str.casefold())[
            metric
        ].sort_index()
        if not np.allclose(
            result_values.to_numpy(dtype=float),
            ranking_values.to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ModelInsightsError(
                f"The saved {metric.replace('_', ' ')} values differ between result artifacts."
            )


@st.cache_data(show_spinner=False)
def _load_comparison_cached(
    results_path_text: str,
    results_token: tuple[str, int, int],
    ranking_path_text: str | None,
    ranking_token: tuple[str, int, int] | None,
) -> pd.DataFrame:
    """Read the persisted comparison/ranking experiment once per file version."""
    del results_token, ranking_token
    try:
        results = normalize_model_comparison(pd.read_csv(Path(results_path_text)))
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError) as error:
        raise ModelInsightsError(
            "The saved model-results artifact could not be read."
        ) from error
    if not ranking_path_text:
        return results.sort_values(
            "model", key=lambda values: values.str.casefold(), kind="stable"
        ).reset_index(drop=True)
    try:
        ranking = normalize_model_comparison(pd.read_csv(Path(ranking_path_text)))
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError) as error:
        raise ModelInsightsError(
            "The saved model-ranking artifact could not be read."
        ) from error
    if "rank" not in ranking:
        raise ModelInsightsError("The saved model-ranking artifact has no rank column.")
    _compare_compatible_experiments(results, ranking)
    return ranking.sort_values("rank", kind="stable").reset_index(drop=True)


def load_model_comparison() -> tuple[pd.DataFrame, str, str | None]:
    """Return the canonical saved comparison table and safe source filenames."""
    artifacts = discover_model_artifacts()
    results_path = artifacts.get("model_results")
    ranking_path = artifacts.get("model_ranking")
    if results_path is None:
        raise ModelInsightsError(
            "No saved model-results CSV was found in the models directory."
        )
    comparison = _load_comparison_cached(
        str(results_path),
        _path_token(results_path),
        str(ranking_path) if ranking_path is not None else None,
        _path_token(ranking_path) if ranking_path is not None else None,
    )
    return (
        comparison,
        results_path.name,
        ranking_path.name if ranking_path is not None else None,
    )


def validate_class_mapping(classes: Sequence[object]) -> dict[int, str]:
    """Verify the saved estimator exposes the project's authoritative classes."""
    try:
        normalized = [
            int(value.item() if hasattr(value, "item") else value) for value in classes
        ]
    except (TypeError, ValueError) as error:
        raise ModelInsightsError(
            "The saved model exposes a non-numeric class label."
        ) from error
    if len(normalized) != len(CLASS_LABELS) or len(set(normalized)) != len(normalized):
        raise ModelInsightsError(
            "The saved model must expose exactly three distinct outcome classes."
        )
    if set(normalized) != set(CLASS_LABELS):
        raise ModelInsightsError(
            "The saved model class mapping does not match the verified home-win/draw/away-win targets."
        )
    return {class_id: CLASS_LABELS[class_id] for class_id in normalized}


def _feature_group(feature_name: str) -> str | None:
    """Map only transparent feature prefixes into additive importance groups."""
    source = feature_name.casefold()
    if "__" in source:
        source = source.split("__", 1)[1]
    if source.startswith(("home_team_", "away_team_", "tournament_", "neutral")):
        return "Match context / team identity"
    if "h2h" in source:
        return "Head-to-head"
    if "form" in source:
        return "Recent form"
    if "attack" in source:
        return "Attack strength"
    if "defense" in source:
        return "Defence strength"
    if "goal_difference" in source or "goal_diff" in source:
        return "Goal difference"
    if "competition" in source:
        return "Competition strength"
    if "wc_" in source or "world_cup" in source:
        return "World Cup experience"
    return None


def validate_feature_importance(
    feature_names: Sequence[object],
    importances: Sequence[object],
    *,
    expected_feature_count: int | None = None,
) -> pd.DataFrame:
    """Align verified preprocessor names with native model importances."""
    names = [str(value).strip() for value in feature_names]
    if not names or any(not value for value in names):
        raise ModelInsightsError(
            "The saved preprocessor did not provide valid feature names."
        )
    try:
        values = np.asarray(importances, dtype=float).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ModelInsightsError(
            "The native feature-importance values are invalid."
        ) from error
    if len(names) != len(values):
        raise ModelInsightsError(
            "Feature-name and feature-importance counts do not match."
        )
    if expected_feature_count is not None and len(names) != int(expected_feature_count):
        raise ModelInsightsError(
            "The feature-name count does not match the saved model input count."
        )
    if not np.isfinite(values).all() or (values < 0).any() or float(values.sum()) <= 0:
        raise ModelInsightsError("The native feature-importance values are invalid.")
    output = pd.DataFrame({"feature": names, "importance": values})
    output["feature_group"] = output["feature"].map(_feature_group)
    return output.sort_values("importance", ascending=False, kind="stable").reset_index(
        drop=True
    )


@st.cache_data(show_spinner=False)
def _load_runtime_summary_cached(
    model_token: tuple[str, int, int],
    preprocessor_token: tuple[str, int, int],
) -> dict[str, Any]:
    """Inspect the cached production predictor without storing it in session state."""
    del model_token, preprocessor_token
    try:
        predictor = get_predictor()
        model = predictor.model
        preprocessor = predictor.preprocessor
    except Exception as error:  # The adapter already wraps expected load failures.
        raise ModelInsightsError(
            "The selected production predictor could not be loaded."
        ) from error
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ModelInsightsError(
            "The selected production model does not expose class labels."
        )
    class_mapping = validate_class_mapping(list(classes))
    get_names = getattr(preprocessor, "get_feature_names_out", None)
    if not callable(get_names):
        raise ModelInsightsError(
            "The saved preprocessing pipeline cannot recover feature names."
        )
    try:
        feature_names = list(get_names())
    except (AttributeError, TypeError, ValueError) as error:
        raise ModelInsightsError(
            "The saved preprocessing pipeline could not recover feature names."
        ) from error
    feature_count = getattr(model, "n_features_in_", None)
    if feature_count is None:
        feature_count = len(feature_names)
    try:
        feature_count = int(feature_count)
    except (TypeError, ValueError) as error:
        raise ModelInsightsError(
            "The selected model has an invalid input-feature count."
        ) from error
    importances = getattr(model, "feature_importances_", None)
    importance = (
        validate_feature_importance(
            feature_names, importances, expected_feature_count=feature_count
        )
        if importances is not None
        else pd.DataFrame(columns=["feature", "importance", "feature_group"])
    )
    grouped = (
        importance.dropna(subset=["feature_group"])
        .groupby("feature_group", as_index=False, sort=True)["importance"]
        .sum()
        .sort_values("importance", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    raw_features = getattr(preprocessor, "feature_names_in_", None)
    return {
        "family": type(model).__name__,
        "class_mapping": class_mapping,
        "class_count": len(class_mapping),
        "feature_count": feature_count,
        "raw_feature_count": len(raw_features) if raw_features is not None else None,
        "feature_importance": importance,
        "feature_group_summary": grouped,
        "importance_method": "Native Extra Trees impurity-based feature importance",
        "importance_source": "best_model.pkl with aligned preprocessor feature names",
    }


def _runtime_summary() -> dict[str, Any]:
    """Return cached, production-model introspection for this artifact version."""
    if not BEST_MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise ModelInsightsError(
            "The production model or preprocessing pipeline is missing."
        )
    return _load_runtime_summary_cached(
        _path_token(BEST_MODEL_PATH),
        _path_token(PREPROCESSOR_PATH),
    )


@st.cache_data(show_spinner=False)
def _split_metadata_cached(
    path_text: str, path_token: tuple[str, int, int]
) -> dict[str, Any]:
    """Read only the target column to document an existing immutable split."""
    del path_token
    path = Path(path_text)
    if not path.exists():
        return {"available": False, "rows": None}
    try:
        target = pd.read_parquet(path, columns=["target"])["target"]
    except (OSError, ValueError, KeyError) as error:
        raise ModelInsightsError(
            f"The saved {path.stem} split could not be read."
        ) from error
    return {"available": True, "rows": int(len(target))}


def _split_metadata(path: Path) -> dict[str, Any]:
    """Return safe split row metadata without loading every model feature."""
    return _split_metadata_cached(str(path), _path_token(path))


def _artifact_status(
    artifacts: Mapping[str, Path | None], *, runtime_available: bool
) -> tuple[dict[str, str], dict[str, bool]]:
    """Build explicit availability labels, never inferred placeholder output."""

    def label(key: str) -> str:
        path = artifacts.get(key)
        return f"Available — {path.name}" if path is not None else "Unavailable"

    shap_installed = find_spec("shap") is not None
    availability = {
        "best_model": artifacts.get("best_model") is not None,
        "preprocessor": artifacts.get("preprocessor") is not None,
        "model_results": artifacts.get("model_results") is not None,
        "model_ranking": artifacts.get("model_ranking") is not None,
        "held_out_test_split": TEST_SPLIT_PATH.exists(),
        "held_out_predictions": artifacts.get("test_predictions") is not None,
        "confusion_matrix": artifacts.get("confusion_matrix") is not None,
        "classification_report": artifacts.get("classification_report") is not None,
        "calibration": artifacts.get("calibration") is not None,
        "feature_names": runtime_available,
        "native_feature_importance": runtime_available,
        "shap": bool(shap_installed and artifacts.get("shap") is not None),
        "local_explanations": False,
        "learning_curves": artifacts.get("learning_curves") is not None,
    }
    status = {
        "Best model": label("best_model"),
        "Preprocessing pipeline": label("preprocessor"),
        "Model results": label("model_results"),
        "Model ranking": label("model_ranking"),
        "Held-out test split": "Available — test.parquet"
        if TEST_SPLIT_PATH.exists()
        else "Unavailable",
        "Held-out predictions": label("test_predictions"),
        "Confusion matrix": label("confusion_matrix"),
        "Classification report": label("classification_report"),
        "Calibration artifact": label("calibration"),
        "Feature names": "Available — preprocessing pipeline"
        if runtime_available
        else "Unavailable",
        "Native feature importance": "Available — selected model"
        if runtime_available
        else "Unavailable",
        "SHAP artifact": "Available" if availability["shap"] else "Unavailable",
        "Learning curves": label("learning_curves"),
    }
    return status, availability


def select_ranked_model(comparison: pd.DataFrame) -> pd.Series:
    """Choose the persisted rank-one record, otherwise fail transparently."""
    if "rank" not in comparison:
        raise ModelInsightsError(
            "The selected production model cannot be identified without a saved ranking."
        )
    selected = comparison.loc[comparison["rank"].eq(comparison["rank"].min())]
    if len(selected) != 1:
        raise ModelInsightsError(
            "The saved model ranking does not identify one selected model."
        )
    return selected.iloc[0]


def _model_identity(value: object) -> str:
    """Normalize common persisted model labels and estimator type names."""
    token = re.sub(r"[^a-z0-9]+", "", str(value).casefold())
    token = token.removesuffix("classifier").removesuffix("regressor")
    aliases = {
        "extratrees": "extratrees",
        "randomforest": "randomforest",
        "logisticregression": "logisticregression",
        "catboost": "catboost",
        "lightgbm": "lightgbm",
        "lgbm": "lightgbm",
        "xgboost": "xgboost",
        "xgb": "xgboost",
    }
    return aliases.get(token, token)


def get_model_insights_overview() -> dict[str, Any]:
    """Build a compact, UI-safe overview from saved comparison/model artifacts."""
    comparison, results_name, ranking_name = load_model_comparison()
    selected_row = select_ranked_model(comparison)
    runtime = _runtime_summary()
    artifacts = discover_model_artifacts()
    status, availability = _artifact_status(artifacts, runtime_available=True)
    try:
        model_size = int(BEST_MODEL_PATH.stat().st_size)
    except OSError as error:
        raise ModelInsightsError(
            "The selected production model file is unavailable."
        ) from error
    train_metadata = _split_metadata(TRAIN_SPLIT_PATH)
    validation_metadata = _split_metadata(VALIDATION_SPLIT_PATH)
    test_metadata = _split_metadata(TEST_SPLIT_PATH)
    metrics = {
        metric: float(selected_row[metric])
        for metric in (*RATE_METRICS, "log_loss", "training_time")
        if metric in selected_row and pd.notna(selected_row[metric])
    }
    selected_name = str(selected_row["model"])
    rank = int(selected_row["rank"])
    if _model_identity(selected_name) != _model_identity(runtime["family"]):
        raise ModelInsightsError(
            "The rank-one comparison record does not match the loaded production model artifact."
        )
    validation_rows = validation_metadata.get("rows")
    selection = {
        "explanation": (
            f"{selected_name} is rank {rank} in the saved model-ranking artifact. "
            "The repository trainer sorts the compared validation results by accuracy "
            "before copying the rank-one artifact to best_model.pkl."
        ),
        "rule": "Saved validation accuracy descending; rank-one model copied to best_model.pkl.",
    }
    selected_model = {
        "name": selected_name,
        "family": runtime["family"],
        "artifact_name": BEST_MODEL_PATH.name,
        "model_size_bytes": model_size,
        "rank": rank,
        "feature_count": runtime["feature_count"],
        "raw_feature_count": runtime["raw_feature_count"],
        "class_count": runtime["class_count"],
        "class_mapping": runtime["class_mapping"],
        "class_mapping_verified": True,
        "preprocessor_status": f"Available — {PREPROCESSOR_PATH.name}",
        "training_time": metrics.get("training_time"),
        "training_dataset_size": train_metadata.get("rows"),
        "validation_dataset_size": validation_rows,
        "test_dataset_size": test_metadata.get("rows"),
    }
    limitations = [
        "The fitted model uses historical pre-match engineered match data; those patterns may not reflect a future team's current strength.",
        "The discovered processed features do not expose player availability, injuries, squad-selection, or weather context.",
        "The saved model comparison provides validation metrics; it does not establish certainty for any individual prediction.",
        "Probability output is not proof of calibration unless it is evaluated on held-out probabilities.",
        "Native feature importance describes model reliance or association, not causal impact.",
        "The feature pipeline processes same-date fixtures sequentially, which is an evaluation caveat for historical feature availability.",
    ]
    return {
        "selected_model": selected_model,
        "comparison": comparison,
        "metrics": metrics,
        "metric_context": {
            "split_label": "Saved validation split",
            "averaging": "Precision, recall, and F1 are weighted averages",
            "source": ", ".join(name for name in (results_name, ranking_name) if name),
            "row_count": validation_rows,
        },
        "selection": selection,
        "feature_importance": runtime["feature_importance"],
        "feature_group_summary": runtime["feature_group_summary"],
        "importance_method": runtime["importance_method"],
        "importance_source": runtime["importance_source"],
        "artifact_status": status,
        "availability": availability,
        "limitations": limitations,
        "transparency": {
            "Metric source files": ", ".join(
                name for name in (results_name, ranking_name) if name
            ),
            "Feature-name source": "preprocessor.get_feature_names_out()",
            "Held-out diagnostics source": "Optional in-memory inference on test.parquet; no prediction artifact is saved.",
            "Calibration status": "No saved calibration artifact; available only after optional held-out test inference.",
            "SHAP status": "No saved SHAP artifact; SHAP is not installed.",
            "Unavailable artifacts": "Saved held-out predictions, confusion matrix, classification report, calibration curve, local explanations, and learning curves were not found.",
        },
    }


def validate_confusion_matrix(
    matrix: Sequence[Sequence[object]] | np.ndarray,
    *,
    class_count: int,
    expected_total: int | None = None,
) -> np.ndarray:
    """Validate raw non-negative confusion counts against the class mapping."""
    try:
        output = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as error:
        raise ModelInsightsError(
            "The confusion matrix contains non-numeric values."
        ) from error
    if output.shape != (class_count, class_count):
        raise ModelInsightsError(
            "The confusion-matrix dimensions do not match the verified classes."
        )
    if (
        not np.isfinite(output).all()
        or (output < 0).any()
        or not np.allclose(output, np.round(output))
    ):
        raise ModelInsightsError(
            "The confusion matrix must contain non-negative whole counts."
        )
    if expected_total is not None and int(round(float(output.sum()))) != int(
        expected_total
    ):
        raise ModelInsightsError(
            "The confusion-matrix total does not match the evaluation row count."
        )
    return output.astype(int)


def validate_probability_rows(
    probabilities: Sequence[Sequence[object]] | np.ndarray,
    *,
    class_count: int,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Validate saved-model probability rows without silently normalizing them."""
    try:
        output = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError) as error:
        raise ModelInsightsError(
            "The held-out probabilities contain non-numeric values."
        ) from error
    if output.ndim != 2 or output.shape[1] != class_count or output.shape[0] == 0:
        raise ModelInsightsError(
            "The held-out probability output has an invalid shape."
        )
    if not np.isfinite(output).all() or (output < 0).any() or (output > 1).any():
        raise ModelInsightsError(
            "The held-out probabilities must be finite values between zero and one."
        )
    if not np.allclose(output.sum(axis=1), 1.0, rtol=0.0, atol=tolerance):
        raise ModelInsightsError("The held-out probability rows do not sum to one.")
    return output


def calculate_calibration_bins(
    targets: Sequence[object] | pd.Series,
    probabilities: np.ndarray,
    classes: Sequence[int],
    *,
    bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create transparent one-vs-rest calibration bins and Brier scores."""
    if bins < 2:
        raise ModelInsightsError("Calibration requires at least two bins.")
    try:
        target_series = pd.to_numeric(pd.Series(targets), errors="raise").astype(int)
    except (TypeError, ValueError) as error:
        raise ModelInsightsError(
            "Calibration targets contain invalid class labels."
        ) from error
    validated_probabilities = validate_probability_rows(
        probabilities,
        class_count=len(classes),
    )
    if (
        len(target_series) != len(validated_probabilities)
        or not np.isin(target_series, classes).all()
    ):
        raise ModelInsightsError(
            "Calibration targets do not align with verified probability rows."
        )
    records: list[dict[str, Any]] = []
    brier: list[dict[str, Any]] = []
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for column, class_id in enumerate(classes):
        class_name = CLASS_LABELS[class_id]
        truth = target_series.eq(class_id).astype(float)
        predicted = validated_probabilities[:, column]
        brier.append(
            {
                "Class": class_name,
                "Brier score": float(np.mean((predicted - truth) ** 2)),
            }
        )
        for index in range(bins):
            lower, upper = boundaries[index], boundaries[index + 1]
            mask = (predicted >= lower) & (
                predicted <= upper if index == bins - 1 else predicted < upper
            )
            count = int(mask.sum())
            if count == 0:
                continue
            records.append(
                {
                    "Class": class_name,
                    "Bin start": float(lower),
                    "Bin end": float(upper),
                    "Mean predicted probability": float(predicted[mask].mean()),
                    "Observed frequency": float(truth[mask].mean()),
                    "Count": count,
                }
            )
    return pd.DataFrame(records), pd.DataFrame(brier)


def _group_error_rates(
    frame: pd.DataFrame, column: str, *, minimum_rows: int = 25
) -> pd.DataFrame:
    """Report test-only error rates with sample sizes and a transparent cutoff."""
    if column not in frame:
        return pd.DataFrame()
    source = frame[[column, "incorrect"]].copy()
    source[column] = source[column].astype("string").str.strip().replace("", pd.NA)
    source = source.dropna(subset=[column])
    if source.empty:
        return pd.DataFrame()
    grouped = (
        source.groupby(column, as_index=False, sort=True)["incorrect"]
        .agg(rows="size", incorrect="sum")
        .rename(columns={column: "Group"})
    )
    grouped = grouped.loc[grouped["rows"].ge(minimum_rows)].copy()
    if grouped.empty:
        return grouped
    grouped["error_rate"] = grouped["incorrect"] / grouped["rows"]
    return grouped.sort_values(
        ["error_rate", "rows"], ascending=[False, False], kind="stable"
    ).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _held_out_test_diagnostics_cached(
    test_path_text: str,
    test_token: tuple[str, int, int],
    model_token: tuple[str, int, int],
    preprocessor_token: tuple[str, int, int],
    calibration_bin_count: int,
) -> dict[str, Any]:
    """Run non-mutating inference on the existing immutable test artifact once."""
    del test_token, model_token, preprocessor_token
    if calibration_bin_count < 3 or calibration_bin_count > 30:
        raise ModelInsightsError("Calibration bins must be between 3 and 30.")
    path = Path(test_path_text)
    if not path.exists():
        raise ModelInsightsError("The immutable held-out test split is unavailable.")
    try:
        source = pd.read_parquet(path)
    except (OSError, ValueError) as error:
        raise ModelInsightsError(
            "The immutable held-out test split could not be read."
        ) from error
    if "target" not in source:
        raise ModelInsightsError(
            "The immutable held-out test split has no target column."
        )
    try:
        predictor = get_predictor()
        model = predictor.model
        preprocessor = predictor.preprocessor
    except Exception as error:
        raise ModelInsightsError(
            "The selected predictor could not be loaded for held-out diagnostics."
        ) from error
    classes = list(validate_class_mapping(getattr(model, "classes_", [])))
    raw_features = getattr(preprocessor, "feature_names_in_", None)
    if raw_features is None:
        raise ModelInsightsError(
            "The saved preprocessing pipeline does not expose its input schema."
        )
    required = [str(column) for column in raw_features]
    missing = [column for column in required if column not in source]
    if missing:
        raise ModelInsightsError(
            "The held-out test split is missing required model input fields."
        )
    try:
        targets = pd.to_numeric(source["target"], errors="raise").astype(int).to_numpy()
    except (TypeError, ValueError) as error:
        raise ModelInsightsError(
            "The held-out test target contains invalid class labels."
        ) from error
    if not np.isin(targets, classes).all():
        raise ModelInsightsError(
            "The held-out test target has classes unknown to the selected model."
        )
    try:
        transformed = preprocessor.transform(source.loc[:, required])
        expected_features = int(getattr(model, "n_features_in_", transformed.shape[1]))
        if int(transformed.shape[1]) != expected_features:
            raise ModelInsightsError(
                "The held-out transformed feature count does not match the selected model."
            )
        probabilities = validate_probability_rows(
            model.predict_proba(transformed),
            class_count=len(classes),
        )
        predicted = np.asarray(model.predict(transformed), dtype=int)
    except ModelInsightsError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ModelInsightsError(
            "The selected model could not produce held-out probabilities."
        ) from error
    if predicted.shape != targets.shape or not np.isin(predicted, classes).all():
        raise ModelInsightsError(
            "The selected model returned invalid held-out class predictions."
        )
    try:
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            log_loss,
            precision_recall_fscore_support,
        )
    except (
        ImportError
    ) as error:  # pragma: no cover - scikit-learn is a project dependency.
        raise ModelInsightsError(
            "Scikit-learn is required to calculate held-out diagnostics."
        ) from error
    matrix = validate_confusion_matrix(
        confusion_matrix(targets, predicted, labels=classes),
        class_count=len(classes),
        expected_total=len(targets),
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predicted,
        labels=classes,
        zero_division=0,
    )
    class_report = pd.DataFrame(
        {
            "Class": [CLASS_LABELS[class_id] for class_id in classes],
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Support": support.astype(int),
        }
    )
    confidence = probabilities.max(axis=1)
    ordered = np.sort(probabilities, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    incorrect = predicted != targets
    prediction_frame = source.copy()
    prediction_frame["true_class"] = targets
    prediction_frame["predicted_class"] = predicted
    prediction_frame["incorrect"] = incorrect
    by_class = (
        prediction_frame.groupby("true_class", as_index=False, sort=True)["incorrect"]
        .agg(support="size", errors="sum")
        .rename(columns={"true_class": "Class id"})
    )
    by_class["Class"] = by_class["Class id"].map(CLASS_LABELS)
    by_class["Error rate"] = by_class["errors"] / by_class["support"]
    pairs = prediction_frame.loc[incorrect, ["true_class", "predicted_class"]].copy()
    if pairs.empty:
        common_confusion: dict[str, Any] | None = None
    else:
        pair_counts = (
            pairs.groupby(["true_class", "predicted_class"], as_index=False)
            .size()
            .sort_values("size", ascending=False, kind="stable")
        )
        common = pair_counts.iloc[0]
        common_confusion = {
            "true_label": CLASS_LABELS[int(common["true_class"])],
            "predicted_label": CLASS_LABELS[int(common["predicted_class"])],
            "count": int(common["size"]),
        }
    calibration, brier = calculate_calibration_bins(
        targets, probabilities, classes, bins=calibration_bin_count
    )
    distribution = pd.DataFrame(
        {
            "Class": [CLASS_LABELS[class_id] for class_id in classes],
            "Count": [int((targets == class_id).sum()) for class_id in classes],
        }
    )
    try:
        test_log_loss = float(log_loss(targets, probabilities, labels=classes))
    except (TypeError, ValueError) as error:
        raise ModelInsightsError(
            "Held-out log loss could not be calculated from model probabilities."
        ) from error
    weighted = precision_recall_fscore_support(
        targets, predicted, labels=classes, average="weighted", zero_division=0
    )
    return {
        "availability": {"available": True, "derived_on_demand": True},
        "split_label": "On-demand inference on the immutable held-out test split",
        "row_count": int(len(targets)),
        "class_mapping": {class_id: CLASS_LABELS[class_id] for class_id in classes},
        "metrics": {
            "accuracy": float(accuracy_score(targets, predicted)),
            "precision": float(weighted[0]),
            "recall": float(weighted[1]),
            "f1": float(weighted[2]),
            "log_loss": test_log_loss,
        },
        "confusion_matrix": {
            "matrix": pd.DataFrame(
                matrix,
                index=[CLASS_LABELS[class_id] for class_id in classes],
                columns=[CLASS_LABELS[class_id] for class_id in classes],
            ),
            "labels": [CLASS_LABELS[class_id] for class_id in classes],
            "source": "Derived in memory from best_model.pkl, preprocessor.pkl, and immutable test.parquet",
        },
        "class_report": class_report,
        "error_analysis": {
            "misclassification_count": int(incorrect.sum()),
            "error_rate": float(incorrect.mean()),
            "high_confidence_incorrect_count": int(
                (incorrect & (confidence >= 0.75)).sum()
            ),
            "low_margin_count": int((margin <= 0.10).sum()),
            "confidence_definition": "Largest predicted class probability; high confidence is at least 0.75.",
            "margin_definition": "Largest predicted probability minus second-largest; low margin is at most 0.10.",
            "most_common_confusion": common_confusion,
            "by_true_class": by_class[["Class", "support", "errors", "Error rate"]],
            "by_competition": _group_error_rates(prediction_frame, "tournament"),
            "by_neutral_venue": _group_error_rates(prediction_frame, "neutral"),
        },
        "calibration": {"bins": calibration, "brier_scores": brier},
        "class_distribution": distribution,
        "provenance": "This is on-demand inference only; it does not retrain, alter the split, or write model artifacts.",
    }


def get_held_out_test_diagnostics(*, calibration_bins: int = 10) -> dict[str, Any]:
    """Return cached optional diagnostics from the immutable held-out test split."""
    return _held_out_test_diagnostics_cached(
        str(TEST_SPLIT_PATH),
        _path_token(TEST_SPLIT_PATH),
        _path_token(BEST_MODEL_PATH),
        _path_token(PREPROCESSOR_PATH),
        int(calibration_bins),
    )


def dataframe_csv(frame: pd.DataFrame) -> bytes:
    """Encode only a displayed artifact table for a non-mutating CSV download."""
    if not isinstance(frame, pd.DataFrame):
        raise ModelInsightsError("Only tabular model artifacts can be exported as CSV.")
    try:
        return frame.to_csv(index=False).encode("utf-8")
    except (OSError, TypeError, ValueError) as error:
        raise ModelInsightsError(
            "The requested model artifact could not be exported as CSV."
        ) from error


__all__ = [
    "CLASS_LABELS",
    "METRIC_ALIASES",
    "ModelInsightsError",
    "calculate_calibration_bins",
    "dataframe_csv",
    "discover_model_artifacts",
    "get_held_out_test_diagnostics",
    "get_model_insights_overview",
    "load_model_comparison",
    "normalize_model_comparison",
    "normalize_rate_values",
    "resolve_metric_columns",
    "select_ranked_model",
    "validate_class_mapping",
    "validate_confusion_matrix",
    "validate_feature_importance",
    "validate_probability_rows",
]
