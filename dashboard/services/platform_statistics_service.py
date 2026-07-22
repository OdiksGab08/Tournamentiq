"""Derive artifact-backed platform metrics for the TournamentIQ Home page.

Purpose:
    Surface real project coverage and production-model facts in the Home
    platform overview without loading estimators or manufacturing values.
Responsibility:
    Read validated historical, ranking, and schema artifacts, cache safe
    primitive metrics, and return ``None`` whenever a source is unavailable.
Inputs:
    The verified match warehouse, saved model-ranking CSV files, and persisted
    model-input dataset schema.
Outputs:
    Home-ready historical, model, feature, competition, and country metrics.
Collaboration:
    ``components.home`` is the presentation consumer; historical records come
    from ``services.statistics_service`` and model metadata remains read-only.
"""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - pyarrow is a project dependency.
    pq = None

try:
    from services.statistics_service import (
        CANONICAL_DATASET_PATH,
        load_historical_matches,
    )
except ImportError:  # Package-oriented test imports can resolve src.services first.
    from dashboard.services.statistics_service import (
        CANONICAL_DATASET_PATH,
        load_historical_matches,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_RESULTS_PATH = PROJECT_ROOT / "models" / "model_results.csv"
MODEL_RANKING_PATH = PROJECT_ROOT / "models" / "model_ranking.csv"
MODEL_INPUT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "ml_dataset.parquet"
_TARGET_COLUMN_NAMES = frozenset({"target", "outcome", "label"})


def _path_token(path: Path) -> tuple[str, int, int]:
    """Return a cache token for one local, read-only artifact."""
    try:
        stat = path.stat()
    except OSError:
        return str(path), -1, -1
    return str(path), int(stat.st_mtime_ns), int(stat.st_size)


def _column(frame: pd.DataFrame, *names: str) -> str | None:
    """Find one case-insensitive column without assuming a CSV schema variant."""
    available = {
        str(column).strip().casefold(): str(column) for column in frame.columns
    }
    return next(
        (available[name.casefold()] for name in names if name.casefold() in available),
        None,
    )


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a model metadata file or return an empty frame on an expected failure."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _clean_text(value: Any) -> str | None:
    """Return a usable non-empty metadata string without manufacturing a value."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _ranked_record(frame: pd.DataFrame) -> dict[str, Any] | None:
    """Select the saved rank-one model record, with file order as a safe fallback."""
    if frame.empty:
        return None
    model_column = _column(frame, "model", "model name")
    if model_column is None:
        return None

    candidates = frame.loc[frame[model_column].map(_clean_text).notna()].copy()
    if candidates.empty:
        return None

    rank_column = _column(candidates, "rank", "model rank")
    if rank_column is not None:
        candidates["_platform_rank"] = pd.to_numeric(
            candidates[rank_column], errors="coerce"
        )
        if candidates["_platform_rank"].notna().any():
            candidates = candidates.sort_values("_platform_rank", na_position="last")

    record = candidates.iloc[0].to_dict()
    record["_platform_model_column"] = model_column
    return record


def _accuracy_from_record(record: dict[str, Any] | None) -> float | None:
    """Return a finite saved model-accuracy value when its column is available."""
    if not record:
        return None

    for column, value in record.items():
        if str(column).strip().casefold() not in {
            "accuracy",
            "test accuracy",
            "test_accuracy",
            "validation accuracy",
        }:
            continue
        try:
            accuracy = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(accuracy) and 0 <= accuracy <= 100:
            return accuracy
    return None


def _model_statistics() -> dict[str, int | float | str | None]:
    """Derive model count, selected production model, and saved accuracy metadata."""
    ranking = _read_csv(MODEL_RANKING_PATH)
    results = _read_csv(MODEL_RESULTS_PATH)
    ranked = _ranked_record(ranking) or _ranked_record(results)

    source = ranking if _column(ranking, "model", "model name") else results
    model_column = _column(source, "model", "model name")
    trained_models: int | None = None
    if model_column is not None:
        names = source[model_column].map(_clean_text).dropna()
        trained_models = (
            int(names.str.casefold().nunique()) if not names.empty else None
        )

    production_model = None
    if ranked:
        production_model = _clean_text(ranked.get(ranked["_platform_model_column"]))

    return {
        "trained_models": trained_models,
        "production_model": production_model,
        "test_accuracy": _accuracy_from_record(ranked),
    }


def _engineered_feature_count() -> int | None:
    """Count persisted model-input columns while excluding only a target column."""
    if pq is None or not MODEL_INPUT_DATASET_PATH.exists():
        return None
    try:
        columns = pq.ParquetFile(MODEL_INPUT_DATASET_PATH).schema_arrow.names
    except (OSError, ValueError):
        return None

    features = [
        name
        for name in columns
        if str(name).strip().casefold() not in _TARGET_COLUMN_NAMES
    ]
    return len(features) or None


def _historical_statistics() -> dict[str, int | None]:
    """Derive coverage counts from the verified, validated historical match table."""
    try:
        matches, quality = load_historical_matches()
    except Exception:
        return {
            "total_historical_matches": None,
            "competitions": None,
            "countries_represented": None,
        }

    competitions = int(matches["competition"].nunique(dropna=True))
    countries = quality.get("unique_teams")
    try:
        countries_represented = int(countries) if countries is not None else None
    except (TypeError, ValueError):
        countries_represented = None
    return {
        "total_historical_matches": int(len(matches)),
        "competitions": competitions,
        "countries_represented": countries_represented,
    }


@st.cache_data(show_spinner=False)
def _load_static_statistics(
    historical_token: tuple[str, int, int],
    results_token: tuple[str, int, int],
    ranking_token: tuple[str, int, int],
    model_input_token: tuple[str, int, int],
) -> dict[str, int | float | str | None]:
    """Load cacheable Home metrics after their source artifacts have been versioned."""
    del historical_token, results_token, ranking_token, model_input_token
    historical = _historical_statistics()
    models = _model_statistics()
    total_matches = historical["total_historical_matches"]
    production_model = models["production_model"]

    # Compatibility aliases preserve current Home consumers while the explicit
    # keys make the source and meaning of each new metric clear.
    return {
        **historical,
        **models,
        "engineered_features": _engineered_feature_count(),
        "international_matches": total_matches,
        "best_model": production_model,
        "latest_simulation_count": None,
    }


def load_platform_statistics() -> dict[str, int | float | str | None]:
    """Return verified Home metrics derived from existing local project artifacts.

    Returns:
        A mapping containing historical coverage, trained-model metadata, and
        persisted model-input feature counts. Values are ``None`` when their
        source artifact is unavailable or invalid; callers must omit those
        metrics rather than present a substitute value.
    """
    return _load_static_statistics(
        _path_token(CANONICAL_DATASET_PATH),
        _path_token(MODEL_RESULTS_PATH),
        _path_token(MODEL_RANKING_PATH),
        _path_token(MODEL_INPUT_DATASET_PATH),
    )


__all__ = ["load_platform_statistics"]
