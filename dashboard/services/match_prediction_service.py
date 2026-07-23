"""Adapt the trained match-outcome model into a validated dashboard contract.

Purpose:
    Supply real three-outcome match probabilities and supporting metadata to
    Streamlit pages without duplicating model, feature, or preprocessing logic.
Responsibility:
    Validate team selections, cache the existing predictor, normalize its class
    probabilities, and expose safe UI-ready result mappings.
Inputs:
    Home or Match Predictor team names plus saved model, preprocessor, and
    project snapshot artifacts.
Outputs:
    Normalized home-win, draw, and away-win probabilities, feature evidence,
    model metadata, team options, or user-facing validation errors.
Collaboration:
    Used by match prediction, comparison, tournament, and Monte Carlo adapters;
    delegates inference to ``src.simulator.predictor.Predictor``.
"""

from __future__ import annotations

# Validate that returned probabilities are finite numeric values.
from math import isfinite
# Build deployment-safe paths for persisted models and datasets.
from pathlib import Path
# Describe UI-safe model and feature payloads.
from typing import Any, Mapping, Sequence

# Read model metadata and normalize feature data.
import pandas as pd
# Cache loaded resources across Streamlit reruns.
import streamlit as st

# Resolve production artifacts and print useful Cloud diagnostics before loading.
from src.config.deployment import find_project_root, log_artifact, log_exception

# Define all persisted artifacts from the discovered repository root.
PROJECT_ROOT = find_project_root(__file__)
MODEL_ROOT = PROJECT_ROOT / "models"
DATA_ROOT = PROJECT_ROOT / "data"
BEST_MODEL_PATH = MODEL_ROOT / "best_model.pkl"
PREPROCESSOR_PATH = MODEL_ROOT / "preprocessor.pkl"
MODEL_RESULTS_PATH = MODEL_ROOT / "model_results.csv"
MODEL_RANKING_PATH = MODEL_ROOT / "model_ranking.csv"
SNAPSHOT_DATA_PATH = DATA_ROOT / "processed" / "final_training_dataset.parquet"
TEAM_RATINGS_PATH = DATA_ROOT / "simulator" / "team_ratings.parquet"

# Verified against src.models.prepare_data.create_target and model.classes_.
# Map the trained classifier's numeric labels to UI-facing outcome names.
CLASS_TO_OUTCOME = {
    0: "home_win",
    1: "draw",
    2: "away_win",
}


class MatchPredictionError(ValueError):
    """A safe, user-facing error raised when real match inference cannot run."""


def _path_signature(path: Path) -> tuple[str, int]:
    # File timestamps make cached values refresh after an artifact changes.
    try:
        return str(path), path.stat().st_mtime_ns
    except OSError:
        return str(path), -1


def _find_column(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    for name in names:
        column = lookup.get(name.lower())
        if column:
            return column
    return None


def _as_python_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def validate_matchup(home_team: str, away_team: str) -> tuple[str, str]:
    """Validate a selected pair before model resources are loaded."""
    home = home_team.strip() if isinstance(home_team, str) else ""
    away = away_team.strip() if isinstance(away_team, str) else ""
    if not home or not away:
        raise MatchPredictionError("Choose both a home team and an away team.")
    if home.casefold() == away.casefold():
        raise MatchPredictionError(
            "Choose two different teams before predicting a match."
        )
    return home, away


def matchup_signature(home_team: str, away_team: str) -> str:
    """Return a stable signature used to prevent stale result presentation."""
    home, away = validate_matchup(home_team, away_team)
    return f"{home.casefold()}::{away.casefold()}::neutral-world-cup"


def is_result_current(result: Mapping[str, Any] | None, signature: str) -> bool:
    """Check whether a saved result belongs to the currently selected matchup."""
    return bool(result and result.get("signature") == signature)


def normalize_class_probabilities(
    classes: Sequence[Any], probabilities: Sequence[Any]
) -> dict[str, float]:
    """Map the model's verified classes to normalized home/draw/away probabilities."""
    if len(classes) != len(probabilities):
        raise MatchPredictionError("The model returned an invalid probability vector.")

    outcome_probabilities: dict[str, float] = {}
    for raw_class, raw_probability in zip(classes, probabilities):
        try:
            model_class = int(_as_python_value(raw_class))
            probability = float(_as_python_value(raw_probability))
        except (TypeError, ValueError) as error:
            raise MatchPredictionError(
                "The model returned non-numeric class probabilities."
            ) from error

        outcome = CLASS_TO_OUTCOME.get(model_class)
        if outcome is None:
            raise MatchPredictionError(
                "The saved model class mapping does not match the expected home/draw/away targets."
            )
        if (
            outcome in outcome_probabilities
            or not isfinite(probability)
            or probability < 0
        ):
            raise MatchPredictionError(
                "The model returned invalid outcome probabilities."
            )
        outcome_probabilities[outcome] = probability

    expected_outcomes = set(CLASS_TO_OUTCOME.values())
    if set(outcome_probabilities) != expected_outcomes:
        raise MatchPredictionError(
            "The saved model did not expose all three required outcome classes."
        )

    total = sum(outcome_probabilities.values())
    if not isfinite(total) or total <= 0:
        raise MatchPredictionError("The model probabilities could not be normalized.")

    return {
        outcome: probability / total
        for outcome, probability in outcome_probabilities.items()
    }


def _normalize_team_names(values: pd.Series) -> list[str]:
    """Remove empty, duplicate selector values while preserving real data names."""
    normalized = (
        values.dropna().astype(str).str.strip().loc[lambda values: values.ne("")]
    )
    return sorted(normalized.drop_duplicates().tolist(), key=str.casefold)


# Cache selector data because team names come from immutable local artifacts.
@st.cache_data(show_spinner=False)
def _load_team_names(
    ratings_path_string: str,
    ratings_modified_at: int,
    snapshots_path_string: str,
    snapshots_modified_at: int,
) -> list[str]:
    """Load simulator teams first, then fall back to processed snapshots."""
    ratings_path = Path(ratings_path_string)
    if ratings_modified_at >= 0 and ratings_path.exists():
        try:
            ratings = pd.read_parquet(ratings_path, columns=["team"])
            rating_teams = _normalize_team_names(ratings["team"])
            if len(rating_teams) >= 2:
                return rating_teams
        except (OSError, ValueError, KeyError):
            pass

    snapshots_path = Path(snapshots_path_string)
    if snapshots_modified_at < 0 or not snapshots_path.exists():
        raise MatchPredictionError("The team snapshot dataset is unavailable.")

    try:
        frame = pd.read_parquet(snapshots_path, columns=["home_team", "away_team"])
    except (OSError, ValueError, KeyError) as error:
        raise MatchPredictionError(
            "The team snapshot dataset could not be read."
        ) from error

    teams = pd.concat([frame["home_team"], frame["away_team"]], ignore_index=True)
    return _normalize_team_names(teams)


def get_available_teams() -> list[str]:
    """Return simulator teams when available, otherwise snapshot-derived teams."""
    return _load_team_names(
        *_path_signature(TEAM_RATINGS_PATH),
        *_path_signature(SNAPSHOT_DATA_PATH),
    )


@st.cache_data(show_spinner=False)
def _load_model_metadata(
    results_path: str,
    results_modified_at: int,
    ranking_path: str,
    ranking_modified_at: int,
) -> dict[str, Any]:
    """Read available model metrics without loading the trained estimator."""
    ranking = pd.DataFrame()
    results = pd.DataFrame()
    try:
        if ranking_modified_at >= 0:
            ranking = pd.read_csv(ranking_path)
        if results_modified_at >= 0:
            results = pd.read_csv(results_path)
    except (OSError, pd.errors.EmptyDataError, UnicodeDecodeError):
        pass

    source = ranking if not ranking.empty else results
    if source.empty:
        return {}

    rank_column = _find_column(source, ("rank",))
    if rank_column:
        source = source.copy()
        source[rank_column] = pd.to_numeric(source[rank_column], errors="coerce")
        source = source.sort_values(rank_column, na_position="last")

    record = {
        key: _as_python_value(value) for key, value in source.iloc[0].to_dict().items()
    }
    model_column = _find_column(source, ("model",))
    model_name = (
        str(record[model_column])
        if model_column and pd.notna(record[model_column])
        else None
    )

    metrics: dict[str, float] = {}
    for canonical, names in {
        "accuracy": ("accuracy",),
        "precision": ("precision",),
        "recall": ("recall",),
        "f1": ("f1",),
        "log_loss": ("log loss", "log_loss"),
        "training_time": ("training time", "training_time"),
    }.items():
        column = _find_column(source, names)
        if column and pd.notna(record.get(column)):
            try:
                metrics[canonical] = float(record[column])
            except (TypeError, ValueError):
                continue

    return {"model_name": model_name, "metrics": metrics}


def get_model_metadata() -> dict[str, Any]:
    """Return best-model metadata from existing result files when available."""
    return _load_model_metadata(
        *_path_signature(MODEL_RESULTS_PATH),
        *_path_signature(MODEL_RANKING_PATH),
    )


# Cache the large model resource once per Streamlit process to avoid reloads.
@st.cache_resource(show_spinner=False)
def get_predictor() -> Any:
    """Load the existing predictor once per Streamlit process, on demand."""
    try:
        # Confirm the Git-LFS model and matching preprocessor exist before deserialization.
        log_artifact(BEST_MODEL_PATH, label="production model")
        log_artifact(PREPROCESSOR_PATH, label="preprocessor")
        from src.simulator.predictor import Predictor

        return Predictor()
    except Exception:
        # Preserve the original exception and full traceback in deployment logs.
        log_exception("trained match predictor load")
        raise


def _feature_snapshot(feature_row: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only interpretable fields from the exact model feature row."""
    evidence_fields = {
        "recent_form": "form_points",
        "attack_strength": "attack_strength",
        "defense_strength": "defense_strength",
        "goal_difference": "goal_difference",
        "competition_strength": "competition_score",
        "world_cup_experience": "wc_appearances",
        "clean_sheet_rate": "clean_sheet_rate",
    }
    home: dict[str, Any] = {}
    away: dict[str, Any] = {}
    for display_name, source_name in evidence_fields.items():
        home_key = f"home_{source_name}"
        away_key = f"away_{source_name}"
        if home_key in feature_row and away_key in feature_row:
            home[display_name] = _as_python_value(feature_row[home_key])
            away[display_name] = _as_python_value(feature_row[away_key])

    head_to_head = {
        key: _as_python_value(feature_row[key])
        for key in (
            "h2h_matches",
            "home_h2h_wins",
            "away_h2h_wins",
            "h2h_draws",
            "home_h2h_win_rate",
            "away_h2h_win_rate",
        )
        if key in feature_row
    }
    return {"home": home, "away": away, "head_to_head": head_to_head}


def predict_match(home_team: str, away_team: str) -> dict[str, Any]:
    """Build real features and return all three trained-model outcome probabilities.

    ``Predictor.predict`` intentionally produces knockout probabilities for the
    tournament engine. This adapter uses the same cached predictor, feature
    builder, preprocessor, and saved classifier, while retaining the raw
    three-class ``predict_proba`` output required by the Match Predictor UI.
    """
    home, away = validate_matchup(home_team, away_team)
    # Reuse the cached trained model, preprocessor, and feature builder.
    predictor = get_predictor()

    try:
        features = predictor.builder.build(home, away)
        transformed = predictor.preprocessor.transform(features)
        # Retain the model's unmodified three-outcome probability distribution.
        raw_probabilities = predictor.model.predict_proba(transformed)[0]
        classes = getattr(predictor.model, "classes_", None)
        if classes is None:
            raise MatchPredictionError("The saved model does not expose class labels.")
        probabilities = normalize_class_probabilities(
            list(classes), list(raw_probabilities)
        )
    except MatchPredictionError:
        raise
    except (AttributeError, KeyError, OSError, ValueError) as error:
        raise MatchPredictionError(
            "The predictor could not build features or return match probabilities for this matchup."
        ) from error

    metadata = get_model_metadata()
    model_name = metadata.get("model_name") or type(predictor.model).__name__
    predicted_outcome = max(probabilities, key=probabilities.get)
    feature_row = features.iloc[0].to_dict()
    return {
        "home_team": home,
        "away_team": away,
        "home_win_probability": probabilities["home_win"],
        "draw_probability": probabilities["draw"],
        "away_win_probability": probabilities["away_win"],
        "predicted_outcome": predicted_outcome,
        "model_name": model_name,
        "model_file": BEST_MODEL_PATH.name if BEST_MODEL_PATH.exists() else None,
        "model_classes": [
            int(_as_python_value(model_class)) for model_class in classes
        ],
        "feature_count": int(transformed.shape[1]),
        "feature_snapshot": _feature_snapshot(feature_row),
        "model_metrics": metadata.get("metrics", {}),
    }
