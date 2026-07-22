"""Build validated, real-data team comparisons for the dashboard.

Purpose:
    Supply team-comparison pages with current engineered indicators, historical
    context, and an optional trained-model outlook without conflating them.
Responsibility:
    Validate team pairs, normalize comparable features transparently, orient
    match history, and keep comparison scores separate from ML probabilities.
Inputs:
    Team selections, processed snapshot and historical-match artifacts, and
    optional results from ``services.match_prediction_service``.
Outputs:
    UI-safe snapshots, metrics, feature verdicts, histories, signatures, and
    actionable ``TeamComparisonError`` failures.
Collaboration:
    Used by ``components.team_comparison`` and reuses the match-prediction
    service only for its explicitly labelled model outlook.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Final, Literal

import pandas as pd
import streamlit as st

from .match_prediction_service import (
    MatchPredictionError,
    get_available_teams,
    predict_match,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
SNAPSHOT_DATA_PATH = DATA_ROOT / "processed" / "final_training_dataset.parquet"
HISTORY_DATA_PATH = (
    DATA_ROOT / "raw" / "international_results" / "international_results.csv"
)

_INVALID_TEAM_NAMES: Final[frozenset[str]] = frozenset(
    {"", "-", "n/a", "na", "nan", "none", "null", "tbd", "unknown"}
)


class TeamComparisonError(ValueError):
    """A safe, user-facing error raised when a comparison cannot be built."""


@dataclass(frozen=True, slots=True)
class FeatureMetadata:
    """Presentation and direction rules for one real engineered feature."""

    label: str
    higher_is_better: bool
    value_format: Literal["decimal", "integer", "percentage", "signed_decimal"]
    description: str
    show_in_key_cards: bool = True
    show_in_radar: bool = True


# This is the single source of truth for labels, formatting, favourable
# direction, metric cards, strength bars, radar normalization, and verdicts.
FEATURE_CONFIG: Final[dict[str, FeatureMetadata]] = {
    "form_win_rate": FeatureMetadata(
        label="Recent Form Win Rate",
        higher_is_better=True,
        value_format="percentage",
        description="Win rate in the latest five-match form window.",
    ),
    "form_points": FeatureMetadata(
        label="Recent Form Points",
        higher_is_better=True,
        value_format="integer",
        description="Points accumulated in the latest five-match form window.",
    ),
    "attack_strength": FeatureMetadata(
        label="Attack Strength",
        higher_is_better=True,
        value_format="decimal",
        description="Engineered attacking indicator from the latest snapshot.",
    ),
    "defense_strength": FeatureMetadata(
        label="Defence Strength",
        higher_is_better=False,
        value_format="decimal",
        description="Engineered defensive indicator; lower values are favourable.",
    ),
    "goal_difference": FeatureMetadata(
        label="Goal Difference",
        higher_is_better=True,
        value_format="signed_decimal",
        description="Recent goals-for minus goals-against indicator.",
    ),
    "clean_sheet_rate": FeatureMetadata(
        label="Clean-sheet Rate",
        higher_is_better=True,
        value_format="percentage",
        description="Share of recent form matches without conceding.",
    ),
    "form_avg_gf": FeatureMetadata(
        label="Recent Goals / Match",
        higher_is_better=True,
        value_format="decimal",
        description="Average goals scored in the current five-match form window.",
        show_in_key_cards=False,
    ),
    "form_avg_ga": FeatureMetadata(
        label="Recent Goals Conceded / Match",
        higher_is_better=False,
        value_format="decimal",
        description="Average goals conceded in the current five-match form window; lower is favourable.",
        show_in_key_cards=False,
    ),
    "competition_score": FeatureMetadata(
        label="Competition Score",
        higher_is_better=True,
        value_format="integer",
        description="Existing competition-strength aggregate in the engineered snapshot.",
    ),
    "wc_appearances": FeatureMetadata(
        label="World Cup Appearances",
        higher_is_better=True,
        value_format="integer",
        description="Recorded FIFA World Cup appearance count in the snapshot.",
        show_in_radar=False,
    ),
    "wc_win_rate": FeatureMetadata(
        label="World Cup Win Rate",
        higher_is_better=True,
        value_format="percentage",
        description="Recorded FIFA World Cup win rate in the engineered snapshot.",
    ),
}


def _path_signature(path: Path) -> tuple[str, int]:
    """Return a cache-safe file identity without exposing paths to the UI."""
    try:
        return str(path), path.stat().st_mtime_ns
    except OSError:
        return str(path), -1


def _to_python(value: Any) -> Any:
    """Convert pandas and NumPy scalar values into session-state-safe values."""
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _clean_team_name(value: Any) -> str | None:
    """Return a canonical display name or ``None`` for invalid selector values."""
    if not isinstance(value, str):
        return None
    name = value.strip()
    if not name or name.casefold() in _INVALID_TEAM_NAMES:
        return None
    return name


def normalize_team_names(values: Iterable[Any]) -> list[str]:
    """Remove null, blank, invalid, and duplicate team values deterministically."""
    names: dict[str, str] = {}
    for value in values:
        name = _clean_team_name(value)
        if name is not None:
            names.setdefault(name.casefold(), name)
    return sorted(names.values(), key=str.casefold)


def validate_comparison_teams(team_a: str, team_b: str) -> tuple[str, str]:
    """Validate two distinct national-team selections before data is loaded."""
    normalized_a = _clean_team_name(team_a)
    normalized_b = _clean_team_name(team_b)
    if normalized_a is None or normalized_b is None:
        raise TeamComparisonError("Choose two valid national teams to compare.")
    if normalized_a.casefold() == normalized_b.casefold():
        raise TeamComparisonError("Choose two different teams before comparing them.")
    return normalized_a, normalized_b


def get_snapshot_version() -> str:
    """Return the processed-snapshot version used to invalidate stale results."""
    _, modified_at = _path_signature(SNAPSHOT_DATA_PATH)
    return str(modified_at)


def comparison_signature(team_a: str, team_b: str, snapshot_version: str) -> str:
    """Create a stable team-and-snapshot signature for session-state results."""
    normalized_a, normalized_b = validate_comparison_teams(team_a, team_b)
    return f"{normalized_a.casefold()}::{normalized_b.casefold()}::{snapshot_version}"


def is_comparison_current(result: Mapping[str, Any] | None, signature: str) -> bool:
    """Check whether a saved comparison belongs to the active team selections."""
    return bool(result and result.get("signature") == signature)


def _finite_number(value: Any) -> float | None:
    """Return a finite float, preserving missing and malformed values as absent."""
    try:
        number = float(_to_python(value))
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def normalize_pair(
    value_a: float, value_b: float, *, higher_is_better: bool
) -> tuple[float, float]:
    """Normalize one real metric pair to [0, 1], respecting feature direction.

    Equal values receive 0.5 each.  Otherwise each pair is min-max normalized
    before an inversion for lower-is-better metrics, allowing unlike raw scales
    to contribute equally to the transparent feature-based comparison.
    """
    numeric_a = _finite_number(value_a)
    numeric_b = _finite_number(value_b)
    if numeric_a is None or numeric_b is None:
        raise TeamComparisonError(
            "A comparison metric contains an invalid numeric value."
        )
    if numeric_a == numeric_b:
        return 0.5, 0.5

    low = min(numeric_a, numeric_b)
    high = max(numeric_a, numeric_b)
    score_a = (numeric_a - low) / (high - low)
    score_b = (numeric_b - low) / (high - low)
    if not higher_is_better:
        score_a, score_b = 1.0 - score_a, 1.0 - score_b
    return score_a, score_b


def format_feature_value(value: Any, metadata: FeatureMetadata) -> str:
    """Format a raw engineered value without replacing it with a normalized score."""
    numeric_value = _finite_number(value)
    if numeric_value is None:
        return "Unavailable"
    if metadata.value_format == "percentage":
        return f"{numeric_value:.0%}"
    if metadata.value_format == "integer":
        return f"{numeric_value:,.0f}"
    if metadata.value_format == "signed_decimal":
        return f"{numeric_value:+.2f}"
    return f"{numeric_value:.2f}"


def _favorable_side(
    score_a: float, score_b: float
) -> Literal["team_a", "team_b", "level"]:
    if abs(score_a - score_b) < 1e-9:
        return "level"
    return "team_a" if score_a > score_b else "team_b"


@st.cache_resource(show_spinner=False)
def _get_live_snapshot_provider(
    snapshot_path_string: str, snapshot_modified_at: int
) -> Any:
    """Load the existing backend snapshot provider once per source version."""
    snapshot_path = Path(snapshot_path_string)
    if snapshot_modified_at < 0 or not snapshot_path.exists():
        raise TeamComparisonError("The processed team snapshot dataset is unavailable.")
    try:
        from src.simulator.live_snapshot import LiveSnapshot

        return LiveSnapshot()
    except (ImportError, OSError, ValueError) as error:
        raise TeamComparisonError(
            "The existing team snapshot provider could not be loaded."
        ) from error


@st.cache_data(show_spinner=False)
def _load_team_snapshot(
    team: str, snapshot_path_string: str, snapshot_modified_at: int
) -> dict[str, Any]:
    """Retrieve one exact backend-engineered snapshot in a UI-safe form."""
    provider = _get_live_snapshot_provider(snapshot_path_string, snapshot_modified_at)
    try:
        snapshot = provider.get_snapshot(team)
    except (KeyError, TypeError, ValueError) as error:
        raise TeamComparisonError(
            f"No valid engineered snapshot is available for {team}."
        ) from error
    return {str(key): _to_python(value) for key, value in snapshot.items()}


def get_team_snapshot(team: str) -> dict[str, Any]:
    """Return the latest real backend snapshot for a selected team."""
    normalized_team = _clean_team_name(team)
    if normalized_team is None:
        raise TeamComparisonError("A valid team name is required to load a snapshot.")
    return _load_team_snapshot(normalized_team, *_path_signature(SNAPSHOT_DATA_PATH))


@st.cache_data(show_spinner=False)
def _load_snapshot_dates(
    snapshot_path_string: str, snapshot_modified_at: int
) -> pd.DataFrame:
    """Load only identity/date columns used to label the latest snapshot."""
    snapshot_path = Path(snapshot_path_string)
    if snapshot_modified_at < 0 or not snapshot_path.exists():
        raise TeamComparisonError("The processed team snapshot dataset is unavailable.")
    try:
        frame = pd.read_parquet(
            snapshot_path, columns=["date", "home_team", "away_team"]
        )
    except (OSError, ValueError, KeyError) as error:
        raise TeamComparisonError(
            "The processed team snapshot dataset has an invalid identity schema."
        ) from error
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.dropna(subset=["date"])


def get_latest_snapshot_date(team: str) -> str | None:
    """Return the date label for a team's latest available engineered snapshot."""
    normalized_team = _clean_team_name(team)
    if normalized_team is None:
        return None
    frame = _load_snapshot_dates(*_path_signature(SNAPSHOT_DATA_PATH))
    folded_name = normalized_team.casefold()
    home = frame["home_team"].astype(str).str.strip().str.casefold().eq(folded_name)
    away = frame["away_team"].astype(str).str.strip().str.casefold().eq(folded_name)
    dates = frame.loc[home | away, "date"]
    if dates.empty:
        return None
    latest = dates.max()
    return latest.date().isoformat() if pd.notna(latest) else None


def get_comparison_teams() -> list[str]:
    """Use the Phase 3 dynamic simulator/snapshot team source for selectors."""
    try:
        teams = normalize_team_names(get_available_teams())
    except MatchPredictionError as error:
        raise TeamComparisonError(str(error)) from error
    if len(teams) < 2:
        raise TeamComparisonError(
            "At least two teams with available snapshots are required."
        )
    return teams


def _read_match_table(path: Path, *, source_label: str) -> tuple[pd.DataFrame, str]:
    """Read a real match-history file and validate its common match columns."""
    required_columns = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
    }
    try:
        if path.suffix.casefold() == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path, parse_dates=["date"])
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise TeamComparisonError(
            "The match-history dataset could not be read."
        ) from error

    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise TeamComparisonError(
            "The match-history dataset is missing required match fields."
        )

    frame = frame.loc[:, sorted(required_columns)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["home_score"] = pd.to_numeric(frame["home_score"], errors="coerce")
    frame["away_score"] = pd.to_numeric(frame["away_score"], errors="coerce")
    frame = frame.dropna(subset=["date", "home_team", "away_team"])
    frame = frame.drop_duplicates(
        subset=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
        ]
    )
    return frame, source_label


@st.cache_data(show_spinner=False)
def _load_match_history(
    history_path_string: str,
    history_modified_at: int,
    snapshot_path_string: str,
    snapshot_modified_at: int,
) -> tuple[pd.DataFrame, str]:
    """Load raw international results, with processed matches as a real fallback."""
    history_path = Path(history_path_string)
    if history_modified_at >= 0 and history_path.exists():
        try:
            return _read_match_table(
                history_path, source_label="International results match history"
            )
        except TeamComparisonError:
            pass

    snapshot_path = Path(snapshot_path_string)
    if snapshot_modified_at >= 0 and snapshot_path.exists():
        return _read_match_table(
            snapshot_path, source_label="Processed international match history"
        )
    raise TeamComparisonError(
        "No real match-history dataset is available for comparison."
    )


def _history_with_source() -> tuple[pd.DataFrame, str]:
    """Return cached historical match records and a safe, displayable source label."""
    return _load_match_history(
        *_path_signature(HISTORY_DATA_PATH), *_path_signature(SNAPSHOT_DATA_PATH)
    )


def _format_date(value: Any) -> str | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.date().isoformat()


def _format_score(value: float | None) -> str | None:
    if value is None:
        return None
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def orient_team_match(row: Mapping[str, Any], team: str) -> dict[str, Any] | None:
    """Orient one real match row around a selected team for recent-form display."""
    selected_team = _clean_team_name(team)
    home_team = _clean_team_name(row.get("home_team"))
    away_team = _clean_team_name(row.get("away_team"))
    if selected_team is None or home_team is None or away_team is None:
        return None

    home_score = _finite_number(row.get("home_score"))
    away_score = _finite_number(row.get("away_score"))
    if home_team.casefold() == selected_team.casefold():
        opponent = away_team
        team_score, opponent_score = home_score, away_score
        venue = "Home"
    elif away_team.casefold() == selected_team.casefold():
        opponent = home_team
        team_score, opponent_score = away_score, home_score
        venue = "Away"
    else:
        return None

    result: str | None = None
    if team_score is not None and opponent_score is not None:
        if team_score > opponent_score:
            result = "W"
        elif team_score < opponent_score:
            result = "L"
        else:
            result = "D"

    return {
        "date": _format_date(row.get("date")),
        "opponent": opponent,
        "venue": venue,
        "team_score": team_score,
        "opponent_score": opponent_score,
        "score": (
            f"{_format_score(team_score)}–{_format_score(opponent_score)}"
            if team_score is not None and opponent_score is not None
            else None
        ),
        "result": result,
        "tournament": _clean_team_name(row.get("tournament")),
    }


def orient_head_to_head_match(
    row: Mapping[str, Any], team_a: str, team_b: str
) -> dict[str, Any] | None:
    """Orient a historical meeting around Team A regardless of original venue."""
    normalized_a, normalized_b = validate_comparison_teams(team_a, team_b)
    home_team = _clean_team_name(row.get("home_team"))
    away_team = _clean_team_name(row.get("away_team"))
    if home_team is None or away_team is None:
        return None

    home_score = _finite_number(row.get("home_score"))
    away_score = _finite_number(row.get("away_score"))
    if (
        home_team.casefold() == normalized_a.casefold()
        and away_team.casefold() == normalized_b.casefold()
    ):
        team_a_score, team_b_score = home_score, away_score
        team_a_venue = "Home"
    elif (
        home_team.casefold() == normalized_b.casefold()
        and away_team.casefold() == normalized_a.casefold()
    ):
        team_a_score, team_b_score = away_score, home_score
        team_a_venue = "Away"
    else:
        return None

    result: str | None = None
    if team_a_score is not None and team_b_score is not None:
        if team_a_score > team_b_score:
            result = "W"
        elif team_a_score < team_b_score:
            result = "L"
        else:
            result = "D"

    return {
        "date": _format_date(row.get("date")),
        "team_a_score": team_a_score,
        "team_b_score": team_b_score,
        "score": (
            f"{_format_score(team_a_score)}–{_format_score(team_b_score)}"
            if team_a_score is not None and team_b_score is not None
            else None
        ),
        "team_a_venue": team_a_venue,
        "result": result,
        "tournament": _clean_team_name(row.get("tournament")),
    }


def _recent_matches(history: pd.DataFrame, team: str) -> list[dict[str, Any]]:
    """Return up to five newest, orientation-safe real match records for one team."""
    folded_name = team.casefold()
    home = history["home_team"].astype(str).str.strip().str.casefold().eq(folded_name)
    away = history["away_team"].astype(str).str.strip().str.casefold().eq(folded_name)
    selected = history.loc[home | away].sort_values("date", ascending=False).head(5)
    records: list[dict[str, Any]] = []
    for record in selected.to_dict(orient="records"):
        oriented = orient_team_match(record, team)
        if oriented is not None:
            records.append(oriented)
    return records


def _head_to_head_history(
    history: pd.DataFrame, team_a: str, team_b: str
) -> dict[str, Any]:
    """Return de-duplicated, venue-normalized historical meetings for two teams."""
    folded_a, folded_b = team_a.casefold(), team_b.casefold()
    home = history["home_team"].astype(str).str.strip().str.casefold()
    away = history["away_team"].astype(str).str.strip().str.casefold()
    selected = history.loc[
        (home.eq(folded_a) & away.eq(folded_b))
        | (home.eq(folded_b) & away.eq(folded_a))
    ].sort_values("date", ascending=False)

    records: list[dict[str, Any]] = []
    for record in selected.to_dict(orient="records"):
        oriented = orient_head_to_head_match(record, team_a, team_b)
        if oriented is not None:
            records.append(oriented)

    team_a_wins = sum(record["result"] == "W" for record in records)
    draws = sum(record["result"] == "D" for record in records)
    team_b_wins = sum(record["result"] == "L" for record in records)
    dates = [record["date"] for record in records if record.get("date")]
    return {
        "meetings": len(records),
        "team_a_wins": team_a_wins,
        "draws": draws,
        "team_b_wins": team_b_wins,
        "records": records[:5],
        "data_range": {
            "start": min(dates) if dates else None,
            "end": max(dates) if dates else None,
        },
    }


def build_comparison_metrics(
    snapshot_a: Mapping[str, Any], snapshot_b: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Create comparable real-feature records from two backend snapshots."""
    metrics: list[dict[str, Any]] = []
    unavailable: list[str] = []
    for key, metadata in FEATURE_CONFIG.items():
        value_a = _finite_number(snapshot_a.get(key))
        value_b = _finite_number(snapshot_b.get(key))
        if value_a is None or value_b is None:
            unavailable.append(metadata.label)
            continue
        normalized_a, normalized_b = normalize_pair(
            value_a, value_b, higher_is_better=metadata.higher_is_better
        )
        relative_gap = abs(value_a - value_b) / max(
            abs(value_a), abs(value_b), abs(value_a - value_b), 1.0
        )
        metrics.append(
            {
                "key": key,
                "label": metadata.label,
                "description": metadata.description,
                "higher_is_better": metadata.higher_is_better,
                "value_a": value_a,
                "value_b": value_b,
                "normalized_a": normalized_a,
                "normalized_b": normalized_b,
                "relative_gap": relative_gap,
                "favorable_side": _favorable_side(normalized_a, normalized_b),
                "show_in_key_cards": metadata.show_in_key_cards,
                "show_in_radar": metadata.show_in_radar,
            }
        )
    return metrics, unavailable


def build_feature_verdict(
    metrics: list[Mapping[str, Any]], team_a: str, team_b: str
) -> dict[str, Any]:
    """Create a deterministic, equal-weight feature-based comparison summary."""
    if not metrics:
        return {
            "leader": None,
            "score_a": None,
            "score_b": None,
            "is_close": True,
            "summary": "No comparable engineered indicators were available for this selection.",
            "supporting_metrics": [],
        }

    score_a = sum(float(metric["normalized_a"]) for metric in metrics) / len(metrics)
    score_b = sum(float(metric["normalized_b"]) for metric in metrics) / len(metrics)
    margin = abs(score_a - score_b)
    is_close = margin <= 0.12
    leader = None if is_close else ("team_a" if score_a > score_b else "team_b")
    leader_name = team_a if leader == "team_a" else team_b

    supporting_metrics = [
        metric for metric in metrics if metric.get("favorable_side") == leader
    ]
    supporting_metrics.sort(
        key=lambda metric: float(metric.get("relative_gap", 0.0)),
        reverse=True,
    )
    supporting_labels = [str(metric["label"]) for metric in supporting_metrics[:2]]

    if is_close:
        summary = (
            "The teams are closely matched across the available engineered indicators. "
            "This equal-weight feature comparison is not a match prediction."
        )
    elif supporting_labels:
        summary = (
            f"{leader_name} leads the feature-based comparison across the available "
            f"normalized indicators, with the clearest separation in "
            f"{', '.join(supporting_labels)}. This is not a match prediction."
        )
    else:
        summary = (
            f"{leader_name} leads the available normalized indicators in this "
            "feature-based comparison. This is not a match prediction."
        )

    return {
        "leader": leader,
        "score_a": score_a,
        "score_b": score_b,
        "is_close": is_close,
        "summary": summary,
        "supporting_metrics": [dict(metric) for metric in supporting_metrics[:3]],
    }


def _strength_label(snapshot: Mapping[str, Any]) -> str | None:
    """Build a concise, factual identity label from available snapshot features."""
    points = _finite_number(snapshot.get("form_points"))
    matches = _finite_number(snapshot.get("form_played"))
    if points is not None and matches is not None:
        return f"Recent form: {points:,.0f} points from {matches:,.0f} matches"
    attack = _finite_number(snapshot.get("attack_strength"))
    if attack is not None:
        return f"Attack strength indicator: {attack:.2f}"
    return None


def _validated_model_outlook(
    team_a: str, team_b: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Reuse the Phase 3 service, keeping model errors separate from comparison data."""
    try:
        result = predict_match(team_a, team_b)
        probabilities = {
            "team_a_win_probability": _finite_number(
                result.get("home_win_probability")
            ),
            "draw_probability": _finite_number(result.get("draw_probability")),
            "team_b_win_probability": _finite_number(
                result.get("away_win_probability")
            ),
        }
        if any(value is None or value < 0 for value in probabilities.values()):
            raise TeamComparisonError(
                "The trained predictor returned invalid probabilities."
            )
        total = sum(
            float(value) for value in probabilities.values() if value is not None
        )
        if total <= 0:
            raise TeamComparisonError(
                "The trained predictor returned an empty probability vector."
            )
        probabilities = {
            key: float(value) / total
            for key, value in probabilities.items()
            if value is not None
        }
        return (
            {
                **probabilities,
                "predicted_outcome": result.get("predicted_outcome"),
                "model_name": result.get("model_name"),
                "model_metrics": result.get("model_metrics", {}),
                "feature_count": result.get("feature_count"),
            },
            None,
        )
    except (MatchPredictionError, TeamComparisonError) as error:
        return None, str(error)
    except Exception:
        return (
            None,
            "The trained match predictor is unavailable for this comparison right now.",
        )


def build_team_comparison(
    team_a: str, team_b: str, *, include_model_outlook: bool = True
) -> dict[str, Any]:
    """Build a normalized UI-safe comparison using only real project data.

    The returned object intentionally contains simple dictionaries and lists so
    it can survive Streamlit reruns without putting dataframes or estimators in
    session state.
    """
    normalized_a, normalized_b = validate_comparison_teams(team_a, team_b)
    snapshot_version = get_snapshot_version()
    snapshot_a = get_team_snapshot(normalized_a)
    snapshot_b = get_team_snapshot(normalized_b)
    metrics, unavailable_features = build_comparison_metrics(snapshot_a, snapshot_b)
    if not metrics:
        raise TeamComparisonError(
            "The selected teams do not share valid engineered comparison features."
        )

    warnings: list[str] = []
    try:
        snapshot_date_a = get_latest_snapshot_date(normalized_a)
        snapshot_date_b = get_latest_snapshot_date(normalized_b)
    except TeamComparisonError:
        snapshot_date_a = None
        snapshot_date_b = None
        warnings.append("Snapshot dates could not be read from the processed dataset.")

    recent_form_a: list[dict[str, Any]] = []
    recent_form_b: list[dict[str, Any]] = []
    head_to_head: dict[str, Any] = {
        "meetings": 0,
        "team_a_wins": 0,
        "draws": 0,
        "team_b_wins": 0,
        "records": [],
        "data_range": {"start": None, "end": None},
    }
    history_source: str | None = None
    try:
        history, history_source = _history_with_source()
        recent_form_a = _recent_matches(history, normalized_a)
        recent_form_b = _recent_matches(history, normalized_b)
        head_to_head = _head_to_head_history(history, normalized_a, normalized_b)
    except TeamComparisonError as error:
        warnings.append(str(error))

    model_outlook: dict[str, Any] | None = None
    model_outlook_error: str | None = None
    if include_model_outlook:
        model_outlook, model_outlook_error = _validated_model_outlook(
            normalized_a, normalized_b
        )
        if model_outlook_error:
            warnings.append(model_outlook_error)

    radar_metrics = [metric for metric in metrics if metric["show_in_radar"]][:8]
    latest_dates = [date for date in (snapshot_date_a, snapshot_date_b) if date]
    return {
        "signature": comparison_signature(normalized_a, normalized_b, snapshot_version),
        "team_a": {
            "name": normalized_a,
            "snapshot_date": snapshot_date_a,
            "strength_label": _strength_label(snapshot_a),
        },
        "team_b": {
            "name": normalized_b,
            "snapshot_date": snapshot_date_b,
            "strength_label": _strength_label(snapshot_b),
        },
        "metrics": metrics,
        "key_metrics": [metric for metric in metrics if metric["show_in_key_cards"]],
        "radar_metrics": radar_metrics,
        "verdict": build_feature_verdict(metrics, normalized_a, normalized_b),
        "recent_form": {"team_a": recent_form_a, "team_b": recent_form_b},
        "head_to_head": head_to_head,
        "model_outlook": model_outlook,
        "model_outlook_error": model_outlook_error,
        "data_transparency": {
            "snapshot_source": "Processed engineered team snapshots",
            "latest_available_date": max(latest_dates) if latest_dates else None,
            "feature_count": len(metrics),
            "unavailable_features": unavailable_features,
            "normalization": (
                "Each available feature is min-max normalized within this two-team "
                "comparison. Equal values receive 0.5; lower-is-better indicators "
                "are inverted; the feature-based score averages the resulting values "
                "with equal weight."
            ),
            "history_source": history_source,
            "model_prediction_included": model_outlook is not None,
            "warnings": warnings,
        },
    }
