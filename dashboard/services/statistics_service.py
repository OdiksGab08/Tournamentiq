"""Load, validate, filter, and summarize verified historical match records.

Purpose:
    Provide one trusted source of historical international-match statistics for
    the Statistics dashboard route.
Responsibility:
    Validate warehouse rows, preserve missing values honestly, apply user
    filters, and derive transparent tables and KPI mappings without mutation.
Inputs:
    The canonical historical parquet artifact and optional normalized filter
    mappings from the Streamlit dashboard.
Outputs:
    Validated match tables, quality metadata, filter choices, aggregates, and
    non-persistent CSV bytes or ``StatisticsDataError`` failures.
Collaboration:
    Consumed by ``components.statistics_dashboard`` and
    ``services.platform_statistics_service``; it never reads engineered model
    inputs as historical match facts.
"""

from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DATASET_PATH = PROJECT_ROOT / "data" / "warehouse" / "master_matches.parquet"
CANONICAL_DATASET_LABEL = "Verified historical international match warehouse"
CANONICAL_DATASET_REASON = (
    "It is the warehouse copy of the project’s cleaned, standardized, verified "
    "match-level dataset. It retains one historical home-versus-away match row "
    "with dates, scores, competition, location, and neutral-venue fields, plus "
    "a stable internal row identifier, without model features."
)
WORLD_CUP_FINALS_LABEL = "FIFA World Cup"
WORLD_CUP_IDENTIFICATION_METHOD = (
    "Exact canonical tournament value ‘FIFA World Cup’; qualification and other "
    "competitions containing ‘World Cup’ are excluded."
)
OUTCOME_ORDER: tuple[str, ...] = ("Home Win", "Draw", "Away Win")

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("date", "match_date", "game_date"),
    "home_team": ("home_team", "home", "team_home"),
    "away_team": ("away_team", "away", "team_away"),
    "home_score": ("home_score", "home_goals", "score_home"),
    "away_score": ("away_score", "away_goals", "score_away"),
    "competition": ("tournament", "competition", "competition_name"),
    "city": ("city", "venue_city"),
    "country": ("country", "venue_country", "host_country"),
    "neutral": ("neutral", "neutral_venue", "is_neutral"),
}
REQUIRED_CONCEPTS: tuple[str, ...] = (
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
)
OPTIONAL_CONCEPTS: tuple[str, ...] = ("competition", "city", "country", "neutral")


class StatisticsDataError(ValueError):
    """A safe, user-facing statistics data or filtering error."""


def _path_version(path: Path) -> int:
    """Return a cheap cache version token without exposing paths in the UI."""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return -1


def _clean_text(series: pd.Series) -> pd.Series:
    """Normalize text values while preserving true missing values."""
    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned.eq(""))


def _resolve_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Resolve source columns only when an alias match is unambiguous."""
    normalized: dict[str, list[str]] = {}
    for column in frame.columns:
        normalized.setdefault(str(column).strip().casefold(), []).append(str(column))

    resolved: dict[str, str] = {}
    for concept, aliases in COLUMN_ALIASES.items():
        matches: list[str] = []
        for alias in aliases:
            matches.extend(normalized.get(alias.casefold(), []))
        unique_matches = list(dict.fromkeys(matches))
        if len(unique_matches) > 1:
            raise StatisticsDataError(
                f"The match dataset has ambiguous columns for {concept}: "
                f"{', '.join(unique_matches)}."
            )
        if unique_matches:
            resolved[concept] = unique_matches[0]

    missing = [concept for concept in REQUIRED_CONCEPTS if concept not in resolved]
    if missing:
        available = ", ".join(map(str, frame.columns)) or "none"
        raise StatisticsDataError(
            "The historical match dataset is missing required fields: "
            f"{', '.join(missing)}. Available columns: {available}."
        )
    return resolved


def _normalize_neutral(series: pd.Series) -> tuple[pd.Series, int]:
    """Return a nullable boolean neutral field and count unrecognized values."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean"), 0

    text = _clean_text(series).str.casefold()
    true_values = {"true", "1", "yes", "y", "neutral"}
    false_values = {"false", "0", "no", "n", "non-neutral", "non neutral"}
    output = pd.Series(pd.NA, index=series.index, dtype="boolean")
    output.loc[text.isin(true_values)] = True
    output.loc[text.isin(false_values)] = False
    invalid = int(text.notna().sum() - text.isin(true_values | false_values).sum())
    return output, invalid


def _empty_optional(index: pd.Index, *, dtype: str = "string") -> pd.Series:
    """Construct an explicit missing optional column without inventing a value."""
    return pd.Series(pd.NA, index=index, dtype=dtype)


def _score_input_missing(series: pd.Series) -> pd.Series:
    """Identify source-score blanks before numeric coercion distinguishes errors."""
    text = series.astype("string").str.strip()
    return text.isna() | text.eq("")


def _source_match_id_column(source: pd.DataFrame) -> str | None:
    """Find the optional warehouse row identifier without treating it as a match key."""
    matches = [
        str(column)
        for column in source.columns
        if str(column).strip().casefold() == "match_id"
    ]
    if len(matches) > 1:
        raise StatisticsDataError("The match dataset has ambiguous match_id columns.")
    return matches[0] if matches else None


def _coerce_world_cup_filter(value: Any) -> bool:
    """Parse a strict World Cup-only flag instead of relying on truthiness."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"", "false", "0", "no"}:
            return False
        if normalized in {"true", "1", "yes"}:
            return True
    raise StatisticsDataError("World Cup-only filter must be a boolean value.")


def prepare_match_table(
    source: pd.DataFrame,
    *,
    source_label: str = CANONICAL_DATASET_LABEL,
    source_file_name: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and derive transparent fields from a raw match-level table.

    Rows with invalid identity/date structure, duplicate records, or a team
    playing itself are excluded from dashboard calculations and counted in the
    returned quality metadata. Rows with missing or invalid scores are retained
    as matches, but excluded from all score-based denominators.
    """
    if not isinstance(source, pd.DataFrame):
        raise StatisticsDataError("The historical dataset did not load as a table.")
    if source.empty:
        raise StatisticsDataError("The historical match dataset is empty.")

    aliases = _resolve_columns(source)
    frame = pd.DataFrame(index=source.index)
    match_id_column = _source_match_id_column(source)
    frame["match_id"] = (
        pd.to_numeric(source[match_id_column], errors="coerce").astype("Int64")
        if match_id_column is not None
        else _empty_optional(frame.index, dtype="Int64")
    )
    frame["date"] = pd.to_datetime(source[aliases["date"]], errors="coerce")
    frame["home_team"] = _clean_text(source[aliases["home_team"]])
    frame["away_team"] = _clean_text(source[aliases["away_team"]])
    frame["home_score"] = pd.to_numeric(source[aliases["home_score"]], errors="coerce")
    frame["away_score"] = pd.to_numeric(source[aliases["away_score"]], errors="coerce")

    for concept in ("competition", "city", "country"):
        source_column = aliases.get(concept)
        frame[concept] = (
            _clean_text(source[source_column])
            if source_column is not None
            else _empty_optional(frame.index)
        )
    neutral_column = aliases.get("neutral")
    if neutral_column is None:
        frame["neutral"] = _empty_optional(frame.index, dtype="boolean")
        invalid_neutral_values = 0
    else:
        frame["neutral"], invalid_neutral_values = _normalize_neutral(
            source[neutral_column]
        )

    source_missing_scores = _score_input_missing(
        source[aliases["home_score"]]
    ) | _score_input_missing(source[aliases["away_score"]])
    missing_scores = frame["home_score"].isna() | frame["away_score"].isna()
    finite_scores = np.isfinite(frame["home_score"].fillna(0)) & np.isfinite(
        frame["away_score"].fillna(0)
    )
    whole_scores = np.isclose(frame["home_score"].fillna(0) % 1, 0) & np.isclose(
        frame["away_score"].fillna(0) % 1, 0
    )
    non_negative_scores = (frame["home_score"].fillna(0) >= 0) & (
        frame["away_score"].fillna(0) >= 0
    )
    frame["score_valid"] = (
        ~missing_scores & finite_scores & whole_scores & non_negative_scores
    )
    invalid_score_values = int((~frame["score_valid"] & ~source_missing_scores).sum())
    missing_score_values = int(source_missing_scores.sum())

    valid_date = frame["date"].notna()
    valid_teams = frame["home_team"].notna() & frame["away_team"].notna()
    self_match = valid_teams & frame["home_team"].eq(frame["away_team"])
    dedupe_columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "competition",
        "city",
        "country",
        "neutral",
    ]
    duplicate = frame.duplicated(subset=dedupe_columns, keep="first")
    fixture_collision = (
        valid_date
        & valid_teams
        & frame.duplicated(subset=["date", "home_team", "away_team"], keep=False)
    )
    valid_match = valid_date & valid_teams & ~self_match & ~duplicate

    frame["total_goals"] = frame["home_score"] + frame["away_score"]
    frame["goal_difference"] = frame["home_score"] - frame["away_score"]
    frame.loc[~frame["score_valid"], ["total_goals", "goal_difference"]] = np.nan
    frame["result"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame.loc[
        frame["score_valid"] & frame["home_score"].gt(frame["away_score"]), "result"
    ] = "Home Win"
    frame.loc[
        frame["score_valid"] & frame["home_score"].eq(frame["away_score"]), "result"
    ] = "Draw"
    frame.loc[
        frame["score_valid"] & frame["home_score"].lt(frame["away_score"]), "result"
    ] = "Away Win"
    frame["both_teams_scored"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    frame.loc[frame["score_valid"], "both_teams_scored"] = frame.loc[
        frame["score_valid"], "home_score"
    ].gt(0) & frame.loc[frame["score_valid"], "away_score"].gt(0)
    frame["either_clean_sheet"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    frame.loc[frame["score_valid"], "either_clean_sheet"] = frame.loc[
        frame["score_valid"], "home_score"
    ].eq(0) | frame.loc[frame["score_valid"], "away_score"].eq(0)
    frame["scoreless_draw"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    frame.loc[frame["score_valid"], "scoreless_draw"] = frame.loc[
        frame["score_valid"], "home_score"
    ].eq(0) & frame.loc[frame["score_valid"], "away_score"].eq(0)
    frame["scoreline"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    valid_score_rows = frame.loc[frame["score_valid"], ["home_score", "away_score"]]
    frame.loc[frame["score_valid"], "scoreline"] = (
        valid_score_rows["home_score"].astype(int).astype(str)
        + "–"
        + valid_score_rows["away_score"].astype(int).astype(str)
    )
    frame["year"] = frame["date"].dt.year.astype("Int64")
    frame["decade"] = ((frame["year"] // 10) * 10).astype("Int64")
    frame["world_cup_finals"] = (
        frame["competition"].str.casefold().eq(WORLD_CUP_FINALS_LABEL.casefold())
    )

    clean = frame.loc[valid_match].copy().sort_values("date").reset_index(drop=True)
    if clean.empty:
        raise StatisticsDataError(
            "No structurally valid historical match rows remain after validation."
        )

    teams = pd.unique(pd.concat([clean["home_team"], clean["away_team"]])).tolist()
    quality = {
        "source_row_count": int(len(source)),
        "valid_match_rows": int(len(clean)),
        "excluded_rows": int((~valid_match).sum()),
        "missing_dates": int((~valid_date).sum()),
        "missing_teams": int((~valid_teams).sum()),
        "self_match_rows": int(self_match.sum()),
        "duplicate_rows_detected": int(duplicate.sum()),
        "oriented_fixture_collision_rows": int(fixture_collision.sum()),
        "oriented_fixture_collision_groups": int(
            frame.loc[fixture_collision]
            .groupby(["date", "home_team", "away_team"], dropna=False)
            .ngroups
        ),
        "missing_score_rows": missing_score_values,
        "invalid_score_rows": invalid_score_values,
        "valid_score_rows": int(clean["score_valid"].sum()),
        "unresolved_competitions": int(clean["competition"].isna().sum()),
        "missing_neutral_rows": int(clean["neutral"].isna().sum()),
        "invalid_neutral_values": invalid_neutral_values,
        "earliest_valid_date": clean["date"].min(),
        "latest_valid_date": clean["date"].max(),
        "unique_teams": int(len(teams)),
        "unique_competitions": int(clean["competition"].nunique(dropna=True)),
        "source_label": source_label,
        "source_file_name": source_file_name,
        "resolved_columns": aliases,
        "has_competition": "competition" in aliases,
        "has_neutral": "neutral" in aliases,
        "world_cup_identification_method": WORLD_CUP_IDENTIFICATION_METHOD,
        "rows_represent_one_match": not bool(fixture_collision.any()),
        "fixture_collision_policy": (
            "Potential same-date, same-home, same-away fixture collisions are retained "
            "because the source has no canonical resolution field for them."
        ),
        "has_source_match_id": match_id_column is not None,
    }
    return clean, quality


@st.cache_data(show_spinner=False)
def _load_cached_match_table(
    path_text: str, version: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read and prepare the canonical table, keyed by its modification time."""
    del version
    path = Path(path_text)
    if not path.exists():
        raise StatisticsDataError(
            "The verified historical match dataset is unavailable."
        )
    if path.suffix.casefold() != ".parquet":
        raise StatisticsDataError(
            "The selected historical dataset has an unsupported type."
        )
    try:
        source = pd.read_parquet(path)
    except (OSError, ValueError, ImportError) as error:
        raise StatisticsDataError(
            "The verified historical match dataset could not be loaded."
        ) from error
    return prepare_match_table(
        source, source_label=CANONICAL_DATASET_LABEL, source_file_name=path.name
    )


def load_historical_matches() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the selected match-level source without exposing a local path to UI code."""
    return _load_cached_match_table(
        str(CANONICAL_DATASET_PATH), _path_version(CANONICAL_DATASET_PATH)
    )


@st.cache_data(show_spinner=False)
def _cached_filter_options(path_text: str, version: int) -> dict[str, Any]:
    """Build static selector options from the cached canonical table."""
    matches, quality = _load_cached_match_table(path_text, version)
    teams = sorted(
        set(matches["home_team"].dropna().tolist())
        | set(matches["away_team"].dropna().tolist()),
        key=str.casefold,
    )
    competitions = sorted(
        matches["competition"].dropna().unique().tolist(), key=str.casefold
    )
    return {
        "teams": teams,
        "competitions": competitions,
        "date_min": matches["date"].min().date(),
        "date_max": matches["date"].max().date(),
        "dataset_version": str(version),
        "has_neutral": bool(
            quality["has_neutral"] and matches["neutral"].notna().any()
        ),
        "has_world_cup_finals": bool(matches["world_cup_finals"].any()),
    }


def get_statistics_filter_options() -> dict[str, Any]:
    """Return cached dynamic filter choices derived from the selected source."""
    return _cached_filter_options(
        str(CANONICAL_DATASET_PATH), _path_version(CANONICAL_DATASET_PATH)
    )


def _coerce_filter_date(value: Any, *, label: str) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        timestamp = pd.Timestamp(value).normalize()
    except (TypeError, ValueError) as error:
        raise StatisticsDataError(f"{label} must be a valid date.") from error
    if pd.isna(timestamp):
        raise StatisticsDataError(f"{label} must be a valid date.")
    return timestamp


def _normalize_string_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        raise StatisticsDataError(
            "Filter values must be text or a list of text values."
        )
    cleaned = {str(item).strip() for item in values if str(item).strip()}
    return tuple(sorted(cleaned, key=str.casefold))


def normalize_statistics_filters(filters: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize UI filter values into one deterministic and validated shape."""
    filters = filters or {}
    start_date = _coerce_filter_date(filters.get("date_start"), label="Start date")
    end_date = _coerce_filter_date(filters.get("date_end"), label="End date")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise StatisticsDataError("Start date cannot be after end date.")
    try:
        minimum_goals = float(filters.get("minimum_total_goals", 0) or 0)
    except (TypeError, ValueError) as error:
        raise StatisticsDataError(
            "Minimum total goals must be a non-negative number."
        ) from error
    if not isfinite(minimum_goals) or minimum_goals < 0:
        raise StatisticsDataError("Minimum total goals must be a non-negative number.")
    neutral = str(filters.get("neutral", "All venues") or "All venues")
    if neutral not in {"All venues", "Neutral venues", "Non-neutral venues"}:
        raise StatisticsDataError("Neutral venue filter is not supported.")
    team = filters.get("team")
    if team is not None:
        team = str(team).strip() or None
    return {
        "date_start": start_date,
        "date_end": end_date,
        "competitions": _normalize_string_values(filters.get("competitions")),
        "team": team,
        "outcomes": _normalize_string_values(filters.get("outcomes")),
        "neutral": neutral,
        "minimum_total_goals": minimum_goals,
        "world_cup_only": _coerce_world_cup_filter(filters.get("world_cup_only")),
    }


def statistics_filter_signature(filters: Mapping[str, Any] | None) -> str:
    """Create a deterministic compact signature for the active historical filters."""
    normalized = normalize_statistics_filters(filters)
    serializable = {
        key: value.isoformat() if isinstance(value, pd.Timestamp) else value
        for key, value in normalized.items()
    }
    encoded = json.dumps(
        serializable, sort_keys=True, default=str, separators=(",", ":")
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def apply_statistics_filters(
    matches: pd.DataFrame, filters: Mapping[str, Any] | None
) -> pd.DataFrame:
    """Apply vectorized filters to a prepared historical match table."""
    normalized = normalize_statistics_filters(filters)
    filtered = matches.copy()
    start_date = normalized["date_start"]
    end_date = normalized["date_end"]
    if start_date is not None:
        filtered = filtered.loc[filtered["date"].ge(start_date)]
    if end_date is not None:
        filtered = filtered.loc[filtered["date"].lt(end_date + pd.Timedelta(days=1))]
    competitions = normalized["competitions"]
    if competitions:
        filtered = filtered.loc[filtered["competition"].isin(competitions)]
    team = normalized["team"]
    if team:
        filtered = filtered.loc[
            filtered["home_team"].eq(team) | filtered["away_team"].eq(team)
        ]
    outcomes = normalized["outcomes"]
    if outcomes:
        unsupported = set(outcomes).difference(OUTCOME_ORDER)
        if unsupported:
            raise StatisticsDataError("An unsupported match outcome was selected.")
        filtered = filtered.loc[filtered["result"].isin(outcomes)]
    if normalized["neutral"] == "Neutral venues":
        filtered = filtered.loc[filtered["neutral"].eq(True)]
    elif normalized["neutral"] == "Non-neutral venues":
        filtered = filtered.loc[filtered["neutral"].eq(False)]
    minimum_goals = float(normalized["minimum_total_goals"])
    if minimum_goals > 0:
        filtered = filtered.loc[
            filtered["score_valid"] & filtered["total_goals"].ge(minimum_goals)
        ]
    if normalized["world_cup_only"]:
        filtered = filtered.loc[filtered["world_cup_finals"]]
    return filtered.copy()


def _scored_matches(matches: pd.DataFrame) -> pd.DataFrame:
    return matches.loc[matches["score_valid"]].copy()


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def outcome_distribution(matches: pd.DataFrame) -> pd.DataFrame:
    """Return real scored-match outcome counts and percentages in a fixed order."""
    scored = _scored_matches(matches)
    counts = scored["result"].value_counts().reindex(OUTCOME_ORDER, fill_value=0)
    total = int(len(scored))
    return pd.DataFrame(
        {
            "outcome": OUTCOME_ORDER,
            "matches": [int(counts[label]) for label in OUTCOME_ORDER],
            "percentage": [
                _safe_rate(int(counts[label]), total) for label in OUTCOME_ORDER
            ],
        }
    )


def calculate_kpis(matches: pd.DataFrame) -> dict[str, Any]:
    """Calculate denominator-safe historical match KPIs from the active filter."""
    scored = _scored_matches(matches)
    outcome = outcome_distribution(matches).set_index("outcome")
    scored_count = int(len(scored))
    total_goals = float(scored["total_goals"].sum()) if scored_count else None
    highest = None
    if scored_count:
        highest_row = scored.sort_values(
            ["total_goals", "date"], ascending=[False, False]
        ).iloc[0]
        highest = {
            "scoreline": str(highest_row["scoreline"]),
            "home_team": str(highest_row["home_team"]),
            "away_team": str(highest_row["away_team"]),
            "total_goals": int(highest_row["total_goals"]),
        }
    unique_teams = len(
        set(matches["home_team"].dropna().tolist())
        | set(matches["away_team"].dropna().tolist())
    )
    return {
        "matches": int(len(matches)),
        "scored_matches": scored_count,
        "total_goals": int(total_goals) if total_goals is not None else None,
        "goals_per_match": _safe_rate(total_goals or 0, scored_count),
        "home_win_rate": outcome.loc["Home Win", "percentage"],
        "draw_rate": outcome.loc["Draw", "percentage"],
        "away_win_rate": outcome.loc["Away Win", "percentage"],
        "both_teams_scored_rate": _safe_rate(
            int(scored["both_teams_scored"].sum()), scored_count
        )
        if scored_count
        else None,
        "clean_sheet_rate": _safe_rate(
            int(scored["either_clean_sheet"].sum()), scored_count
        )
        if scored_count
        else None,
        "scoreless_draw_rate": _safe_rate(
            int(scored["scoreless_draw"].sum()), scored_count
        )
        if scored_count
        else None,
        "highest_scoring_match": highest,
        "unique_teams": int(unique_teams),
        "unique_competitions": int(matches["competition"].nunique(dropna=True)),
    }


def goals_over_time(matches: pd.DataFrame) -> pd.DataFrame:
    """Aggregate only years with genuine scored matches; no missing years are filled."""
    scored = _scored_matches(matches).dropna(subset=["year"])
    if scored.empty:
        return pd.DataFrame(
            columns=["year", "matches", "total_goals", "goals_per_match"]
        )
    summary = (
        scored.groupby("year", as_index=False)
        .agg(matches=("total_goals", "size"), total_goals=("total_goals", "sum"))
        .sort_values("year")
    )
    summary["goals_per_match"] = summary["total_goals"] / summary["matches"]
    return summary


def goal_distribution(
    matches: pd.DataFrame, *, top_scorelines: int = 12
) -> dict[str, Any]:
    """Build chart-ready goal and scoreline distributions from valid scores only."""
    scored = _scored_matches(matches)
    if scored.empty:
        empty = pd.DataFrame()
        return {
            "total_goals": empty,
            "scorelines": empty,
            "over_2_5_rate": None,
            "under_2_5_rate": None,
            "scoreless_draw_rate": None,
        }
    total_goals = (
        scored.groupby("total_goals", as_index=False)
        .size()
        .rename(columns={"size": "matches"})
        .sort_values("total_goals")
    )
    scorelines = (
        scored.groupby("scoreline", as_index=False)
        .size()
        .rename(columns={"size": "matches"})
        .sort_values(["matches", "scoreline"], ascending=[False, True])
        .head(top_scorelines)
    )
    denominator = int(len(scored))
    return {
        "total_goals": total_goals,
        "scorelines": scorelines,
        "over_2_5_rate": _safe_rate(
            int(scored["total_goals"].gt(2.5).sum()), denominator
        ),
        "under_2_5_rate": _safe_rate(
            int(scored["total_goals"].lt(2.5).sum()), denominator
        ),
        "scoreless_draw_rate": _safe_rate(
            int(scored["scoreless_draw"].sum()), denominator
        ),
    }


def home_advantage_summary(matches: pd.DataFrame) -> dict[str, Any]:
    """Summarize home/away orientation without making a causal claim."""
    scored = _scored_matches(matches)
    outcome = outcome_distribution(matches).set_index("outcome")
    result: dict[str, Any] = {
        "scored_matches": int(len(scored)),
        "home_win_rate": outcome.loc["Home Win", "percentage"],
        "draw_rate": outcome.loc["Draw", "percentage"],
        "away_win_rate": outcome.loc["Away Win", "percentage"],
        "average_home_goals": float(scored["home_score"].mean())
        if len(scored)
        else None,
        "average_away_goals": float(scored["away_score"].mean())
        if len(scored)
        else None,
        "average_goal_difference": float(scored["goal_difference"].mean())
        if len(scored)
        else None,
        "neutral_comparison": None,
    }
    neutral_scored = scored.dropna(subset=["neutral"])
    if not neutral_scored.empty:
        grouped = (
            neutral_scored.assign(
                venue_type=np.where(neutral_scored["neutral"], "Neutral", "Non-neutral")
            )
            .groupby("venue_type", as_index=False)
            .agg(
                matches=("total_goals", "size"),
                home_goals=("home_score", "mean"),
                away_goals=("away_score", "mean"),
                average_goal_difference=("goal_difference", "mean"),
            )
        )
        counts = pd.crosstab(
            np.where(neutral_scored["neutral"], "Neutral", "Non-neutral"),
            neutral_scored["result"],
        ).reindex(columns=OUTCOME_ORDER, fill_value=0)
        for label, column in (
            ("Home Win", "home_win_rate"),
            ("Draw", "draw_rate"),
            ("Away Win", "away_win_rate"),
        ):
            grouped[column] = grouped["venue_type"].map(
                (counts[label] / counts.sum(axis=1)).to_dict()
            )
        result["neutral_comparison"] = grouped
    return result


def competition_summary(
    matches: pd.DataFrame, *, minimum_matches: int = 1
) -> pd.DataFrame:
    """Calculate competition statistics from current filtered historical rows."""
    if minimum_matches < 1:
        raise StatisticsDataError("Competition minimum matches must be at least one.")
    base = matches.dropna(subset=["competition"]).copy()
    columns = [
        "competition",
        "matches",
        "scored_matches",
        "total_goals",
        "goals_per_match",
        "home_win_rate",
        "draw_rate",
        "away_win_rate",
        "unique_teams",
        "first_match",
        "last_match",
    ]
    if base.empty:
        return pd.DataFrame(columns=columns)
    overview = base.groupby("competition", as_index=False).agg(
        matches=("date", "size"),
        first_match=("date", "min"),
        last_match=("date", "max"),
    )
    scored = _scored_matches(base)
    scored_summary = scored.groupby("competition", as_index=False).agg(
        scored_matches=("total_goals", "size"), total_goals=("total_goals", "sum")
    )
    outcomes = pd.crosstab(scored["competition"], scored["result"]).reindex(
        columns=OUTCOME_ORDER, fill_value=0
    )
    team_rows = pd.concat(
        [
            base[["competition", "home_team"]].rename(columns={"home_team": "team"}),
            base[["competition", "away_team"]].rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    )
    unique_teams = team_rows.groupby("competition", as_index=False).agg(
        unique_teams=("team", "nunique")
    )
    summary = overview.merge(scored_summary, on="competition", how="left").merge(
        unique_teams, on="competition", how="left"
    )
    summary["scored_matches"] = summary["scored_matches"].fillna(0).astype(int)
    summary["goals_per_match"] = np.where(
        summary["scored_matches"].gt(0),
        summary["total_goals"] / summary["scored_matches"],
        np.nan,
    )
    outcome_map = outcomes.to_dict(orient="index")
    for label, column in (
        ("Home Win", "home_win_rate"),
        ("Draw", "draw_rate"),
        ("Away Win", "away_win_rate"),
    ):
        summary[column] = summary.apply(
            lambda row: (
                _safe_rate(
                    int(outcome_map.get(row["competition"], {}).get(label, 0)),
                    int(row["scored_matches"]),
                )
                if int(row["scored_matches"]) > 0
                else None
            ),
            axis=1,
        )
    summary = summary.loc[summary["matches"].ge(minimum_matches)].copy()
    return summary.sort_values(
        ["matches", "competition"], ascending=[False, True]
    ).reset_index(drop=True)


def team_performance_table(
    matches: pd.DataFrame, *, minimum_matches: int = 1
) -> pd.DataFrame:
    """Aggregate score-valid matches orientation-safely for every participating team."""
    if minimum_matches < 1:
        raise StatisticsDataError("Minimum team matches must be at least one.")
    scored = _scored_matches(matches)
    columns = [
        "team",
        "matches_played",
        "wins",
        "draws",
        "losses",
        "goals_scored",
        "goals_conceded",
        "goal_difference",
        "win_rate",
        "points",
        "points_per_match",
        "competitions",
        "first_match",
        "last_match",
    ]
    if scored.empty:
        return pd.DataFrame(columns=columns)
    home = pd.DataFrame(
        {
            "team": scored["home_team"],
            "wins": scored["result"].eq("Home Win").astype(int),
            "draws": scored["result"].eq("Draw").astype(int),
            "losses": scored["result"].eq("Away Win").astype(int),
            "goals_scored": scored["home_score"],
            "goals_conceded": scored["away_score"],
            "competition": scored["competition"],
            "date": scored["date"],
        }
    )
    away = pd.DataFrame(
        {
            "team": scored["away_team"],
            "wins": scored["result"].eq("Away Win").astype(int),
            "draws": scored["result"].eq("Draw").astype(int),
            "losses": scored["result"].eq("Home Win").astype(int),
            "goals_scored": scored["away_score"],
            "goals_conceded": scored["home_score"],
            "competition": scored["competition"],
            "date": scored["date"],
        }
    )
    oriented = pd.concat([home, away], ignore_index=True)
    summary = oriented.groupby("team", as_index=False).agg(
        wins=("wins", "sum"),
        draws=("draws", "sum"),
        losses=("losses", "sum"),
        goals_scored=("goals_scored", "sum"),
        goals_conceded=("goals_conceded", "sum"),
        competitions=("competition", "nunique"),
        first_match=("date", "min"),
        last_match=("date", "max"),
    )
    summary["matches_played"] = summary[["wins", "draws", "losses"]].sum(axis=1)
    summary["goal_difference"] = summary["goals_scored"] - summary["goals_conceded"]
    summary["win_rate"] = summary["wins"] / summary["matches_played"]
    summary["points"] = 3 * summary["wins"] + summary["draws"]
    summary["points_per_match"] = summary["points"] / summary["matches_played"]
    if (
        not (summary["wins"] + summary["draws"] + summary["losses"])
        .eq(summary["matches_played"])
        .all()
    ):
        raise StatisticsDataError("Team records do not reconcile to matches played.")
    if not np.isclose(
        float(summary["goals_scored"].sum()), float(summary["goals_conceded"].sum())
    ):
        raise StatisticsDataError(
            "Team goal totals do not reconcile across orientations."
        )
    summary = summary.loc[summary["matches_played"].ge(minimum_matches), columns]
    return summary.sort_values(
        ["points_per_match", "goal_difference", "goals_scored", "team"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def team_spotlight(matches: pd.DataFrame, team: str) -> dict[str, Any] | None:
    """Return a compact, filtered historical profile for a selected team."""
    selected = str(team).strip()
    if not selected:
        return None
    team_matches = matches.loc[
        matches["home_team"].eq(selected) | matches["away_team"].eq(selected)
    ].copy()
    if team_matches.empty:
        return None
    table = team_performance_table(team_matches, minimum_matches=1)
    record = table.loc[table["team"].eq(selected)]
    if record.empty:
        return None
    scored = _scored_matches(team_matches).copy()
    is_home = scored["home_team"].eq(selected)
    scored["opponent"] = np.where(is_home, scored["away_team"], scored["home_team"])
    scored["venue_role"] = np.where(is_home, "Home label", "Away label")
    scored["team_goals"] = np.where(is_home, scored["home_score"], scored["away_score"])
    scored["opponent_goals"] = np.where(
        is_home, scored["away_score"], scored["home_score"]
    )
    scored["team_result"] = np.select(
        [
            scored["team_goals"].gt(scored["opponent_goals"]),
            scored["team_goals"].eq(scored["opponent_goals"]),
        ],
        ["Win", "Draw"],
        default="Loss",
    )
    scored["score"] = (
        scored["team_goals"].astype(int).astype(str)
        + "–"
        + scored["opponent_goals"].astype(int).astype(str)
    )
    recent = (
        scored.sort_values("date", ascending=False)
        .head(5)[
            ["date", "opponent", "venue_role", "score", "team_result", "competition"]
        ]
        .reset_index(drop=True)
    )
    opponents = (
        scored.groupby("opponent", as_index=False)
        .size()
        .rename(columns={"size": "matches"})
        .sort_values(["matches", "opponent"], ascending=[False, True])
        .head(8)
        .reset_index(drop=True)
    )
    competition = (
        scored.groupby("competition", dropna=False, as_index=False)
        .agg(
            matches=("date", "size"),
            goals_for=("team_goals", "sum"),
            goals_against=("opponent_goals", "sum"),
        )
        .sort_values(["matches", "competition"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return {
        "record": record.iloc[0].to_dict(),
        "recent_matches": recent,
        "opponents": opponents,
        "competition_breakdown": competition,
        "matches_without_valid_scores": int(len(team_matches) - len(scored)),
    }


def world_cup_historical_summary(matches: pd.DataFrame) -> dict[str, Any] | None:
    """Return finals-only World Cup history when the exact canonical field exists."""
    world_cup = matches.loc[matches["world_cup_finals"]].copy()
    if world_cup.empty:
        return None
    scored = _scored_matches(world_cup)
    high_scoring = scored.sort_values(
        ["total_goals", "date"], ascending=[False, False]
    ).head(8)[
        ["date", "home_team", "away_team", "scoreline", "total_goals", "competition"]
    ]
    return {
        "matches": int(len(world_cup)),
        "kpis": calculate_kpis(world_cup),
        "outcomes": outcome_distribution(world_cup),
        "leading_teams": team_performance_table(world_cup, minimum_matches=1).head(10),
        "highest_scoring_matches": high_scoring.reset_index(drop=True),
        "yearly": goals_over_time(world_cup),
        "identification_method": WORLD_CUP_IDENTIFICATION_METHOD,
    }


def scoreline_explorer(matches: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Expose real scoreline, high-goal, margin, and draw-score tables."""
    scored = _scored_matches(matches)
    columns = ["date", "home_team", "away_team", "scoreline", "competition"]
    if scored.empty:
        empty = pd.DataFrame(columns=columns)
        return {
            "most_common_scorelines": pd.DataFrame(columns=["scoreline", "matches"]),
            "highest_scoring_matches": empty,
            "biggest_margins": empty,
            "draw_scorelines": pd.DataFrame(columns=["scoreline", "matches"]),
        }
    common = (
        scored.groupby("scoreline", as_index=False)
        .size()
        .rename(columns={"size": "matches"})
        .sort_values(["matches", "scoreline"], ascending=[False, True])
        .head(12)
        .reset_index(drop=True)
    )
    highest = (
        scored.sort_values(["total_goals", "date"], ascending=[False, False])
        .head(10)[columns]
        .reset_index(drop=True)
    )
    margins = (
        scored.loc[scored["goal_difference"].ne(0)]
        .assign(winning_margin=lambda frame: frame["goal_difference"].abs())
        .sort_values(["winning_margin", "date"], ascending=[False, False])
        .head(10)[columns + ["winning_margin"]]
        .reset_index(drop=True)
    )
    draws = (
        scored.loc[scored["result"].eq("Draw")]
        .groupby("scoreline", as_index=False)
        .size()
        .rename(columns={"size": "matches"})
        .sort_values(["matches", "scoreline"], ascending=[False, True])
        .head(10)
        .reset_index(drop=True)
    )
    return {
        "most_common_scorelines": common,
        "highest_scoring_matches": highest,
        "biggest_margins": margins,
        "draw_scorelines": draws,
    }


def export_match_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """Return readable current-filter match data without local paths or model fields."""
    columns = [
        "match_id",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "scoreline",
        "result",
        "total_goals",
        "competition",
        "city",
        "country",
        "neutral",
        "world_cup_finals",
        "score_valid",
    ]
    output = matches.reindex(columns=columns).copy()
    return output.rename(
        columns={
            "home_team": "home_team",
            "away_team": "away_team",
            "home_score": "home_score",
            "away_score": "away_score",
            "competition": "competition",
            "world_cup_finals": "fifa_world_cup_finals",
        }
    )


def dataframe_csv(frame: pd.DataFrame) -> bytes:
    """Encode an existing current UI table as a non-persistent CSV download."""
    try:
        return frame.to_csv(index=False).encode("utf-8")
    except (OSError, ValueError, TypeError) as error:
        raise StatisticsDataError(
            "The requested statistics export could not be created."
        ) from error
