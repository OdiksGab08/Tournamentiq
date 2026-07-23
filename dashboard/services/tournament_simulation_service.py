"""Adapt one real knockout-tournament engine run for Streamlit presentation.

Purpose:
    Expose the retained ``TournamentEngine`` as a safe, validated single-run
    tournament simulation without changing its bracket or prediction behavior.
Responsibility:
    Validate fixed configuration and seeds, normalize engine output, and create
    stable result signatures and exports for session-local dashboard state.
Inputs:
    Optional user seed values, existing tournament configuration, snapshots,
    cached predictor resources, and raw engine result mappings.
Outputs:
    Validated overview/preflight data, normalized rounds and champion paths,
    export frames, and user-facing ``TournamentSimulationError`` failures.
Collaboration:
    Consumed by ``components.tournament_simulation`` and reused by the Monte
    Carlo service for compatible tournament metadata and validation.
"""

from __future__ import annotations

# Timestamp completed simulations in a timezone-aware, export-safe format.
from datetime import datetime, timezone
from math import isfinite, log2
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from uuid import uuid4

# Build result tables and cache static tournament configuration across reruns.
import pandas as pd
import streamlit as st

from src.config.deployment import find_project_root, log_exception
from .match_prediction_service import (
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
    MatchPredictionError,
    get_model_metadata,
    get_predictor,
)
from .team_comparison_service import (
    TeamComparisonError,
    get_latest_snapshot_date,
    get_team_snapshot,
)


# Resolve required snapshot data from the project root rather than CWD.
PROJECT_ROOT = find_project_root(__file__)
SNAPSHOT_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "final_training_dataset.parquet"
)


class TournamentSimulationError(ValueError):
    """A safe, user-facing failure raised by the tournament dashboard adapter."""


def _path_signature(path: Path) -> tuple[str, int]:
    """Return an inexpensive version token for cache and stale-result checks."""
    try:
        return str(path), path.stat().st_mtime_ns
    except OSError:
        return str(path), -1


def _clean_team(value: Any) -> str | None:
    """Return a valid display team name or ``None`` for unusable configuration data."""
    if not isinstance(value, str):
        return None
    team = value.strip()
    return team or None


def _team_key(team: str) -> str:
    return team.strip().casefold()


def _safe_text(value: Any) -> str | None:
    """Return a trimmed display string without manufacturing a missing field."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _finite_probability(value: Any) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise TournamentSimulationError(
            "The tournament engine returned a non-numeric probability."
        ) from error
    if not isfinite(probability) or probability < 0:
        raise TournamentSimulationError(
            "The tournament engine returned an invalid probability."
        )
    return probability


def normalize_probabilities(values: Mapping[str, Any]) -> dict[str, float]:
    """Normalize decimal or percentage probability values to a valid total of 1.

    The function intentionally never substitutes equal probabilities.  Inputs
    must all be finite non-negative numbers and have a positive total.
    """
    if not values:
        raise TournamentSimulationError("A probability vector is required.")
    normalized_values = {
        str(label): _finite_probability(value) for label, value in values.items()
    }
    total = sum(normalized_values.values())
    if not isfinite(total) or total <= 0:
        raise TournamentSimulationError(
            "The tournament engine returned an empty probability vector."
        )
    return {label: value / total for label, value in normalized_values.items()}


def parse_simulation_seed(value: str | int | None) -> int | None:
    """Parse an optional integer seed supported by ProbabilitySimulator."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TournamentSimulationError(
            "The optional simulation seed must be an integer."
        )
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError as error:
            raise TournamentSimulationError(
                "The optional simulation seed must be a whole number."
            ) from error
    raise TournamentSimulationError("The optional simulation seed must be an integer.")


# The fixed tournament field has no user-specific state, so cache it safely.
@st.cache_data(show_spinner=False)
def _load_engine_configuration() -> dict[str, Any]:
    """Read the active fixed tournament field directly from TournamentEngine."""
    # Import lazily so page metadata can load without constructing the large model.
    try:
        from src.simulator.tournament_engine import TournamentEngine

        configuration = TournamentEngine.tournament_configuration()
    except Exception:
        # Keep the complete underlying traceback in local and Community Cloud logs.
        log_exception("tournament engine configuration load")
        raise

    return {
        **configuration,
        "rounds": list(configuration.get("rounds", ())),
        "quarter_final_fixtures": [
            list(fixture) for fixture in configuration.get("quarter_final_fixtures", ())
        ],
        "teams": list(configuration.get("teams", ())),
    }


def validate_tournament_configuration(configuration: Mapping[str, Any]) -> list[str]:
    """Validate the exact backend-supported knockout field and first-round bracket."""
    raw_teams = configuration.get("teams")
    raw_fixtures = configuration.get("quarter_final_fixtures")
    raw_rounds = configuration.get("rounds")
    declared_count = configuration.get("team_count")

    if not isinstance(raw_teams, Sequence) or isinstance(raw_teams, str):
        raise TournamentSimulationError(
            "The tournament engine did not return a valid team field."
        )
    teams = [_clean_team(team) for team in raw_teams]
    if any(team is None for team in teams):
        raise TournamentSimulationError(
            "The tournament field contains an invalid team name."
        )
    clean_teams = [str(team) for team in teams]
    if len(clean_teams) < 2 or len(clean_teams) & (len(clean_teams) - 1):
        raise TournamentSimulationError(
            "The tournament engine returned an unsupported knockout bracket size."
        )
    if len({_team_key(team) for team in clean_teams}) != len(clean_teams):
        raise TournamentSimulationError(
            "The tournament field contains duplicate teams."
        )
    if declared_count != len(clean_teams):
        raise TournamentSimulationError(
            "The tournament engine team count does not match its active field."
        )

    if not isinstance(raw_fixtures, Sequence) or isinstance(raw_fixtures, str):
        raise TournamentSimulationError(
            "The tournament engine returned an invalid opening bracket."
        )
    fixtures: list[tuple[str, str]] = []
    for fixture in raw_fixtures:
        if (
            not isinstance(fixture, Sequence)
            or isinstance(fixture, str)
            or len(fixture) != 2
        ):
            raise TournamentSimulationError(
                "The tournament engine returned an invalid opening fixture."
            )
        home, away = _clean_team(fixture[0]), _clean_team(fixture[1])
        if home is None or away is None or _team_key(home) == _team_key(away):
            raise TournamentSimulationError(
                "The tournament engine returned an invalid opening fixture."
            )
        fixtures.append((home, away))

    fixture_teams = [team for fixture in fixtures for team in fixture]
    if fixture_teams != clean_teams:
        raise TournamentSimulationError(
            "The configured opening bracket does not match the active tournament field."
        )
    if len(fixtures) != len(clean_teams) // 2:
        raise TournamentSimulationError(
            "The tournament engine opening bracket has an invalid number of fixtures."
        )

    if not isinstance(raw_rounds, Sequence) or isinstance(raw_rounds, str):
        raise TournamentSimulationError(
            "The tournament engine did not return valid round names."
        )
    expected_round_count = int(log2(len(clean_teams)))
    if len(raw_rounds) != expected_round_count or any(
        _safe_text(round_name) is None for round_name in raw_rounds
    ):
        raise TournamentSimulationError(
            "The tournament engine round configuration does not match its bracket size."
        )
    return clean_teams


def get_tournament_configuration() -> dict[str, Any]:
    """Return the validated configuration supplied by the existing engine."""
    configuration = _load_engine_configuration()
    validate_tournament_configuration(configuration)
    return configuration


def get_tournament_overview() -> dict[str, Any]:
    """Load static field and snapshot labels without loading the trained model."""
    configuration = get_tournament_configuration()
    teams = validate_tournament_configuration(configuration)
    snapshot_dates: dict[str, str | None] = {}
    warnings: list[str] = []
    # Validate every configured team has the real engineered snapshot needed for prediction.
    for team in teams:
        try:
            snapshot_dates[team] = get_latest_snapshot_date(team)
        except TeamComparisonError:
            snapshot_dates[team] = None
            warnings.append(f"The latest snapshot date is unavailable for {team}.")

    available_dates = [date for date in snapshot_dates.values() if date]
    return {
        "configuration": configuration,
        "teams": teams,
        "latest_snapshot_date": max(available_dates) if available_dates else None,
        "snapshot_dates": snapshot_dates,
        "warnings": warnings,
    }


def tournament_signature(seed: int | None) -> str:
    """Create a configuration version token that prevents stale result display."""
    configuration = get_tournament_configuration()
    teams = validate_tournament_configuration(configuration)
    snapshot_version = _path_signature(SNAPSHOT_DATA_PATH)[1]
    model_version = _path_signature(BEST_MODEL_PATH)[1]
    seed_label = "random" if seed is None else str(seed)
    return "::".join(
        (
            "|".join(_team_key(team) for team in teams),
            str(snapshot_version),
            str(model_version),
            seed_label,
        )
    )


def is_tournament_result_current(
    result: Mapping[str, Any] | None, signature: str
) -> bool:
    """Check that a saved simulation belongs to the active field and seed."""
    return bool(result and result.get("signature") == signature)


def _validate_team_snapshots(teams: Sequence[str]) -> None:
    """Ensure each configured side has the backend's complete live snapshot."""
    for team in teams:
        try:
            snapshot = get_team_snapshot(team)
        except TeamComparisonError as error:
            raise TournamentSimulationError(
                f"The engineered snapshot for {team} is unavailable."
            ) from error
        if not snapshot:
            raise TournamentSimulationError(
                f"The engineered snapshot for {team} is empty."
            )


def validate_tournament_preflight() -> dict[str, Any]:
    """Run non-stochastic checks before the model and engine begin a tournament."""
    configuration = get_tournament_configuration()
    teams = validate_tournament_configuration(configuration)
    if not BEST_MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise TournamentSimulationError(
            "The trained model or preprocessor file is missing from the models directory."
        )
    _validate_team_snapshots(teams)
    return {"configuration": configuration, "teams": teams}


def _normalize_three_outcome_probabilities(
    raw_match: Mapping[str, Any],
) -> tuple[float | None, float | None, float | None]:
    """Normalize real raw home/draw/away model values when all are exposed."""
    raw_values = {
        "home": raw_match.get("home_win_probability"),
        "draw": raw_match.get("draw_probability"),
        "away": raw_match.get("away_win_probability"),
    }
    present = [value is not None for value in raw_values.values()]
    if not any(present):
        return None, None, None
    if not all(present):
        raise TournamentSimulationError(
            "The tournament engine returned an incomplete three-outcome probability vector."
        )
    normalized = normalize_probabilities(raw_values)
    return normalized["home"], normalized["draw"], normalized["away"]


def _normalize_advancement_probabilities(
    raw_match: Mapping[str, Any],
) -> tuple[float | None, float | None]:
    """Normalize the draw-adjusted advancement values used by ProbabilitySimulator."""
    raw_values = {
        "home": raw_match.get("home_probability"),
        "away": raw_match.get("away_probability"),
    }
    present = [value is not None for value in raw_values.values()]
    if not any(present):
        return None, None
    if not all(present):
        raise TournamentSimulationError(
            "The tournament engine returned incomplete knockout advancement probabilities."
        )
    normalized = normalize_probabilities(raw_values)
    return normalized["home"], normalized["away"]


def _round_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-") or "round"


def _normalize_match(
    raw_match: Mapping[str, Any], *, round_name: str, match_number: int
) -> dict[str, Any]:
    """Validate and normalize one actual engine match record for the UI."""
    home_team = _clean_team(raw_match.get("home_team"))
    away_team = _clean_team(raw_match.get("away_team"))
    winner = _clean_team(raw_match.get("winner"))
    if home_team is None or away_team is None or winner is None:
        raise TournamentSimulationError(
            "The tournament engine returned an incomplete match record."
        )
    if _team_key(home_team) == _team_key(away_team):
        raise TournamentSimulationError(
            "The tournament engine returned a fixture with duplicate teams."
        )
    if _team_key(winner) not in {_team_key(home_team), _team_key(away_team)}:
        raise TournamentSimulationError(
            "The tournament engine returned a winner outside the fixture."
        )

    home_win_probability, draw_probability, away_win_probability = (
        _normalize_three_outcome_probabilities(raw_match)
    )
    home_advancement_probability, away_advancement_probability = (
        _normalize_advancement_probabilities(raw_match)
    )
    resolution = _safe_text(raw_match.get("resolution"))
    return {
        "match_id": f"{_round_slug(round_name)}-{match_number}",
        "round": round_name,
        "home_team": home_team,
        "away_team": away_team,
        "home_probability": home_win_probability,
        "draw_probability": draw_probability,
        "away_probability": away_win_probability,
        "home_advancement_probability": home_advancement_probability,
        "away_advancement_probability": away_advancement_probability,
        "winner": winner,
        "score": _safe_text(raw_match.get("score")),
        "resolution": resolution,
    }


def _validate_round_progression(rounds: Sequence[Mapping[str, Any]]) -> None:
    """Ensure advancing teams become the exact following-round fixtures in order."""
    for current_round, next_round in zip(rounds, rounds[1:]):
        winners = [str(match["winner"]) for match in current_round["matches"]]
        expected_next_teams = winners
        actual_next_teams = [
            team
            for match in next_round["matches"]
            for team in (str(match["home_team"]), str(match["away_team"]))
        ]
        if [_team_key(team) for team in actual_next_teams] != [
            _team_key(team) for team in expected_next_teams
        ]:
            raise TournamentSimulationError(
                "The tournament engine returned an inconsistent advancing-team bracket."
            )


def _matchup_label(match: Mapping[str, Any]) -> str:
    return f"{match['home_team']} vs {match['away_team']}"


def build_round_summary(round_data: Mapping[str, Any]) -> dict[str, Any]:
    """Derive transparent round metrics from real match advancement probabilities.

    An upset is counted only when the sampled advancing team had a strictly
    lower draw-adjusted advancement probability than its opponent.
    """
    matches = list(round_data.get("matches", []))
    advancing_teams = [str(match["winner"]) for match in matches]
    comparable_matches = [
        match
        for match in matches
        if match.get("home_advancement_probability") is not None
        and match.get("away_advancement_probability") is not None
    ]

    favorite_rows: list[dict[str, Any]] = []
    upsets = 0
    for match in comparable_matches:
        home_probability = float(match["home_advancement_probability"])
        away_probability = float(match["away_advancement_probability"])
        home_team = str(match["home_team"])
        away_team = str(match["away_team"])
        favorite_team = home_team if home_probability >= away_probability else away_team
        favorite_probability = max(home_probability, away_probability)
        winner_probability = (
            home_probability
            if _team_key(str(match["winner"])) == _team_key(home_team)
            else away_probability
        )
        opponent_probability = (
            away_probability
            if _team_key(str(match["winner"])) == _team_key(home_team)
            else home_probability
        )
        if winner_probability < opponent_probability:
            upsets += 1
        favorite_rows.append(
            {
                "match_id": match["match_id"],
                "matchup": _matchup_label(match),
                "favorite_team": favorite_team,
                "favorite_probability": favorite_probability,
                "winner": match["winner"],
                "difference": abs(home_probability - away_probability),
            }
        )

    highest_confidence = (
        max(favorite_rows, key=lambda row: float(row["favorite_probability"]))
        if favorite_rows
        else None
    )
    closest_match = (
        min(favorite_rows, key=lambda row: float(row["difference"]))
        if favorite_rows
        else None
    )
    return {
        "round": round_data.get("name"),
        "match_count": len(matches),
        "advancing_teams": advancing_teams,
        "average_favorite_probability": (
            sum(float(row["favorite_probability"]) for row in favorite_rows)
            / len(favorite_rows)
            if favorite_rows
            else None
        ),
        "upsets": upsets if favorite_rows else None,
        "highest_confidence": highest_confidence,
        "closest_match": closest_match,
    }


def extract_champion_path(
    rounds: Sequence[Mapping[str, Any]], champion: str
) -> list[dict[str, Any]]:
    """Extract the winner's actual chronological path through normalized rounds."""
    path: list[dict[str, Any]] = []
    champion_key = _team_key(champion)
    for round_data in rounds:
        match = next(
            (
                candidate
                for candidate in round_data["matches"]
                if champion_key
                in {
                    _team_key(str(candidate["home_team"])),
                    _team_key(str(candidate["away_team"])),
                }
            ),
            None,
        )
        if match is None:
            raise TournamentSimulationError(
                "The simulated champion does not have a complete tournament path."
            )
        opponent = (
            match["away_team"]
            if champion_key == _team_key(str(match["home_team"]))
            else match["home_team"]
        )
        path.append(
            {
                "round": round_data["name"],
                "opponent": opponent,
                "winner": match["winner"],
                "home_probability": match.get("home_probability"),
                "draw_probability": match.get("draw_probability"),
                "away_probability": match.get("away_probability"),
                "home_advancement_probability": match.get(
                    "home_advancement_probability"
                ),
                "away_advancement_probability": match.get(
                    "away_advancement_probability"
                ),
                "score": match.get("score"),
                "resolution": match.get("resolution"),
            }
        )
    return path


def normalize_tournament_result(
    raw_result: Mapping[str, Any],
    configuration: Mapping[str, Any],
    *,
    simulation_id: str,
    simulated_at: str,
    seed: int | None,
    model_name: str | None,
) -> dict[str, Any]:
    """Normalize and validate a detailed real engine output for Streamlit state."""
    teams = validate_tournament_configuration(configuration)
    raw_rounds = raw_result.get("rounds")
    configured_rounds = list(configuration.get("rounds", []))
    if not isinstance(raw_rounds, Sequence) or isinstance(raw_rounds, str):
        raise TournamentSimulationError("The tournament engine returned no round data.")
    if len(raw_rounds) != len(configured_rounds):
        raise TournamentSimulationError(
            "The tournament engine returned an incomplete knockout bracket."
        )

    rounds: list[dict[str, Any]] = []
    for round_index, (raw_round, expected_name) in enumerate(
        zip(raw_rounds, configured_rounds), start=1
    ):
        if (
            not isinstance(raw_round, Mapping)
            or _safe_text(raw_round.get("name")) != expected_name
        ):
            raise TournamentSimulationError(
                "The tournament engine returned rounds in an unsupported order."
            )
        raw_matches = raw_round.get("matches")
        expected_match_count = len(teams) // (2**round_index)
        if (
            not isinstance(raw_matches, Sequence)
            or isinstance(raw_matches, str)
            or len(raw_matches) != expected_match_count
        ):
            raise TournamentSimulationError(
                "The tournament engine returned an incomplete round of fixtures."
            )
        matches = [
            _normalize_match(raw_match, round_name=expected_name, match_number=index)
            for index, raw_match in enumerate(raw_matches, start=1)
            if isinstance(raw_match, Mapping)
        ]
        if len(matches) != len(raw_matches):
            raise TournamentSimulationError(
                "The tournament engine returned a malformed fixture record."
            )
        rounds.append({"name": expected_name, "matches": matches})

    opening_teams = [
        team
        for match in rounds[0]["matches"]
        for team in (match["home_team"], match["away_team"])
    ]
    if [_team_key(team) for team in opening_teams] != [
        _team_key(team) for team in teams
    ]:
        raise TournamentSimulationError(
            "The tournament engine opening fixtures differ from its active configuration."
        )
    _validate_round_progression(rounds)

    final_match = rounds[-1]["matches"][0]
    champion = _clean_team(raw_result.get("champion")) or str(final_match["winner"])
    if _team_key(champion) != _team_key(str(final_match["winner"])):
        raise TournamentSimulationError(
            "The tournament engine champion does not match the final winner."
        )
    runner_up = (
        str(final_match["away_team"])
        if _team_key(champion) == _team_key(str(final_match["home_team"]))
        else str(final_match["home_team"])
    )
    semifinalists = [
        team
        for match in rounds[-2]["matches"]
        for team in (str(match["home_team"]), str(match["away_team"]))
    ]
    round_summaries = [build_round_summary(round_data) for round_data in rounds]
    champion_path = extract_champion_path(rounds, champion)

    draw_resolution = _safe_text(raw_result.get("draw_resolution")) or _safe_text(
        configuration.get("draw_resolution")
    )
    scores_generated = bool(
        raw_result.get("scores_generated", configuration.get("scores_generated", False))
    )
    missing_fields = [
        "Scorelines" if not scores_generated else None,
        "Extra-time and penalty details"
        if draw_resolution == "draw_probability_split_evenly_then_sampled"
        else None,
    ]
    return {
        "tournament_name": _safe_text(raw_result.get("tournament_name"))
        or str(configuration.get("tournament_name")),
        "format": _safe_text(raw_result.get("format"))
        or str(configuration.get("format")),
        "simulation_id": simulation_id,
        "simulated_at": simulated_at,
        "seed": seed,
        "teams": teams,
        "rounds": rounds,
        "round_summaries": round_summaries,
        "champion": champion,
        "runner_up": runner_up,
        "semifinalists": semifinalists,
        "champion_path": champion_path,
        "model_name": model_name,
        "match_count": sum(len(round_data["matches"]) for round_data in rounds),
        "data_transparency": {
            "format": _safe_text(raw_result.get("format"))
            or str(configuration.get("format")),
            "team_source": "Configured TournamentEngine knockout field",
            "feature_snapshot_source": "LiveSnapshot engineered team snapshots",
            "simulation_method": "Trained Predictor with ProbabilitySimulator sampling",
            "draw_resolution": draw_resolution,
            "scores_generated": scores_generated,
            "randomness_involved": True,
            "seed": seed,
            "missing_fields": [field for field in missing_fields if field],
        },
    }


def run_tournament_simulation(seed: int | None = None) -> dict[str, Any]:
    """Run one real detailed tournament without caching its stochastic result."""
    parsed_seed = parse_simulation_seed(seed)
    preflight = validate_tournament_preflight()
    configuration = preflight["configuration"]
    try:
        predictor = get_predictor()
        from src.simulator.tournament_engine import TournamentEngine

        engine = TournamentEngine(predictor=predictor, seed=parsed_seed)
        raw_result = engine.simulate_detailed()
    except MatchPredictionError as error:
        raise TournamentSimulationError(str(error)) from error
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
        raise TournamentSimulationError(
            "The tournament engine could not build features or complete the configured knockout run."
        ) from error

    metadata = get_model_metadata()
    model_name = metadata.get("model_name") or type(predictor.model).__name__
    return normalize_tournament_result(
        raw_result,
        configuration,
        simulation_id=uuid4().hex,
        simulated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        seed=parsed_seed,
        model_name=model_name,
    )


def tournament_result_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Create a tabular export from the actual normalized current simulation."""
    rows: list[dict[str, Any]] = []
    for round_data in result.get("rounds", []):
        for match in round_data.get("matches", []):
            rows.append(
                {
                    "simulation_id": result.get("simulation_id"),
                    "simulated_at": result.get("simulated_at"),
                    "round": round_data.get("name"),
                    "match_id": match.get("match_id"),
                    "home_team": match.get("home_team"),
                    "away_team": match.get("away_team"),
                    "home_win_probability": match.get("home_probability"),
                    "draw_probability": match.get("draw_probability"),
                    "away_win_probability": match.get("away_probability"),
                    "home_advancement_probability": match.get(
                        "home_advancement_probability"
                    ),
                    "away_advancement_probability": match.get(
                        "away_advancement_probability"
                    ),
                    "simulated_winner": match.get("winner"),
                    "score": match.get("score"),
                    "resolution": match.get("resolution"),
                }
            )
    return pd.DataFrame(rows)


def tournament_result_csv(result: Mapping[str, Any]) -> bytes:
    """Return UTF-8 CSV bytes for the precise current normalized tournament run."""
    return tournament_result_frame(result).to_csv(index=False).encode("utf-8")
