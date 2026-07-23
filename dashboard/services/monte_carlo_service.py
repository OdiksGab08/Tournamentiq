"""Adapt the existing Monte Carlo simulator into validated dashboard analysis.

Purpose:
    Execute and normalize repeated real tournament simulations for the Monte
    Carlo page while preserving the simulator's original business logic.
Responsibility:
    Validate UI settings, run preflight checks, delegate simulation work, and
    transform detailed outputs into stable, exportable UI records.
Inputs:
    Simulation counts, optional seeds, cached predictor resources, tournament
    configuration, and callbacks supplied by the Streamlit presentation layer.
Outputs:
    Normalized rankings, confidence intervals, result signatures, export frames,
    and safe ``MonteCarloAnalysisError`` failures.
Collaboration:
    Uses tournament and match-prediction services plus
    ``src.simulator.monte_carlo``; consumed by ``components.monte_carlo_analysis``.
"""

from __future__ import annotations

# Timestamp a completed analysis in a portable UTC representation.
from datetime import datetime, timezone
import json
from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

# Build exportable tables from validated simulation results.
import pandas as pd

from .match_prediction_service import (
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
    MatchPredictionError,
    get_model_metadata,
    get_predictor,
)
from .tournament_simulation_service import (
    SNAPSHOT_DATA_PATH,
    TournamentSimulationError,
    get_tournament_overview,
    parse_simulation_seed,
    validate_tournament_preflight,
)


# Limit interactive work to tested run sizes that keep the UI responsive.
SUPPORTED_SIMULATION_COUNTS: tuple[int, ...] = (100, 250, 500, 1000)
WILSON_Z_95 = 1.959963984540054
ProgressCallback = Callable[[int, int], None]


class MonteCarloAnalysisError(ValueError):
    """A safe user-facing error raised when an analysis cannot be normalized."""


def _path_version(path: Path) -> int:
    # Reuse the tournament service's parser so both pages accept the same seed values.
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return -1


def _team_key(team: str) -> str:
    return team.strip().casefold()


def _clean_team(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _finite_number(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise MonteCarloAnalysisError(
            f"{field} contains a non-numeric value."
        ) from error
    if not isfinite(number):
        raise MonteCarloAnalysisError(f"{field} contains a non-finite value.")
    return number


def _safe_int(value: Any, *, field: str) -> int:
    number = _finite_number(value, field=field)
    if number < 0 or not number.is_integer():
        raise MonteCarloAnalysisError(
            f"{field} must contain non-negative whole counts."
        )
    return int(number)


def validate_simulation_count(
    value: Any, *, allowed_counts: Sequence[int] = SUPPORTED_SIMULATION_COUNTS
) -> int:
    """Validate bounded interactive count options without accepting booleans."""
    if isinstance(value, bool):
        raise MonteCarloAnalysisError("Choose a positive supported simulation count.")
    try:
        simulation_count = int(value)
    except (TypeError, ValueError) as error:
        raise MonteCarloAnalysisError(
            "Choose a positive supported simulation count."
        ) from error
    if simulation_count <= 0 or simulation_count not in set(allowed_counts):
        options = ", ".join(f"{count:,}" for count in allowed_counts)
        raise MonteCarloAnalysisError(
            f"Choose one of the supported simulation counts: {options}."
        )
    return simulation_count


def parse_monte_carlo_seed(value: str | int | None) -> int | None:
    """Wrap the real underlying seed parser with page-specific errors."""
    try:
        return parse_simulation_seed(value)
    except TournamentSimulationError as error:
        raise MonteCarloAnalysisError(str(error)) from error


def _find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    lookup = {str(column).strip().casefold(): str(column) for column in frame.columns}
    for candidate in candidates:
        column = lookup.get(candidate.casefold())
        if column:
            return column
    return None


def _as_tabular_frame(raw_result: Any) -> pd.DataFrame:
    """Convert supported legacy output containers into a tabular result safely."""
    if isinstance(raw_result, pd.DataFrame):
        return raw_result.copy()
    if isinstance(raw_result, pd.Series):
        return raw_result.rename("Titles").rename_axis("Team").reset_index()
    if isinstance(raw_result, Mapping):
        values = list(raw_result.values())
        if values and all(
            not isinstance(value, (Mapping, list, tuple)) for value in values
        ):
            return pd.DataFrame({"Team": list(raw_result.keys()), "Titles": values})
        try:
            return pd.DataFrame(raw_result)
        except (TypeError, ValueError) as error:
            raise MonteCarloAnalysisError(
                "The Monte Carlo backend returned an unsupported mapping schema."
            ) from error
    if isinstance(raw_result, list):
        try:
            return pd.DataFrame(raw_result)
        except (TypeError, ValueError) as error:
            raise MonteCarloAnalysisError(
                "The Monte Carlo backend returned an unsupported list schema."
            ) from error
    raise MonteCarloAnalysisError(
        "The Monte Carlo backend returned an unsupported result type."
    )


def normalize_championship_probabilities(values: Mapping[str, Any]) -> dict[str, float]:
    """Normalize a complete decimal-or-percent probability vector to [0, 1]."""
    if not values:
        raise MonteCarloAnalysisError(
            "The Monte Carlo backend returned no probabilities."
        )
    numeric: dict[str, float] = {}
    for team, value in values.items():
        probability = _finite_number(value, field="championship probability")
        if probability < 0:
            raise MonteCarloAnalysisError(
                "Championship probabilities cannot be negative."
            )
        numeric[team] = probability

    maximum = max(numeric.values())
    if maximum > 100 + 1e-8:
        raise MonteCarloAnalysisError(
            "Championship probabilities must be decimals or percentages."
        )
    total = sum(numeric.values())
    if total <= 0:
        raise MonteCarloAnalysisError(
            "Championship probabilities must have a positive total."
        )
    expected_total = 1.0 if maximum <= 1 + 1e-8 else 100.0
    if abs(total - expected_total) > 1e-5:
        raise MonteCarloAnalysisError(
            "Championship probabilities must sum to 1.0 or 100% before normalization."
        )
    normalized = {team: value / total for team, value in numeric.items()}
    if not 0.999999 <= sum(normalized.values()) <= 1.000001:
        raise MonteCarloAnalysisError(
            "Championship probabilities could not be normalized consistently."
        )
    return normalized


def wilson_interval(
    successes: int, total: int, *, z_score: float = WILSON_Z_95
) -> tuple[float, float]:
    """Return a two-sided Wilson proportion interval for Monte Carlo counts."""
    if isinstance(successes, bool) or isinstance(total, bool):
        raise MonteCarloAnalysisError("Wilson interval inputs must be integer counts.")
    if not isinstance(successes, int) or not isinstance(total, int):
        raise MonteCarloAnalysisError("Wilson interval inputs must be integer counts.")
    if total <= 0 or successes < 0 or successes > total:
        raise MonteCarloAnalysisError(
            "Wilson interval counts are outside valid bounds."
        )
    proportion = successes / total
    z_squared = z_score**2
    denominator = 1 + z_squared / total
    center = (proportion + z_squared / (2 * total)) / denominator
    margin = (
        z_score
        * sqrt(proportion * (1 - proportion) / total + z_squared / (4 * total**2))
        / denominator
    )
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    if successes == 0:
        lower = 0.0
    if successes == total:
        upper = 1.0
    return lower, upper


def _normalize_count_mapping(
    raw_counts: Mapping[str, Any], teams: Sequence[str], *, field: str, total: int
) -> dict[str, int]:
    """Validate a team-count mapping and add zeroes only for configured teams."""
    canonical_teams = {_team_key(team): team for team in teams}
    normalized: dict[str, int] = {team: 0 for team in teams}
    seen: set[str] = set()
    for raw_team, raw_count in raw_counts.items():
        team = _clean_team(raw_team)
        if team is None:
            raise MonteCarloAnalysisError(f"{field} contains an invalid team name.")
        key = _team_key(team)
        if key not in canonical_teams:
            raise MonteCarloAnalysisError(
                f"{field} contains a team outside the active tournament field."
            )
        if key in seen:
            raise MonteCarloAnalysisError(f"{field} contains duplicate team names.")
        count = _safe_int(raw_count, field=field)
        if count > total:
            raise MonteCarloAnalysisError(
                f"{field} contains a count larger than the simulation total."
            )
        normalized[canonical_teams[key]] = count
        seen.add(key)
    return normalized


def _extract_legacy_rankings(
    raw_result: Any, teams: Sequence[str], simulation_count: int
) -> tuple[dict[str, int] | None, dict[str, float] | None]:
    """Extract only unambiguous championship count/probability columns."""
    frame = _as_tabular_frame(raw_result)
    if frame.empty:
        raise MonteCarloAnalysisError(
            "The Monte Carlo backend returned an empty result."
        )
    team_column = _find_column(frame, ("team", "country", "champion"))
    count_column = _find_column(frame, ("titles", "wins", "championships", "count"))
    probability_column = _find_column(
        frame, ("probability", "champion probability", "champion_probability")
    )
    if not team_column or (not count_column and not probability_column):
        available = ", ".join(map(str, frame.columns)) or "none"
        raise MonteCarloAnalysisError(
            "The Monte Carlo result schema needs a Team/Country column and a "
            f"Titles/Count or Probability column. Available columns: {available}."
        )

    canonical_teams = {_team_key(team): team for team in teams}
    counts: dict[str, int] = {team: 0 for team in teams}
    probabilities: dict[str, float] = {}
    seen: set[str] = set()
    for _, row in frame.iterrows():
        team = _clean_team(row[team_column])
        if team is None:
            raise MonteCarloAnalysisError(
                "The Monte Carlo result contains an invalid team."
            )
        key = _team_key(team)
        if key not in canonical_teams or key in seen:
            raise MonteCarloAnalysisError(
                "The Monte Carlo result contains duplicate or unsupported teams."
            )
        canonical_team = canonical_teams[key]
        if count_column:
            count = _safe_int(row[count_column], field="championship count")
            if count > simulation_count:
                raise MonteCarloAnalysisError(
                    "A championship count exceeds the requested simulation total."
                )
            counts[canonical_team] = count
        if probability_column:
            probabilities[canonical_team] = _finite_number(
                row[probability_column], field="championship probability"
            )
        seen.add(key)

    normalized_probabilities = (
        normalize_championship_probabilities(
            {team: probabilities.get(team, 0.0) for team in teams}
        )
        if probability_column
        else None
    )
    return (counts if count_column else None), normalized_probabilities


def _normalize_stage_counts(
    raw_stage_counts: Any,
    teams: Sequence[str],
    simulation_count: int,
    configured_rounds: Sequence[str],
    champion_counts: Mapping[str, int],
) -> dict[str, dict[str, int]] | None:
    """Validate real stage appearance counts and their tournament monotonicity."""
    if raw_stage_counts is None:
        return None
    if not isinstance(raw_stage_counts, Mapping):
        raise MonteCarloAnalysisError("Stage counts have an unsupported schema.")

    expected = [str(round_name) for round_name in configured_rounds]
    if set(raw_stage_counts) != set(expected):
        raise MonteCarloAnalysisError(
            "Stage counts do not match the configured tournament rounds."
        )
    normalized: dict[str, dict[str, int]] = {}
    for stage_index, stage in enumerate(expected):
        raw_counts = raw_stage_counts.get(stage)
        if not isinstance(raw_counts, Mapping):
            raise MonteCarloAnalysisError(f"Stage counts are missing {stage} data.")
        normalized[stage] = _normalize_count_mapping(
            raw_counts, teams, field=f"{stage} appearances", total=simulation_count
        )
        expected_appearances = len(teams) // (2**stage_index) * simulation_count
        if sum(normalized[stage].values()) != expected_appearances:
            raise MonteCarloAnalysisError(
                "Stage appearance counts do not match the configured tournament structure."
            )

    stage_sequence = [*expected, "Champion"]
    stage_maps: dict[str, Mapping[str, int]] = {
        **normalized,
        "Champion": champion_counts,
    }
    for team in teams:
        previous = simulation_count
        for stage in stage_sequence:
            current = int(stage_maps[stage][team])
            if current > previous:
                raise MonteCarloAnalysisError(
                    "Stage appearance counts violate tournament progression monotonicity."
                )
            previous = current
    return normalized


def _normalize_convergence(
    raw_convergence: Any, teams: Sequence[str], simulation_count: int
) -> list[dict[str, Any]] | None:
    """Validate genuine cumulative champion checkpoints from one run path."""
    if raw_convergence is None:
        return None
    if not isinstance(raw_convergence, list) or not raw_convergence:
        raise MonteCarloAnalysisError("Convergence checkpoints have an invalid schema.")
    normalized: list[dict[str, Any]] = []
    previous_completed = 0
    for checkpoint in raw_convergence:
        if not isinstance(checkpoint, Mapping):
            raise MonteCarloAnalysisError(
                "Convergence checkpoints contain invalid records."
            )
        completed = _safe_int(
            checkpoint.get("completed_simulations"), field="completed simulations"
        )
        if completed <= previous_completed or completed > simulation_count:
            raise MonteCarloAnalysisError(
                "Convergence checkpoints must be strictly increasing within the run total."
            )
        raw_counts = checkpoint.get("champion_counts")
        if not isinstance(raw_counts, Mapping):
            raise MonteCarloAnalysisError(
                "Convergence checkpoints are missing champion counts."
            )
        counts = _normalize_count_mapping(
            raw_counts, teams, field="checkpoint championship count", total=completed
        )
        if sum(counts.values()) != completed:
            raise MonteCarloAnalysisError(
                "Checkpoint championship counts do not equal completed simulations."
            )
        normalized.append(
            {
                "completed_simulations": completed,
                "probabilities": {
                    team: count / completed for team, count in counts.items()
                },
            }
        )
        previous_completed = completed
    return normalized


def normalize_monte_carlo_output(
    raw_result: Any,
    *,
    simulation_count: int,
    teams: Sequence[str],
    tournament_format: str,
    configured_rounds: Sequence[str],
    simulation_id: str,
    simulated_at: str,
    seed: int | None,
    model_name: str | None,
) -> dict[str, Any]:
    """Normalize real detailed or legacy Monte Carlo output into UI-safe data."""
    if simulation_count <= 0:
        raise MonteCarloAnalysisError("Simulation count must be positive.")
    clean_teams = [_clean_team(team) for team in teams]
    if any(team is None for team in clean_teams):
        raise MonteCarloAnalysisError(
            "The active tournament field contains invalid teams."
        )
    canonical_teams = [str(team) for team in clean_teams]
    if len({_team_key(team) for team in canonical_teams}) != len(canonical_teams):
        raise MonteCarloAnalysisError(
            "The active tournament field contains duplicate teams."
        )

    detailed = (
        raw_result
        if isinstance(raw_result, Mapping) and "champion_counts" in raw_result
        else None
    )
    if detailed is not None:
        raw_simulation_count = detailed.get("simulation_count")
        if (
            raw_simulation_count is not None
            and _safe_int(raw_simulation_count, field="detailed simulation count")
            != simulation_count
        ):
            raise MonteCarloAnalysisError(
                "The detailed Monte Carlo result count does not match the requested run count."
            )
        raw_teams = detailed.get("teams")
        if (
            raw_teams is not None
            and [str(team) for team in raw_teams] != canonical_teams
        ):
            raise MonteCarloAnalysisError(
                "The detailed Monte Carlo result field does not match the active tournament teams."
            )
        raw_counts = detailed.get("champion_counts")
        if not isinstance(raw_counts, Mapping):
            raise MonteCarloAnalysisError(
                "The detailed Monte Carlo result has no champion counts."
            )
        champion_counts = _normalize_count_mapping(
            raw_counts,
            canonical_teams,
            field="championship count",
            total=simulation_count,
        )
        raw_probability_values = detailed.get("champion_probabilities")
        provided_probabilities = (
            normalize_championship_probabilities(raw_probability_values)
            if isinstance(raw_probability_values, Mapping)
            else None
        )
        stage_counts = _normalize_stage_counts(
            detailed.get("stage_counts"),
            canonical_teams,
            simulation_count,
            configured_rounds,
            champion_counts,
        )
        convergence = _normalize_convergence(
            detailed.get("convergence"), canonical_teams, simulation_count
        )
        raw_format = detailed.get("format")
        if raw_format and str(raw_format) != tournament_format:
            raise MonteCarloAnalysisError(
                "The Monte Carlo result format does not match the active tournament configuration."
            )
    else:
        champion_counts, provided_probabilities = _extract_legacy_rankings(
            raw_result, canonical_teams, simulation_count
        )
        stage_counts = None
        convergence = None

    if champion_counts is not None:
        if sum(champion_counts.values()) != simulation_count:
            raise MonteCarloAnalysisError(
                "Championship counts do not equal the requested simulation total."
            )
        probabilities = {
            team: champion_counts[team] / simulation_count for team in canonical_teams
        }
        if provided_probabilities is not None:
            for team in canonical_teams:
                if abs(probabilities[team] - provided_probabilities[team]) > 1e-6:
                    raise MonteCarloAnalysisError(
                        "Provided championship probabilities do not match championship counts."
                    )
    elif provided_probabilities is not None:
        probabilities = {
            team: provided_probabilities.get(team, 0.0) for team in canonical_teams
        }
    else:
        raise MonteCarloAnalysisError(
            "The Monte Carlo backend did not expose counts or championship probabilities."
        )

    if not 0.999999 <= sum(probabilities.values()) <= 1.000001:
        raise MonteCarloAnalysisError("Championship probabilities do not sum to 100%.")

    rankings: list[dict[str, Any]] = []
    for team in canonical_teams:
        titles = champion_counts.get(team) if champion_counts is not None else None
        interval_low, interval_high = (
            wilson_interval(titles, simulation_count)
            if titles is not None
            else (None, None)
        )
        record: dict[str, Any] = {
            "team": team,
            "championships": titles,
            "champion_probability": probabilities[team],
            "confidence_interval_low": interval_low,
            "confidence_interval_high": interval_high,
        }
        if stage_counts is not None:
            stage_key_map = {
                "quarterfinal_appearances": configured_rounds[0],
                "semifinal_appearances": configured_rounds[1],
                "final_appearances": configured_rounds[2],
            }
            for output_key, stage_name in stage_key_map.items():
                count = stage_counts[stage_name][team]
                probability_key = output_key.replace("appearances", "probability")
                record[output_key] = count
                record[probability_key] = count / simulation_count
        rankings.append(record)

    rankings.sort(
        key=lambda record: (
            -float(record["champion_probability"]),
            str(record["team"]).casefold(),
        )
    )
    for rank, record in enumerate(rankings, start=1):
        record["rank"] = rank
        if record["confidence_interval_low"] is not None and (
            record["confidence_interval_low"] > record["champion_probability"]
            or record["confidence_interval_high"] < record["champion_probability"]
        ):
            raise MonteCarloAnalysisError(
                "A Monte Carlo sampling interval does not contain its point estimate."
            )

    stage_probabilities: list[dict[str, Any]] | None = None
    if stage_counts is not None:
        stage_probabilities = []
        for record in rankings:
            stage_probabilities.append(
                {
                    "team": record["team"],
                    "Quarter-finals": record.get("quarterfinal_probability"),
                    "Semi-finals": record.get("semifinal_probability"),
                    "Final": record.get("final_probability"),
                    "Champion": record["champion_probability"],
                }
            )

    leading = rankings[0]
    concentration = {
        "top_team_probability": leading["champion_probability"],
        "top_three_probability": sum(
            float(record["champion_probability"]) for record in rankings[:3]
        ),
        "teams_above_five_percent": sum(
            float(record["champion_probability"]) > 0.05 for record in rankings
        ),
    }
    balance_summary = _competitive_balance_summary(rankings, concentration)
    return {
        "simulation_id": simulation_id,
        "simulated_at": simulated_at,
        "simulation_count": simulation_count,
        "seed": seed,
        "tournament_format": tournament_format,
        "model_name": model_name,
        "teams": canonical_teams,
        "rankings": rankings,
        "stage_probabilities": stage_probabilities,
        "convergence": convergence,
        "raw_run_count": simulation_count,
        "concentration": concentration,
        "competitive_balance": balance_summary,
        "metadata": {
            "result_type": "detailed" if detailed is not None else "legacy",
            "has_counts": champion_counts is not None,
            "has_stage_probabilities": stage_probabilities is not None,
            "has_convergence": convergence is not None,
            "interval_method": "Wilson 95% binomial proportion interval"
            if champion_counts is not None
            else None,
        },
    }


def _competitive_balance_summary(
    rankings: Sequence[Mapping[str, Any]], concentration: Mapping[str, Any]
) -> str:
    """Generate a deterministic interpretation of the observed frequency distribution."""
    first = float(rankings[0]["champion_probability"])
    second = float(rankings[1]["champion_probability"]) if len(rankings) > 1 else 0.0
    gap = first - second
    top_three = float(concentration["top_three_probability"])
    if top_three >= 0.7:
        return "The simulated championship frequency is concentrated around a small group of leading teams."
    if gap <= 0.015:
        return (
            "The top two simulated championship frequencies are closely separated "
            f"by {gap:.1%}."
        )
    return "The simulated championship frequency is distributed across multiple teams in the field."


def _checkpoint_schedule(simulation_count: int) -> tuple[int, ...]:
    """Return real cumulative checkpoints suitable for the selected run size."""
    candidates = (50, 100, 250, 500, 1000)
    points = [point for point in candidates if point <= simulation_count]
    if simulation_count not in points:
        points.append(simulation_count)
    return tuple(sorted(set(points)))


def get_monte_carlo_overview() -> dict[str, Any]:
    """Load static tournament metadata without running any simulations or loading a model."""
    try:
        overview = get_tournament_overview()
    except TournamentSimulationError as error:
        raise MonteCarloAnalysisError(str(error)) from error
    return {
        "configuration": overview["configuration"],
        "teams": overview["teams"],
        "latest_snapshot_date": overview.get("latest_snapshot_date"),
        "warnings": overview.get("warnings", []),
        "supported_simulation_counts": SUPPORTED_SIMULATION_COUNTS,
        "model_files_available": BEST_MODEL_PATH.exists()
        and PREPROCESSOR_PATH.exists(),
        "seed_supported": True,
    }


def monte_carlo_signature(simulation_count: int, seed: int | None) -> str:
    """Create a versioned signature to keep stale analyses from active settings."""
    count = validate_simulation_count(simulation_count)
    overview = get_monte_carlo_overview()
    team_part = "|".join(_team_key(team) for team in overview["teams"])
    configuration_part = json.dumps(
        overview["configuration"], sort_keys=True, default=str, separators=(",", ":")
    )
    seed_part = "random" if seed is None else str(seed)
    return "::".join(
        (
            configuration_part,
            team_part,
            str(_path_version(SNAPSHOT_DATA_PATH)),
            str(_path_version(BEST_MODEL_PATH)),
            str(_path_version(PREPROCESSOR_PATH)),
            str(count),
            seed_part,
        )
    )


def is_monte_carlo_result_current(
    result: Mapping[str, Any] | None, signature: str
) -> bool:
    """Check whether a saved analysis belongs to the active controls and sources."""
    return bool(result and result.get("signature") == signature)


def validate_monte_carlo_preflight(simulation_count: int) -> dict[str, Any]:
    """Validate bounded count, model files, field, bracket, and team snapshots."""
    count = validate_simulation_count(simulation_count)
    try:
        preflight = validate_tournament_preflight()
    except TournamentSimulationError as error:
        raise MonteCarloAnalysisError(str(error)) from error
    return {"simulation_count": count, **preflight}


def run_monte_carlo_analysis(
    simulation_count: int,
    seed: int | None = None,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the existing detailed MonteCarloSimulator without caching the result."""
    count = validate_simulation_count(simulation_count)
    parsed_seed = parse_monte_carlo_seed(seed)
    # Confirm model files, snapshots, and the fixed bracket before expensive iterations start.
    preflight = validate_monte_carlo_preflight(count)
    configuration = preflight["configuration"]
    teams = preflight["teams"]

    try:
        from src.simulator.monte_carlo import MonteCarloSimulator

        predictor = get_predictor()
        # Reuse the cached predictor while the simulator executes repeated real brackets.
        simulator = MonteCarloSimulator(
            simulations=count, predictor=predictor, seed=parsed_seed
        )
        raw_result = simulator.run_detailed(
            checkpoints=_checkpoint_schedule(count), progress_callback=progress_callback
        )
    except MatchPredictionError as error:
        raise MonteCarloAnalysisError(str(error)) from error
    except (
        AttributeError,
        ImportError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        raise MonteCarloAnalysisError(
            "The Monte Carlo simulator could not complete the configured tournament runs."
        ) from error

    metadata = get_model_metadata()
    model_name = metadata.get("model_name") or type(predictor.model).__name__
    return normalize_monte_carlo_output(
        raw_result,
        simulation_count=count,
        teams=teams,
        tournament_format=str(configuration["format"]),
        configured_rounds=list(configuration["rounds"]),
        simulation_id=uuid4().hex,
        simulated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        seed=parsed_seed,
        model_name=model_name,
    )


def monte_carlo_result_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Create an exportable DataFrame from one normalized real analysis result."""
    rows: list[dict[str, Any]] = []
    for ranking in result.get("rankings", []):
        rows.append(
            {
                "simulation_id": result.get("simulation_id"),
                "simulated_at": result.get("simulated_at"),
                "simulation_count": result.get("simulation_count"),
                "seed": result.get("seed"),
                "tournament_format": result.get("tournament_format"),
                "rank": ranking.get("rank"),
                "team": ranking.get("team"),
                "championships": ranking.get("championships"),
                "champion_probability": ranking.get("champion_probability"),
                "sampling_interval_low": ranking.get("confidence_interval_low"),
                "sampling_interval_high": ranking.get("confidence_interval_high"),
                "quarterfinal_appearances": ranking.get("quarterfinal_appearances"),
                "quarterfinal_probability": ranking.get("quarterfinal_probability"),
                "semifinal_appearances": ranking.get("semifinal_appearances"),
                "semifinal_probability": ranking.get("semifinal_probability"),
                "final_appearances": ranking.get("final_appearances"),
                "final_probability": ranking.get("final_probability"),
            }
        )
    return pd.DataFrame(rows)


def monte_carlo_result_csv(result: Mapping[str, Any]) -> bytes:
    """Return UTF-8 CSV bytes for the current normalized analysis only."""
    return monte_carlo_result_frame(result).to_csv(index=False).encode("utf-8")
