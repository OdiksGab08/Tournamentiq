"""Focused contracts for the real tournament simulation dashboard adapter."""

from copy import deepcopy

import pytest

from dashboard.services.tournament_simulation_service import (
    TournamentSimulationError,
    extract_champion_path,
    is_tournament_result_current,
    normalize_probabilities,
    normalize_tournament_result,
    tournament_result_csv,
    validate_tournament_configuration,
)
from src.simulator.tournament_engine import TournamentEngine


class _PredictorStub:
    def predict(self, home: str, away: str) -> dict[str, float | str]:
        return {
            "home": home,
            "away": away,
            "home_probability": 0.6,
            "away_probability": 0.4,
            "home_win_probability": 0.5,
            "draw_probability": 0.2,
            "away_win_probability": 0.3,
        }


class _HomeTeamSimulator:
    def choose(
        self, home: str, away: str, home_probability: float, away_probability: float
    ) -> str:
        return home


def _configuration() -> dict[str, object]:
    return TournamentEngine.tournament_configuration()


def _normalized_result() -> dict[str, object]:
    raw = TournamentEngine(
        predictor=_PredictorStub(), simulator=_HomeTeamSimulator(), seed=2026
    ).simulate_detailed()
    return normalize_tournament_result(
        raw,
        _configuration(),
        simulation_id="test-run",
        simulated_at="2026-07-12T00:00:00+00:00",
        seed=2026,
        model_name="Stub model",
    )


def test_probability_normalization_accepts_decimals_and_percentages():
    """Verify match probabilities accept equivalent decimal and percentage input."""
    decimals = normalize_probabilities({"home": 0.2, "draw": 0.3, "away": 0.5})
    percentages = normalize_probabilities({"home": 20, "draw": 30, "away": 50})

    assert decimals == pytest.approx({"home": 0.2, "draw": 0.3, "away": 0.5})
    assert percentages == pytest.approx(decimals)
    with pytest.raises(TournamentSimulationError):
        normalize_probabilities({"home": -0.1, "away": 1.1})


def test_bracket_configuration_rejects_duplicate_teams():
    """Verify tournament configuration rejects duplicate entrants in the bracket."""
    configuration = _configuration()
    configuration["teams"] = list(configuration["teams"])
    configuration["teams"][1] = configuration["teams"][0]

    with pytest.raises(TournamentSimulationError):
        validate_tournament_configuration(configuration)


def test_bracket_configuration_rejects_unsupported_non_power_of_two_field():
    """Verify bracket validation rejects unsupported non-power-of-two team counts."""
    configuration = _configuration()
    configuration["teams"] = list(configuration["teams"][:6])
    configuration["team_count"] = 6
    configuration["quarter_final_fixtures"] = [
        list(fixture) for fixture in configuration["quarter_final_fixtures"][:3]
    ]
    configuration["rounds"] = ["Quarter-finals", "Semi-finals"]

    with pytest.raises(TournamentSimulationError):
        validate_tournament_configuration(configuration)


def test_detailed_engine_round_order_and_advancing_teams_are_consistent():
    """Verify detailed tournament rounds preserve order and valid advancement."""
    result = _normalized_result()
    rounds = result["rounds"]

    assert [round_data["name"] for round_data in rounds] == [
        "Quarter-finals",
        "Semi-finals",
        "Final",
    ]
    assert [len(round_data["matches"]) for round_data in rounds] == [4, 2, 1]
    assert [match["winner"] for match in rounds[0]["matches"]] == [
        match[team]
        for match in rounds[1]["matches"]
        for team in ("home_team", "away_team")
    ]
    assert result["champion"] == rounds[-1]["matches"][0]["winner"]


def test_legacy_engine_simulate_contract_is_preserved_for_monte_carlo_callers():
    """Verify the legacy champion-returning simulator contract remains intact."""
    engine = TournamentEngine(
        predictor=_PredictorStub(), simulator=_HomeTeamSimulator(), seed=2026
    )

    assert engine.simulate() == "France"


def test_champion_path_is_chronological_and_uses_actual_matches():
    """Verify a champion path uses chronological matches and real opponents."""
    result = _normalized_result()
    path = extract_champion_path(result["rounds"], str(result["champion"]))

    assert [step["round"] for step in path] == [
        "Quarter-finals",
        "Semi-finals",
        "Final",
    ]
    assert [step["opponent"] for step in path] == ["Morocco", "Spain", "Norway"]
    assert all(step["score"] is None for step in path)


def test_result_signature_guard_and_malformed_winner_handling():
    """Verify stale signatures and invalid match winners are rejected safely."""
    result = _normalized_result()
    result["signature"] = "active"
    assert is_tournament_result_current(result, "active")
    assert not is_tournament_result_current(result, "other")

    raw = TournamentEngine(
        predictor=_PredictorStub(), simulator=_HomeTeamSimulator()
    ).simulate_detailed()
    malformed = deepcopy(raw)
    malformed["rounds"][0]["matches"][0]["winner"] = "Not in fixture"
    with pytest.raises(TournamentSimulationError):
        normalize_tournament_result(
            malformed,
            _configuration(),
            simulation_id="bad",
            simulated_at="2026-07-12T00:00:00+00:00",
            seed=None,
            model_name="Stub model",
        )


def test_draw_resolution_metadata_and_csv_export_use_current_result_only():
    """Verify current draw metadata and CSV output reflect only the active result."""
    result = _normalized_result()
    assert (
        result["data_transparency"]["draw_resolution"]
        == "draw_probability_split_evenly_then_sampled"
    )
    assert all(
        match["score"] is None
        for round_data in result["rounds"]
        for match in round_data["matches"]
    )

    csv_text = tournament_result_csv(result).decode("utf-8")
    assert "simulation_id" in csv_text
    assert "test-run" in csv_text
    assert csv_text.count("Quarter-finals") == 4
