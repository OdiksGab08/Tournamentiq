"""Focused contracts for the Monte Carlo analysis service and backend adapter."""

from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO

import pandas as pd
import pytest

from dashboard.services.monte_carlo_service import (
    MonteCarloAnalysisError,
    is_monte_carlo_result_current,
    monte_carlo_signature,
    monte_carlo_result_csv,
    normalize_championship_probabilities,
    normalize_monte_carlo_output,
    wilson_interval,
)
from src.simulator.monte_carlo import MonteCarloSimulator
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


def _detailed_raw(simulation_count: int = 100) -> dict[str, object]:
    configuration = _configuration()
    teams = list(configuration["teams"])
    champion_counts = dict(zip(teams, (50, 10, 20, 5, 5, 5, 3, 2), strict=True))
    semi_counts = dict(zip(teams, (60, 40, 60, 40, 50, 50, 50, 50), strict=True))
    final_counts = dict(zip(teams, (60, 20, 40, 20, 20, 15, 15, 10), strict=True))
    return {
        "simulation_count": simulation_count,
        "teams": teams,
        "format": configuration["format"],
        "champion_counts": champion_counts,
        "stage_counts": {
            "Quarter-finals": {team: simulation_count for team in teams},
            "Semi-finals": semi_counts,
            "Final": final_counts,
        },
        "convergence": [
            {
                "completed_simulations": 50,
                "champion_counts": dict(
                    zip(teams, (25, 5, 10, 3, 3, 2, 1, 1), strict=True)
                ),
            },
            {"completed_simulations": 100, "champion_counts": champion_counts},
        ],
    }


def _normalized(raw: dict[str, object] | None = None) -> dict[str, object]:
    configuration = _configuration()
    return normalize_monte_carlo_output(
        raw or _detailed_raw(),
        simulation_count=100,
        teams=list(configuration["teams"]),
        tournament_format=str(configuration["format"]),
        configured_rounds=list(configuration["rounds"]),
        simulation_id="test-analysis",
        simulated_at="2026-07-12T00:00:00+00:00",
        seed=2026,
        model_name="Stub model",
    )


def test_probability_normalization_accepts_decimal_and_percentage_vectors():
    """Verify championship probabilities accept equivalent decimal and percent input."""
    decimal = normalize_championship_probabilities({"France": 0.6, "Spain": 0.4})
    percentage = normalize_championship_probabilities({"France": 60, "Spain": 40})

    assert decimal == pytest.approx({"France": 0.6, "Spain": 0.4})
    assert percentage == pytest.approx(decimal)
    with pytest.raises(MonteCarloAnalysisError):
        normalize_championship_probabilities({"France": 60, "Spain": 30})


def test_legacy_count_output_adds_configured_zero_title_teams_and_ranks_descending():
    """Verify legacy counts include configured teams and retain descending ranks."""
    configuration = _configuration()
    legacy = pd.DataFrame(
        {"Team": ["France", "Spain"], "Titles": [60, 40], "Probability": [60, 40]}
    )
    result = normalize_monte_carlo_output(
        legacy,
        simulation_count=100,
        teams=list(configuration["teams"]),
        tournament_format=str(configuration["format"]),
        configured_rounds=list(configuration["rounds"]),
        simulation_id="legacy",
        simulated_at="2026-07-12T00:00:00+00:00",
        seed=None,
        model_name="Stub model",
    )

    assert len(result["rankings"]) == 8
    assert result["rankings"][0]["team"] == "France"
    assert result["rankings"][-1]["championships"] == 0
    assert sum(record["championships"] for record in result["rankings"]) == 100


def test_invalid_schema_and_count_total_are_rejected():
    """Verify malformed result schemas and inconsistent title totals are rejected."""
    configuration = _configuration()
    with pytest.raises(MonteCarloAnalysisError):
        normalize_monte_carlo_output(
            pd.DataFrame({"Country": ["France"], "Metric": [1]}),
            simulation_count=100,
            teams=list(configuration["teams"]),
            tournament_format=str(configuration["format"]),
            configured_rounds=list(configuration["rounds"]),
            simulation_id="bad-schema",
            simulated_at="2026-07-12T00:00:00+00:00",
            seed=None,
            model_name=None,
        )

    malformed = _detailed_raw()
    malformed["champion_counts"]["France"] = 49
    with pytest.raises(MonteCarloAnalysisError):
        _normalized(malformed)


def test_wilson_interval_handles_zero_and_all_wins():
    """Verify Wilson intervals remain valid at both probability extremes."""
    zero_low, zero_high = wilson_interval(0, 100)
    all_low, all_high = wilson_interval(100, 100)

    assert zero_low == pytest.approx(0.0)
    assert 0 < zero_high < 0.1
    assert 0.9 < all_low < 1
    assert all_high == pytest.approx(1.0)


def test_stage_monotonicity_ranking_signature_and_csv_export():
    """Verify normalized stages, result signatures, and CSV export stay coherent."""
    result = _normalized()
    assert [record["rank"] for record in result["rankings"]] == list(range(1, 9))
    assert sum(
        record["champion_probability"] for record in result["rankings"]
    ) == pytest.approx(1.0)
    assert result["stage_probabilities"] is not None
    assert result["convergence"] is not None
    result["signature"] = "active"
    assert is_monte_carlo_result_current(result, "active")
    assert not is_monte_carlo_result_current(result, "stale")

    csv_text = monte_carlo_result_csv(result).decode("utf-8")
    assert "test-analysis" in csv_text
    assert "champion_probability" in csv_text

    malformed = deepcopy(_detailed_raw())
    malformed["stage_counts"]["Final"]["France"] = 80
    with pytest.raises(MonteCarloAnalysisError):
        _normalized(malformed)


def test_signature_changes_when_the_tournament_configuration_changes(monkeypatch):
    """Verify changing tournament rules invalidates the stored analysis signature."""
    configuration = _configuration()
    overview = {
        "configuration": dict(configuration),
        "teams": list(configuration["teams"]),
    }
    monkeypatch.setattr(
        "dashboard.services.monte_carlo_service.get_monte_carlo_overview",
        lambda: overview,
    )

    original = monte_carlo_signature(100, 2026)
    overview["configuration"]["draw_resolution"] = "different_backend_rule"

    assert monte_carlo_signature(100, 2026) != original


def test_real_monte_carlo_detailed_aggregation_has_real_stage_and_checkpoints():
    """Verify detailed simulation aggregates real stages and requested checkpoints."""
    engine = TournamentEngine(
        predictor=_PredictorStub(), simulator=_HomeTeamSimulator(), seed=7
    )
    result = MonteCarloSimulator(simulations=4, engine=engine, seed=7).run_detailed(
        checkpoints=(2, 4)
    )

    assert sum(result["champion_counts"].values()) == 4
    assert sum(result["stage_counts"]["Quarter-finals"].values()) == 32
    assert sum(result["stage_counts"]["Semi-finals"].values()) == 16
    assert sum(result["stage_counts"]["Final"].values()) == 8
    assert [item["completed_simulations"] for item in result["convergence"]] == [2, 4]


def test_detailed_aggregation_rejects_inconsistent_advancing_fixtures():
    """Verify malformed advancement fixtures fail detailed aggregation validation."""
    engine = TournamentEngine(
        predictor=_PredictorStub(), simulator=_HomeTeamSimulator(), seed=7
    )
    malformed = engine.simulate_detailed()
    malformed["rounds"][1]["matches"][0]["home_team"] = "Morocco"
    engine.simulate_detailed = lambda: malformed

    with pytest.raises(ValueError, match="invalid Semi-finals fixture order"):
        MonteCarloSimulator(simulations=1, engine=engine, seed=7).run_detailed()


def test_legacy_dataframe_run_contract_is_preserved_for_existing_home_consumers():
    """Verify the legacy dataframe output remains available to existing callers."""
    engine = TournamentEngine(
        predictor=_PredictorStub(), simulator=_HomeTeamSimulator(), seed=7
    )
    simulator = MonteCarloSimulator(simulations=3, engine=engine, seed=7)
    with redirect_stdout(StringIO()):
        frame = simulator.run()

    assert list(frame.columns) == ["Team", "Titles", "Probability"]
    assert int(frame["Titles"].sum()) == 3
