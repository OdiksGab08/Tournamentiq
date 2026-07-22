"""Contracts for Home metrics derived from real TournamentIQ artifacts."""

from dashboard.services.platform_statistics_service import load_platform_statistics


def test_platform_statistics_expose_only_verified_project_artifact_values():
    """Ensure every Home coverage and model metric resolves from current artifacts."""
    statistics = load_platform_statistics()

    expected_keys = {
        "total_historical_matches",
        "trained_models",
        "production_model",
        "engineered_features",
        "competitions",
        "countries_represented",
        "test_accuracy",
    }
    assert expected_keys <= set(statistics)

    for key in (
        "total_historical_matches",
        "trained_models",
        "engineered_features",
        "competitions",
        "countries_represented",
    ):
        assert isinstance(statistics[key], int)
        assert statistics[key] > 0

    assert isinstance(statistics["production_model"], str)
    assert statistics["production_model"].strip()
    assert isinstance(statistics["test_accuracy"], float)
    assert 0 <= statistics["test_accuracy"] <= 1

    # Existing Home consumers retain these compatibility values without a
    # second artifact read or a fabricated simulation result.
    assert statistics["international_matches"] == statistics["total_historical_matches"]
    assert statistics["best_model"] == statistics["production_model"]
    assert statistics["latest_simulation_count"] is None
