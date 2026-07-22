"""Compatibility contracts for retained backend service adapters.

These adapters are not used by the Streamlit dashboard, but they remain part
of the backend surface for existing integrations. The tests ensure their
delegation stays aligned with the active simulator interfaces without loading
the persisted production model or datasets.
"""

from src.services.comparison_services import ComparisonService
from src.services.prediction_service import PredictionService


class _PredictorStub:
    """Record predictor calls while returning a valid three-outcome payload."""

    def __init__(self) -> None:
        """Initialize an empty call log for the adapter contract test."""
        self.calls: list[tuple[str, str]] = []

    def predict(self, home_team: str, away_team: str) -> dict[str, float]:
        """Return a deterministic payload compatible with ``PredictionService``."""
        self.calls.append((home_team, away_team))
        return {
            "home_probability": 0.62,
            "draw_probability": 0.24,
            "away_probability": 0.38,
        }


class _SnapshotStub:
    """Return traceable team snapshots without reading persisted feature data."""

    def get_snapshot(self, team: str) -> dict[str, str]:
        """Return a minimal snapshot with the requested canonical team name."""
        return {"team": team}


def test_prediction_service_retains_legacy_arguments_without_breaking_predictor_contract():
    """Ensure legacy context parameters are not forwarded to the current predictor."""
    service = PredictionService.__new__(PredictionService)
    predictor = _PredictorStub()
    service.predictor = predictor

    result = service.predict_match(
        "France", "Argentina", tournament="Friendly", neutral=False
    )

    assert predictor.calls == [("France", "Argentina")]
    assert result["confidence"] == 0.62
    assert result["confidence_level"] == "Medium"


def test_comparison_service_uses_the_current_live_snapshot_provider():
    """Ensure the retained adapter delegates to ``LiveSnapshot``-compatible data."""
    service = ComparisonService.__new__(ComparisonService)
    service.snapshot = _SnapshotStub()

    assert service.compare("France", "Argentina") == {
        "team1": {"team": "France"},
        "team2": {"team": "Argentina"},
    }
