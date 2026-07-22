"""Focused contract tests for the dashboard match-prediction adapter."""

import pytest

from dashboard.services.match_prediction_service import (
    MatchPredictionError,
    is_result_current,
    matchup_signature,
    normalize_class_probabilities,
    validate_matchup,
)


def test_normalize_class_probabilities_preserves_verified_mapping():
    """Verify verified class labels map to the expected probability fields."""
    result = normalize_class_probabilities([0, 1, 2], [0.2, 0.3, 0.5])

    assert result == pytest.approx({"home_win": 0.2, "draw": 0.3, "away_win": 0.5})
    assert sum(result.values()) == pytest.approx(1.0)


def test_normalize_class_probabilities_rejects_unknown_or_missing_classes():
    """Verify incomplete or unsupported prediction classes are rejected."""
    with pytest.raises(MatchPredictionError):
        normalize_class_probabilities([0, 1, 3], [0.2, 0.3, 0.5])

    with pytest.raises(MatchPredictionError):
        normalize_class_probabilities([0, 2], [0.4, 0.6])


def test_validate_matchup_rejects_duplicate_or_blank_teams():
    """Verify matchup validation rejects ambiguous or empty team selections."""
    with pytest.raises(MatchPredictionError):
        validate_matchup("France", "france")

    with pytest.raises(MatchPredictionError):
        validate_matchup("", "Argentina")


def test_result_signature_prevents_stale_matchup_display():
    """Verify result signatures prevent a reversed matchup from using stale data."""
    france_argentina = matchup_signature("France", "Argentina")
    argentina_france = matchup_signature("Argentina", "France")
    result = {"signature": france_argentina}

    assert is_result_current(result, france_argentina)
    assert not is_result_current(result, argentina_france)
