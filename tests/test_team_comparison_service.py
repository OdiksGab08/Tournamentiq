"""Focused contracts for the Team Comparison data adapter."""

import pytest

from dashboard.services.team_comparison_service import (
    TeamComparisonError,
    build_comparison_metrics,
    comparison_signature,
    is_comparison_current,
    normalize_pair,
    normalize_team_names,
    orient_head_to_head_match,
    validate_comparison_teams,
)


def test_team_name_normalization_preserves_canonical_names_and_filters_invalid_values():
    """Verify team normalization retains canonical names and removes invalid input."""
    teams = normalize_team_names(
        [" France ", None, "france", "", "N/A", "Argentina", "Argentina"]
    )

    assert teams == ["Argentina", "France"]


def test_duplicate_team_validation_is_case_insensitive():
    """Verify comparison validation rejects the same team despite casing changes."""
    with pytest.raises(TeamComparisonError):
        validate_comparison_teams("France", "france")


def test_pair_normalization_handles_equal_and_lower_is_better_features():
    """Verify pair normalization supports ties and lower-is-better metrics."""
    assert normalize_pair(0.8, 0.2, higher_is_better=True) == pytest.approx((1.0, 0.0))
    assert normalize_pair(0.2, 0.8, higher_is_better=False) == pytest.approx((1.0, 0.0))
    assert normalize_pair(3.0, 3.0, higher_is_better=True) == pytest.approx((0.5, 0.5))


def test_comparison_signature_prevents_stale_selection_display():
    """Verify selection signatures prevent stale team comparisons from rendering."""
    signature = comparison_signature("France", "Argentina", "snapshot-1")
    stale_signature = comparison_signature("Argentina", "France", "snapshot-1")

    assert is_comparison_current({"signature": signature}, signature)
    assert not is_comparison_current({"signature": signature}, stale_signature)


def test_head_to_head_orientation_normalizes_away_team_a_record():
    """Verify head-to-head records normalize an away Team A perspective correctly."""
    record = orient_head_to_head_match(
        {
            "date": "2026-06-01",
            "home_team": "Belgium",
            "away_team": "France",
            "home_score": 1,
            "away_score": 2,
            "tournament": "Friendly",
        },
        "France",
        "Belgium",
    )

    assert record is not None
    assert record["team_a_venue"] == "Away"
    assert record["team_a_score"] == 2.0
    assert record["team_b_score"] == 1.0
    assert record["result"] == "W"


def test_malformed_feature_rows_are_marked_unavailable_not_fabricated():
    """Verify invalid feature values are reported unavailable rather than invented."""
    metrics, unavailable = build_comparison_metrics(
        {"form_win_rate": "not-a-number"},
        {"form_win_rate": 0.6},
    )

    assert metrics == []
    assert "Recent Form Win Rate" in unavailable
