"""Focused contracts for the historical statistics data service."""

from datetime import date

import pandas as pd
import pytest

from dashboard.services.statistics_service import (
    StatisticsDataError,
    apply_statistics_filters,
    calculate_kpis,
    competition_summary,
    dataframe_csv,
    export_match_rows,
    outcome_distribution,
    prepare_match_table,
    scoreline_explorer,
    statistics_filter_signature,
    team_performance_table,
    world_cup_historical_summary,
)


def _prepared_matches() -> tuple[pd.DataFrame, dict[str, object]]:
    source = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-02-01", "2025-01-01", "2025-02-01"],
            "home_team": ["France", "Spain", "Brazil", "Germany"],
            "away_team": ["Spain", "France", "France", "Italy"],
            "home_score": [2, 1, 0, -1],
            "away_score": [1, 1, 3, 1],
            "tournament": [
                "FIFA World Cup",
                "FIFA World Cup qualification",
                "Friendly",
                "FIFA World Cup 2026",
            ],
            "neutral": ["yes", "no", True, "unknown"],
        }
    )
    return prepare_match_table(source)


def test_prepare_derives_real_score_results_and_retains_invalid_score_rows():
    """Verify preparation derives score fields while retaining invalid source rows."""
    matches, quality = _prepared_matches()

    assert len(matches) == 4
    assert matches.loc[matches["score_valid"], "result"].tolist() == [
        "Home Win",
        "Draw",
        "Away Win",
    ]
    assert matches.loc[0, "scoreline"] == "2–1"
    assert matches.loc[2, "total_goals"] == 3
    assert matches.loc[2, "goal_difference"] == -3

    invalid = matches.loc[matches["home_team"].eq("Germany")].iloc[0]
    assert not invalid["score_valid"]
    assert pd.isna(invalid["result"])
    assert pd.isna(invalid["scoreline"])
    assert quality["invalid_score_rows"] == 1
    assert quality["valid_score_rows"] == 3

    kpis = calculate_kpis(matches)
    assert kpis["matches"] == 4
    assert kpis["scored_matches"] == 3
    assert outcome_distribution(matches)["matches"].tolist() == [1, 1, 1]


def test_filters_apply_date_competition_and_neutral_venue_without_fabrication():
    """Verify statistics filters combine real date, competition, and venue criteria."""
    matches, _ = _prepared_matches()

    filtered = apply_statistics_filters(
        matches,
        {
            "date_start": "2024-01-15",
            "date_end": "2025-01-31",
            "competitions": ["Friendly", "FIFA World Cup qualification"],
            "neutral": "Non-neutral venues",
        },
    )
    assert filtered[["home_team", "away_team"]].values.tolist() == [["Spain", "France"]]

    neutral_only = apply_statistics_filters(matches, {"neutral": "Neutral venues"})
    assert set(neutral_only["home_team"]) == {"France", "Brazil"}


def test_team_aggregation_is_orientation_safe_and_awards_points_correctly():
    """Verify team aggregation handles home and away records with correct points."""
    matches, _ = _prepared_matches()
    table = team_performance_table(matches).set_index("team")
    france = table.loc["France"]

    assert france["matches_played"] == 3
    assert france[["wins", "draws", "losses"]].tolist() == [2, 1, 0]
    assert france[["goals_scored", "goals_conceded", "goal_difference"]].tolist() == [
        6.0,
        2.0,
        4.0,
    ]
    assert france["points"] == 7
    assert france["points_per_match"] == pytest.approx(7 / 3)
    assert table["goals_scored"].sum() == table["goals_conceded"].sum()


def test_empty_filtered_results_have_safe_empty_statistics():
    """Verify empty filter results produce safe zero or empty statistics."""
    matches, _ = _prepared_matches()
    empty = apply_statistics_filters(matches, {"competitions": ["Not in source"]})

    assert empty.empty
    assert calculate_kpis(empty)["matches"] == 0
    assert calculate_kpis(empty)["total_goals"] is None
    assert outcome_distribution(empty)["matches"].tolist() == [0, 0, 0]
    assert outcome_distribution(empty)["percentage"].isna().all()
    assert team_performance_table(empty).empty


def test_filter_signature_is_deterministic_for_equivalent_filter_values():
    """Verify equivalent filter representations yield a stable data signature."""
    first = {
        "date_start": date(2024, 1, 1),
        "competitions": ["Friendly", "FIFA World Cup"],
        "outcomes": ["Draw", "Home Win"],
        "neutral": "All venues",
    }
    equivalent = {
        "date_start": "2024-01-01",
        "competitions": ["FIFA World Cup", "Friendly"],
        "outcomes": ["Home Win", "Draw"],
        "neutral": "All venues",
    }

    assert statistics_filter_signature(first) == statistics_filter_signature(equivalent)
    assert statistics_filter_signature(first) != statistics_filter_signature(
        {**equivalent, "neutral": "Neutral venues"}
    )


def test_world_cup_filter_uses_the_exact_canonical_competition_value():
    """Verify World Cup filtering uses only the canonical finals competition label."""
    matches, _ = _prepared_matches()

    assert matches["world_cup_finals"].tolist() == [True, False, False, False]
    filtered = apply_statistics_filters(matches, {"world_cup_only": True})
    assert filtered["competition"].tolist() == ["FIFA World Cup"]

    summary = world_cup_historical_summary(matches)
    assert summary is not None
    assert summary["matches"] == 1
    assert summary["kpis"]["scored_matches"] == 1


def test_collision_and_duplicate_quality_metadata_are_transparent():
    """Verify duplicate and fixture-collision metadata remains visible to consumers."""
    source = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "home_team": ["France", "France", "France"],
            "away_team": ["Spain", "Spain", "Spain"],
            "home_score": [1, 2, 1],
            "away_score": [0, 0, 0],
            "tournament": ["Friendly", "Friendly", "Friendly"],
            "neutral": [False, False, False],
        }
    )
    matches, quality = prepare_match_table(source)

    assert len(matches) == 2
    assert quality["duplicate_rows_detected"] == 1
    assert quality["oriented_fixture_collision_rows"] == 3
    assert quality["oriented_fixture_collision_groups"] == 1
    assert quality["excluded_rows"] == 1
    assert not quality["rows_represent_one_match"]


def test_export_csv_contains_the_current_real_match_rows_and_labels():
    """Verify match CSV export contains current source-backed rows and labels."""
    matches, _ = _prepared_matches()

    csv_text = dataframe_csv(export_match_rows(matches)).decode("utf-8")

    assert "fifa_world_cup_finals" in csv_text
    assert "France" in csv_text
    assert "score_valid" in csv_text


def test_world_cup_filter_parses_only_explicit_boolean_values():
    """Verify the World Cup filter accepts only explicit boolean-like values."""
    matches, _ = _prepared_matches()

    assert len(apply_statistics_filters(matches, {"world_cup_only": "false"})) == 4
    assert len(apply_statistics_filters(matches, {"world_cup_only": "true"})) == 1
    with pytest.raises(StatisticsDataError):
        apply_statistics_filters(matches, {"world_cup_only": "sometimes"})


def test_scoreline_explorer_does_not_call_draws_winning_margins():
    """Verify scoreline analysis excludes draws from winning-margin results."""
    source = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "home_team": ["France", "Spain"],
            "away_team": ["Spain", "France"],
            "home_score": [0, 2],
            "away_score": [0, 2],
        }
    )
    matches, _ = prepare_match_table(source)

    assert scoreline_explorer(matches)["biggest_margins"].empty


def test_competition_summary_retains_matches_with_invalid_scores():
    """Verify competition summaries retain matches whose scores are unavailable."""
    source = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "home_team": ["France"],
            "away_team": ["Spain"],
            "home_score": ["unknown"],
            "away_score": ["unknown"],
            "tournament": ["Friendly"],
        }
    )
    matches, _ = prepare_match_table(source)
    summary = competition_summary(matches).iloc[0]

    assert summary["matches"] == 1
    assert summary["scored_matches"] == 0
    assert pd.isna(summary["goals_per_match"])
