"""Render the verified historical-statistics dashboard route.

Purpose:
    Present filterable historical international-match KPIs and summaries from
    the project's validated warehouse dataset.
Responsibility:
    Manage dashboard filters and display returned statistics without changing
    source data or deriving substitute values for unavailable fields.
Inputs:
    Streamlit filter state and validated tables from
    ``services.statistics_service``.
Outputs:
    Historical filter controls, active-filter summaries, and KPI cards.
Collaboration:
    Serves the Statistics view and delegates all data loading, filtering, and
    metric calculations to the statistics service.
"""

from __future__ import annotations

from datetime import date
from html import escape
from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st

from services.statistics_service import (
    OUTCOME_ORDER,
    StatisticsDataError,
    apply_statistics_filters,
    calculate_kpis,
    get_statistics_filter_options,
    load_historical_matches,
    normalize_statistics_filters,
    statistics_filter_signature,
)
from ui import (
    animated_container,
    apply_theme,
    glass_card,
    metric_card,
    page_header,
    render_html,
    section_title,
)

ALL_TEAMS_LABEL = "All teams"


def _format_number(value: Any, *, decimals: int = 0) -> str:
    """Format a genuine numeric value without converting missing data to zero."""
    try:
        if value is None or pd.isna(value):
            return "Unavailable"
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "Unavailable"


def _format_percentage(value: Any, *, decimals: int = 1) -> str:
    """Format a 0-to-1 source rate only when it is present and numeric."""
    try:
        if value is None or pd.isna(value):
            return "Unavailable"
        return f"{float(value):.{decimals}%}"
    except (TypeError, ValueError):
        return "Unavailable"


def _format_date(value: Any) -> str:
    """Format an available source date for historical filter summaries."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return "Unavailable"
    return "Unavailable" if pd.isna(timestamp) else timestamp.strftime("%d %b %Y")


def _short_list(values: Sequence[Any], *, limit: int = 3) -> str:
    """Show active multi-select values without producing an oversized summary."""
    cleaned = [str(value) for value in values if str(value).strip()]
    if not cleaned:
        return "All"
    suffix = f" +{len(cleaned) - limit} more" if len(cleaned) > limit else ""
    return ", ".join(cleaned[:limit]) + suffix


def _default_filters(options: Mapping[str, Any]) -> dict[str, Any]:
    """Create the existing all-data filter shape from source bounds."""
    return {
        "date_start": options["date_min"],
        "date_end": options["date_max"],
        "competitions": (),
        "team": None,
        "outcomes": (),
        "neutral": "All venues",
        "minimum_total_goals": 0.0,
        "world_cup_only": False,
    }


def _normalize_date_range(value: Any, options: Mapping[str, Any]) -> tuple[date, date]:
    """Clamp a Streamlit date input to the selected source range."""
    lower, upper = options["date_min"], options["date_max"]

    def clamp(candidate: date) -> date:
        timestamp = pd.Timestamp(candidate)
        if pd.isna(timestamp):
            return lower
        return min(max(timestamp.date(), lower), upper)

    if isinstance(value, (tuple, list)) and len(value) == 2:
        start, end = value
        if isinstance(start, date) and isinstance(end, date):
            start, end = clamp(start), clamp(end)
            return (start, end) if start <= end else (end, start)
    if isinstance(value, date):
        selected = clamp(value)
        return selected, selected
    return lower, upper


def _initialize_statistics_state(options: Mapping[str, Any]) -> None:
    """Preserve the existing filter-session keys without storing dataframes."""
    defaults = _default_filters(options)
    state_defaults: dict[str, Any] = {
        "statistics_date_range": (options["date_min"], options["date_max"]),
        "statistics_competitions": [],
        "statistics_team_filter": ALL_TEAMS_LABEL,
        "statistics_outcomes": [],
        "statistics_neutral_filter": "All venues",
        "statistics_minimum_goals": 0.0,
        "statistics_world_cup_only": False,
        "statistics_applied_filters": defaults,
        "statistics_filter_signature": statistics_filter_signature(defaults),
    }
    for key, value in state_defaults.items():
        st.session_state.setdefault(key, value)

    team_options = {ALL_TEAMS_LABEL, *map(str, options["teams"])}
    if st.session_state.statistics_team_filter not in team_options:
        st.session_state.statistics_team_filter = ALL_TEAMS_LABEL
    selected_competitions = st.session_state.statistics_competitions
    valid_competitions = {str(item) for item in options["competitions"]}
    st.session_state.statistics_competitions = (
        [str(item) for item in selected_competitions if str(item) in valid_competitions]
        if isinstance(selected_competitions, (list, tuple, set))
        else []
    )
    st.session_state.statistics_date_range = _normalize_date_range(
        st.session_state.statistics_date_range, options
    )
    if not options.get("has_neutral"):
        st.session_state.statistics_neutral_filter = "All venues"
    if not options.get("has_world_cup_finals"):
        st.session_state.statistics_world_cup_only = False

    try:
        applied = normalize_statistics_filters(
            st.session_state.statistics_applied_filters
        )
    except StatisticsDataError:
        applied = defaults
    applied["date_start"], applied["date_end"] = _normalize_date_range(
        (applied["date_start"], applied["date_end"]), options
    )
    applied["competitions"] = tuple(
        item for item in applied["competitions"] if item in valid_competitions
    )
    if applied["team"] not in team_options:
        applied["team"] = None
    if not options.get("has_neutral"):
        applied["neutral"] = "All venues"
    if not options.get("has_world_cup_finals"):
        applied["world_cup_only"] = False
    st.session_state.statistics_applied_filters = applied
    st.session_state.statistics_filter_signature = statistics_filter_signature(applied)


def _reset_statistics_filters(options: Mapping[str, Any]) -> None:
    """Restore the existing all-data filter selection from the source bounds."""
    defaults = _default_filters(options)
    st.session_state.statistics_date_range = (options["date_min"], options["date_max"])
    st.session_state.statistics_competitions = []
    st.session_state.statistics_team_filter = ALL_TEAMS_LABEL
    st.session_state.statistics_outcomes = []
    st.session_state.statistics_neutral_filter = "All venues"
    st.session_state.statistics_minimum_goals = 0.0
    st.session_state.statistics_world_cup_only = False
    st.session_state.statistics_applied_filters = defaults
    st.session_state.statistics_filter_signature = statistics_filter_signature(defaults)


def _render_styles() -> None:
    """Keep filter and KPI responsive treatments local to this page."""
    render_html("""
        <style>
            .statistics-filter-summary {
                margin: 0;
                color: var(--ui-color-text-secondary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.9rem;
            }

            .statistics-filter-chip {
                display: inline-block;
                margin: 0.2rem 0.32rem 0.2rem 0;
                padding: 0.25rem 0.52rem;
                border: 1px solid var(--ui-color-border-subtle);
                border-radius: var(--ui-radius-sm);
                background: color-mix(in srgb, var(--ui-color-primary) 11%, transparent);
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.76rem;
            }

            @media (max-width: 760px) {
                .st-key-statistics-filter-form [data-testid="stHorizontalBlock"],
                [class*="st-key-statistics-kpi-"] [data-testid="stHorizontalBlock"] {
                    flex-direction: column;
                }

                .st-key-statistics-filter-form [data-testid="stColumn"],
                [class*="st-key-statistics-kpi-"] [data-testid="stColumn"] {
                    width: 100% !important;
                    flex: 1 1 100% !important;
                }
            }
        </style>
        """)


def _render_header() -> None:
    """Render the shared Statistics heading without redundant source metadata."""
    page_header(
        "Statistics Dashboard",
        eyebrow="Historical international football",
        subtitle="Explore verified historical match data through the active filters and KPI cards.",
    )


def _filters_from_widget(options: Mapping[str, Any]) -> dict[str, Any]:
    """Build the service filter shape from the retained widgets."""
    start, end = _normalize_date_range(st.session_state.statistics_date_range, options)
    selected_team = st.session_state.statistics_team_filter
    return {
        "date_start": start,
        "date_end": end,
        "competitions": list(st.session_state.statistics_competitions),
        "team": None if selected_team == ALL_TEAMS_LABEL else selected_team,
        "outcomes": list(st.session_state.statistics_outcomes),
        "neutral": st.session_state.statistics_neutral_filter,
        "minimum_total_goals": st.session_state.statistics_minimum_goals,
        "world_cup_only": st.session_state.statistics_world_cup_only,
    }


def _render_filter_panel(options: Mapping[str, Any]) -> None:
    """Render the existing submitted-together historical filter form."""
    section_title("Explore historical matches", compact=True)
    with glass_card("statistics-filter-panel"):
        with st.form("statistics-filter-form", clear_on_submit=False):
            first_row = st.columns((1.25, 1, 1))
            with first_row[0]:
                st.date_input(
                    "Match date range",
                    min_value=options["date_min"],
                    max_value=options["date_max"],
                    key="statistics_date_range",
                )
            with first_row[1]:
                st.multiselect(
                    "Competitions",
                    options=list(options["competitions"]),
                    key="statistics_competitions",
                )
            with first_row[2]:
                st.selectbox(
                    "Team included",
                    options=[ALL_TEAMS_LABEL, *list(options["teams"])],
                    key="statistics_team_filter",
                )

            second_row = st.columns((1, 1, 1))
            with second_row[0]:
                st.multiselect(
                    "Match outcomes",
                    options=list(OUTCOME_ORDER),
                    key="statistics_outcomes",
                )
            with second_row[1]:
                neutral_options = ["All venues"]
                if options.get("has_neutral"):
                    neutral_options.extend(["Neutral venues", "Non-neutral venues"])
                if st.session_state.statistics_neutral_filter not in neutral_options:
                    st.session_state.statistics_neutral_filter = "All venues"
                st.selectbox(
                    "Venue label",
                    options=neutral_options,
                    key="statistics_neutral_filter",
                    disabled=not bool(options.get("has_neutral")),
                )
            with second_row[2]:
                st.number_input(
                    "Minimum total goals",
                    min_value=0.0,
                    step=0.5,
                    key="statistics_minimum_goals",
                )

            if options.get("has_world_cup_finals"):
                st.checkbox(
                    "FIFA World Cup finals only", key="statistics_world_cup_only"
                )

            submitted = st.form_submit_button(
                "Apply Filters", width="stretch", type="primary"
            )

        st.button(
            "Reset Filters",
            key="statistics-reset-filters",
            width="content",
            on_click=_reset_statistics_filters,
            args=(options,),
        )

    if submitted:
        try:
            applied = normalize_statistics_filters(_filters_from_widget(options))
        except StatisticsDataError as error:
            st.error(str(error))
            return
        st.session_state.statistics_applied_filters = applied
        st.session_state.statistics_filter_signature = statistics_filter_signature(
            applied
        )


def _filter_chips(filters: Mapping[str, Any], filtered_count: int) -> list[str]:
    """Summarize the active filters without recreating removed data sections."""
    chips = [f"{filtered_count:,} matches selected"]
    chips.append(
        f"{_format_date(filters.get('date_start'))} – {_format_date(filters.get('date_end'))}"
    )
    competitions = tuple(filters.get("competitions") or ())
    if competitions:
        chips.append(f"Competitions: {_short_list(competitions)}")
    if filters.get("team"):
        chips.append(f"Team: {filters['team']}")
    outcomes = tuple(filters.get("outcomes") or ())
    if outcomes:
        chips.append(f"Outcomes: {_short_list(outcomes)}")
    if filters.get("neutral") not in (None, "All venues"):
        chips.append(str(filters["neutral"]))
    if float(filters.get("minimum_total_goals") or 0) > 0:
        chips.append(f"At least {float(filters['minimum_total_goals']):g} total goals")
    if filters.get("world_cup_only"):
        chips.append("FIFA World Cup finals only")
    return chips


def _render_filter_summary(filters: Mapping[str, Any], filtered_count: int) -> None:
    """Keep the compact filter state visible before the retained KPIs."""
    section_title("Current selection")
    with glass_card("statistics-filter-summary"):
        markup = "".join(
            f'<span class="statistics-filter-chip">{escape(chip)}</span>'
            for chip in _filter_chips(filters, filtered_count)
        )
        render_html(f'<p class="statistics-filter-summary">{markup}</p>')
        if filtered_count == 0:
            st.warning("No historical matches match the active filters.")


def _render_kpis(kpis: Mapping[str, Any]) -> None:
    """Render the retained source-derived KPI cards."""
    section_title("Key performance indicators")
    cards = (
        ("Matches", _format_number(kpis.get("matches")), "All selected match rows"),
        (
            "Total goals",
            _format_number(kpis.get("total_goals")),
            "Valid scored matches",
        ),
        (
            "Goals per match",
            _format_number(kpis.get("goals_per_match"), decimals=2),
            "Valid scored-match denominator",
        ),
        (
            "Home win rate",
            _format_percentage(kpis.get("home_win_rate")),
            "Valid scored matches",
        ),
        (
            "Draw rate",
            _format_percentage(kpis.get("draw_rate")),
            "Valid scored matches",
        ),
        (
            "Away win rate",
            _format_percentage(kpis.get("away_win_rate")),
            "Valid scored matches",
        ),
        (
            "Both teams scored",
            _format_percentage(kpis.get("both_teams_scored_rate")),
            "Valid scored matches",
        ),
        (
            "Clean-sheet rate",
            _format_percentage(kpis.get("clean_sheet_rate")),
            "At least one team conceded zero",
        ),
        (
            "Scoreless draws",
            _format_percentage(kpis.get("scoreless_draw_rate")),
            "Valid scored matches",
        ),
    )
    for start in range(0, len(cards), 3):
        with st.container(key=f"statistics-kpi-{start}", border=False):
            columns = st.columns(3)
            for column, (label, value, caption) in zip(
                columns, cards[start : start + 3]
            ):
                with column:
                    metric_card(label, value, caption=caption)


def render_statistics_dashboard() -> None:
    """Render only the retained Statistics header, filters, and KPIs."""
    apply_theme()
    _render_styles()
    try:
        matches, _ = load_historical_matches()
        options = get_statistics_filter_options()
    except StatisticsDataError as error:
        page_header("Statistics Dashboard", eyebrow="Historical international football")
        st.error(str(error))
        return

    _initialize_statistics_state(options)
    _render_header()
    _render_filter_panel(options)
    try:
        filters = normalize_statistics_filters(
            st.session_state.statistics_applied_filters
        )
        st.session_state.statistics_filter_signature = statistics_filter_signature(
            filters
        )
        filtered = apply_statistics_filters(matches, filters)
        kpis = calculate_kpis(filtered)
    except StatisticsDataError as error:
        st.error(str(error))
        return

    _render_filter_summary(filters, len(filtered))
    with animated_container("statistics-results", animation="fade_up"):
        _render_kpis(kpis)


__all__ = ["render_statistics_dashboard"]
