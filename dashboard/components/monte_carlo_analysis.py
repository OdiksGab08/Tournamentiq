"""Render real repeated-tournament Monte Carlo analysis for TournamentIQ.

Purpose:
    Let users configure, run, inspect, and export the existing Monte Carlo
    tournament analysis without inventing simulated outcomes.
Responsibility:
    Coordinate Streamlit controls and session-local result presentation while
    delegating validation and simulation work to the service adapter.
Inputs:
    User-selected simulation settings, Streamlit session state, and normalized
    output from ``services.monte_carlo_service``.
Outputs:
    Progress feedback, championship rankings, contender cards, and stage results
    for completed real simulator runs.
Collaboration:
    Called by the Monte Carlo view and shares tournament metadata with the
    tournament-simulation service through the dedicated analysis adapter.
"""

from __future__ import annotations

import traceback
from html import escape
import re
from typing import Any, Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.team_selector import render_team_flag
from services.monte_carlo_service import (
    MonteCarloAnalysisError,
    get_monte_carlo_overview,
    is_monte_carlo_result_current,
    monte_carlo_signature,
    parse_monte_carlo_seed,
    run_monte_carlo_analysis,
    validate_monte_carlo_preflight,
)
from ui import (
    animated_container,
    apply_theme,
    glass_card,
    gradient_button,
    metric_card,
    page_header,
    section_title,
)


def _initialize_monte_carlo_state() -> None:
    """Create page-local state without putting model objects or frames in session."""
    defaults: dict[str, Any] = {
        "monte_carlo_simulation_count": 100,
        "monte_carlo_seed": "",
        "monte_carlo_result": None,
        "monte_carlo_error": None,
        "monte_carlo_signature": None,
        "monte_carlo_running": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _safe_key(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.casefold()).strip("-") or "item"


def _format_probability(value: Any, *, decimals: int = 1) -> str:
    try:
        return f"{float(value):.{decimals}%}"
    except (TypeError, ValueError):
        return "Unavailable"


def _render_monte_carlo_styles() -> None:
    """Apply focused glass, ranking, interval, and responsive page styling."""
    st.markdown(
        """
        <style>
            @keyframes mc-gradient-flow {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            .mc-setup-note {
                margin: 0;
                color: var(--ui-color-text-secondary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.9rem;
                line-height: 1.58;
            }

            .mc-setup-note { margin: 0.8rem 0 1rem; font-size: 0.82rem; }

            [class*="st-key-mc-field-team-"] {
                min-height: 6.25rem;
                padding: 0.78rem;
                border: 1px solid var(--ui-color-border-subtle);
                border-radius: var(--ui-radius-md);
                background: var(--ui-glass-background);
            }

            .mc-field-name {
                margin: 0.48rem 0 0;
                overflow: hidden;
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.84rem;
                font-weight: 750;
                text-overflow: ellipsis;
                white-space: nowrap;
            }

            [class*="st-key-mc-field-team-"] [data-testid="stImage"],
            [class*="st-key-mc-ranking-flag-"] [data-testid="stImage"] {
                width: fit-content;
                padding: 0.24rem;
                border: 1px solid var(--ui-color-border-subtle);
                border-radius: var(--ui-radius-sm);
                background: rgba(25, 17, 13, 0.5);
            }

            .mc-ranking-row {
                display: grid;
                grid-template-columns: auto minmax(8.25rem, 1.25fr) minmax(10rem, 2.3fr) minmax(5rem, 0.7fr);
                align-items: center;
                gap: 0.8rem;
                padding: 0.92rem 0;
                border-top: 1px solid var(--ui-color-border-subtle);
            }

            .mc-ranking-row:first-child { border-top: 0; }
            .mc-ranking-rank {
                display: grid;
                width: 1.8rem;
                height: 1.8rem;
                place-items: center;
                border: 1px solid rgba(214, 151, 71, 0.34);
                border-radius: 50%;
                background: rgba(185, 78, 19, 0.1);
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-mono);
                font-size: 0.72rem;
                font-weight: 800;
            }

            .mc-ranking-team {
                display: flex;
                align-items: center;
                gap: 0.55rem;
                min-width: 0;
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.88rem;
                font-weight: 750;
            }

            .mc-ranking-team__name {
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }

            .mc-ranking-bar-track {
                height: 0.55rem;
                overflow: hidden;
                border: 1px solid var(--ui-color-border-subtle);
                border-radius: 999px;
                background: var(--ui-color-surface-muted);
            }

            .mc-ranking-bar-fill {
                height: 100%;
                border-radius: inherit;
                background: linear-gradient(90deg, var(--ui-color-primary), var(--ui-color-accent));
            }

            .mc-ranking-meta {
                margin: 0.35rem 0 0;
                color: var(--ui-color-text-muted);
                font-family: var(--ui-type-font-sans);
                font-size: 0.68rem;
                line-height: 1.35;
            }

            .mc-ranking-probability {
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-mono);
                font-size: 0.87rem;
                font-weight: 800;
                text-align: right;
            }

            [class*="st-key-ui-glass-mc-contender-"] {
                position: relative;
                min-height: 12.2rem;
                overflow: hidden;
            }

            [class*="st-key-ui-glass-mc-contender-"]::before {
                content: "";
                position: absolute;
                top: 0;
                right: 0;
                left: 0;
                height: 3px;
                background: linear-gradient(90deg, var(--ui-color-primary), var(--ui-color-accent), var(--ui-color-warning), var(--ui-color-primary));
                background-size: 240% 100%;
                animation: mc-gradient-flow 4.5s linear infinite;
            }

            .mc-contender-rank {
                margin: 0;
                color: var(--ui-color-accent);
                font-family: var(--ui-type-font-mono);
                font-size: 0.68rem;
                font-weight: 800;
                letter-spacing: 0.12em;
                text-transform: uppercase;
            }

            .mc-contender-team {
                margin: 0.45rem 0 0;
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: 1.18rem;
                font-weight: 800;
                letter-spacing: -0.03em;
            }

            .mc-contender-probability {
                margin: 1.1rem 0 0;
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: clamp(1.6rem, 3vw, 2.35rem);
                font-weight: 850;
                letter-spacing: -0.055em;
                line-height: 1;
            }

            .mc-contender-meta {
                margin: 0.55rem 0 0;
                color: var(--ui-color-text-secondary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.74rem;
                line-height: 1.45;
            }

            .st-key-mc-ranking-chart [data-testid="stPlotlyChart"],
            .st-key-mc-stage-chart [data-testid="stPlotlyChart"] {
                border: 1px solid var(--ui-color-border-subtle);
                border-radius: var(--ui-radius-md);
                background: rgba(25, 17, 13, 0.34);
            }

            @media (max-width: 900px) {
                [class*="st-key-mc-setup-team-grid-"] [data-testid="stHorizontalBlock"],
                .st-key-mc-summary-grid [data-testid="stHorizontalBlock"],
                [class*="st-key-mc-contender-grid-"] [data-testid="stHorizontalBlock"] {
                    flex-direction: column;
                }

                [class*="st-key-mc-setup-team-grid-"] [data-testid="stColumn"],
                .st-key-mc-summary-grid [data-testid="stColumn"],
                [class*="st-key-mc-contender-grid-"] [data-testid="stColumn"] {
                    width: 100% !important;
                    flex: 1 1 100% !important;
                }
            }

            @media (max-width: 680px) {
                .mc-ranking-row { grid-template-columns: auto minmax(0, 1fr); gap: 0.5rem; }
                .mc-ranking-bar-track { grid-column: 2; }
                .mc-ranking-probability { grid-column: 2; text-align: left; }
            }

            @media (prefers-reduced-motion: reduce) {
                [class*="st-key-ui-glass-mc-contender-"]::before { animation: none; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    """Render the page hierarchy before the analysis controls."""
    page_header(
        "Monte Carlo Analysis",
        eyebrow="FIFA World Cup 2026 forecasting",
        subtitle=(
            "Run many complete tournaments through the trained match model to estimate "
            "how often each configured team reaches each knockout stage and wins the title."
        ),
    )


def _render_active_teams(teams: Sequence[str]) -> None:
    for start in range(0, len(teams), 4):
        with st.container(key=f"mc-setup-team-grid-{start}", border=False):
            columns = st.columns(4)
            for column, team in zip(columns, teams[start : start + 4]):
                with column:
                    with st.container(
                        key=f"mc-field-team-{_safe_key(str(team))}", border=False
                    ):
                        render_team_flag(str(team), width=30)
                        st.markdown(
                            f'<p class="mc-field-name">{escape(str(team))}</p>',
                            unsafe_allow_html=True,
                        )


def _render_setup(overview: Mapping[str, Any]) -> tuple[int | None, int | None, bool]:
    """Render only bounded real simulator controls and the active backend field."""
    supported_counts = list(overview["supported_simulation_counts"])
    if st.session_state.monte_carlo_simulation_count not in supported_counts:
        st.session_state.monte_carlo_simulation_count = supported_counts[0]
    section_title(
        "Analysis setup",
        eyebrow="Real sequential Monte Carlo workflow",
        description="The active tournament field is fixed by the current backend configuration. Higher simulation counts generally produce more stable frequency estimates but require more processing time.",
        compact=True,
    )
    with glass_card("mc-setup"):
        st.selectbox(
            "Number of tournaments to simulate",
            supported_counts,
            key="monte_carlo_simulation_count",
            format_func=lambda count: f"{count:,} tournaments",
            help="The page uses bounded options to keep the sequential simulator practical.",
        )
        simulation_count = int(st.session_state.monte_carlo_simulation_count)
        matches_per_tournament = max(len(overview["teams"]) - 1, 0)
        workload_text = (
            f"This run will simulate {simulation_count:,} complete "
            f"{matches_per_tournament}-match knockout tournaments "
            f"({simulation_count * matches_per_tournament:,} simulated matches) sequentially. "
            "Higher simulation counts generally produce more stable frequency estimates but require more processing time."
        )
        st.markdown(
            f'<p class="mc-setup-note">{escape(workload_text)}</p>',
            unsafe_allow_html=True,
        )
        st.text_input(
            "Optional simulation seed",
            key="monte_carlo_seed",
            placeholder="For example: 2026",
            help="Passed to the existing ProbabilitySimulator. Identical seeds are reproducible only in an isolated sequential process.",
        )
        st.markdown(
            '<p class="mc-setup-note">The current implementation runs sequentially and does not use parallel workers. A supplied seed repeats the existing sampling stream only in an isolated sequential process; leaving it blank produces a fresh stochastic run.</p>',
            unsafe_allow_html=True,
        )
        _render_active_teams(list(overview["teams"]))
        try:
            seed = parse_monte_carlo_seed(st.session_state.monte_carlo_seed)
        except MonteCarloAnalysisError as error:
            st.error(str(error))
            return None, None, False
        clicked = gradient_button(
            "Run Monte Carlo Analysis",
            key="monte-carlo-run-button",
            width="content",
            disabled=st.session_state.monte_carlo_running,
        )
    return simulation_count, seed, clicked


def _run_analysis(simulation_count: int, seed: int | None, signature: str) -> None:
    """Run the actual backend and report only genuine completed-run progress."""
    st.session_state.monte_carlo_running = True
    st.session_state.monte_carlo_error = None
    progress = None
    last_progress = 0
    try:
        with st.status(
            "Preparing real Monte Carlo analysis…", expanded=False
        ) as status:
            status.update(
                label="Preparing tournament configuration and team snapshots…",
                state="running",
            )
            validate_monte_carlo_preflight(simulation_count)
            status.update(
                label="Loading the cached trained match predictor…", state="running"
            )
            progress = st.progress(0, text="Running tournament simulations…")

            def on_progress(completed: int, total: int) -> None:
                nonlocal last_progress
                update_every = max(1, total // 20)
                if completed == total or completed - last_progress >= update_every:
                    progress.progress(
                        completed / total,
                        text=f"Completed {completed:,} of {total:,} real tournaments",
                    )
                    last_progress = completed

            status.update(
                label="Running tournament simulations and aggregating frequencies…",
                state="running",
            )
            result = run_monte_carlo_analysis(
                simulation_count, seed, progress_callback=on_progress
            )
            result["signature"] = signature
            st.session_state.monte_carlo_result = result
            st.session_state.monte_carlo_signature = signature
            status.update(label="Monte Carlo analysis complete", state="complete")
    except MonteCarloAnalysisError as error:
        st.session_state.monte_carlo_result = None
        st.session_state.monte_carlo_signature = signature
        st.session_state.monte_carlo_error = str(error)
    except Exception as error:
        st.error(f"{type(error).__name__}: {error}")
        st.code(traceback.format_exc())
        raise
    finally:
        if progress is not None:
            progress.empty()
        st.session_state.monte_carlo_running = False


def _render_summary(result: Mapping[str, Any]) -> None:
    rankings = list(result["rankings"])
    leader = rankings[0]
    section_title("Monte Carlo summary", eyebrow="Completed tournament frequency")
    with st.container(key="mc-summary-grid", border=False):
        count_column, teams_column, leader_column, probability_column = st.columns(4)
        with count_column:
            metric_card(
                "Tournaments simulated",
                f"{int(result['simulation_count']):,}",
                caption="Complete tournament runs",
            )
        with teams_column:
            metric_card(
                "Participating teams",
                len(result["teams"]),
                caption=str(result["tournament_format"]),
            )
        with leader_column:
            metric_card(
                "Most frequent simulated champion",
                str(leader["team"]),
                caption="Observed title frequency",
            )
        with probability_column:
            metric_card(
                "Most frequent champion probability",
                _format_probability(leader["champion_probability"]),
                caption="Monte Carlo frequency",
            )


def _build_ranking_chart(rankings: Sequence[Mapping[str, Any]]) -> go.Figure:
    ordered = list(reversed(rankings))
    figure = go.Figure(
        go.Bar(
            x=[float(record["champion_probability"]) * 100 for record in ordered],
            y=[str(record["team"]) for record in ordered],
            orientation="h",
            marker={"color": "#B94E13"},
            text=[
                _format_probability(record["champion_probability"])
                for record in ordered
            ],
            textposition="auto",
            customdata=[record.get("championships") for record in ordered],
            hovertemplate=(
                "<b>%{y}</b><br>Championship frequency: %{x:.1f}%"
                "<br>Simulated titles: %{customdata}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        height=max(360, 56 * len(ordered)),
        margin={"l": 95, "r": 30, "t": 24, "b": 38},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#FFF8F2", "family": "Manrope, Inter, system-ui, sans-serif"},
        xaxis={
            "title": "Championship frequency (%)",
            "range": [0, 100],
            "gridcolor": "rgba(255, 218, 184, 0.14)",
        },
        yaxis={"gridcolor": "rgba(255, 218, 184, 0.08)"},
        showlegend=False,
    )
    return figure


def _render_ranking_row(record: Mapping[str, Any]) -> None:
    team = str(record["team"])
    probability = float(record["champion_probability"])
    titles = record.get("championships")
    with st.container(key=f"mc-ranking-{_safe_key(team)}", border=False):
        flag_column, content_column = st.columns((0.18, 1))
        with flag_column:
            with st.container(key=f"mc-ranking-flag-{_safe_key(team)}", border=False):
                render_team_flag(team, width=28)
        with content_column:
            st.markdown(
                f"""
                <div class="mc-ranking-row">
                    <span class="mc-ranking-rank">{int(record["rank"])}</span>
                    <div class="mc-ranking-team"><span class="mc-ranking-team__name">{escape(team)}</span></div>
                    <div>
                        <div class="mc-ranking-bar-track"><div class="mc-ranking-bar-fill" style="width: {probability * 100:.2f}%"></div></div>
                        <p class="mc-ranking-meta">{escape(f"{titles:,} simulated titles" if isinstance(titles, int) else "Title count unavailable")}</p>
                    </div>
                    <span class="mc-ranking-probability">{_format_probability(probability)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_rankings(result: Mapping[str, Any]) -> None:
    rankings = list(result["rankings"])
    section_title(
        "Championship probability ranking",
        eyebrow="Observed title frequencies",
        description="Teams are ranked from real championship counts across the completed tournament runs. The top line is a frequency estimate, not certainty.",
    )
    with glass_card("mc-ranking-chart-card"):
        with st.container(key="mc-ranking-chart", border=False):
            st.plotly_chart(
                _build_ranking_chart(rankings),
                width="stretch",
                config={"displayModeBar": False, "responsive": True},
            )
    with glass_card("mc-ranking-list"):
        for record in rankings:
            _render_ranking_row(record)


def _render_top_contenders(result: Mapping[str, Any]) -> None:
    contenders = list(result["rankings"][:5])
    section_title("Top contenders", eyebrow="Top five observed title frequencies")
    for start in range(0, len(contenders), 3):
        with st.container(key=f"mc-contender-grid-{start}", border=False):
            columns = st.columns(3)
            for column, record in zip(columns, contenders[start : start + 3]):
                team = str(record["team"])
                with column:
                    with glass_card(f"mc-contender-{_safe_key(team)}"):
                        st.markdown(
                            f'<p class="mc-contender-rank">Rank {int(record["rank"])}</p>',
                            unsafe_allow_html=True,
                        )
                        render_team_flag(team, width=38)
                        st.markdown(
                            f'<h3 class="mc-contender-team">{escape(team)}</h3>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<p class="mc-contender-probability">{_format_probability(record["champion_probability"])}</p>',
                            unsafe_allow_html=True,
                        )
                        titles = record.get("championships")
                        details = (
                            f"{titles:,} tournament wins"
                            if isinstance(titles, int)
                            else "Tournament win count unavailable"
                        )
                        st.markdown(
                            f'<p class="mc-contender-meta">{escape(details)}</p>',
                            unsafe_allow_html=True,
                        )


def _build_stage_chart(stage_probabilities: Sequence[Mapping[str, Any]]) -> go.Figure:
    teams = [str(record["team"]) for record in stage_probabilities]
    figure = go.Figure()
    colors = {
        "Quarter-finals": "#9B887B",
        "Semi-finals": "#B94E13",
        "Final": "#D69747",
        "Champion": "#57B38F",
    }
    for stage in ("Quarter-finals", "Semi-finals", "Final", "Champion"):
        figure.add_trace(
            go.Bar(
                name=stage,
                y=teams,
                x=[float(record[stage]) * 100 for record in stage_probabilities],
                orientation="h",
                marker={"color": colors[stage]},
                hovertemplate=f"<b>%{{y}}</b><br>{stage}: %{{x:.1f}}%<extra></extra>",
            )
        )
    figure.update_layout(
        barmode="group",
        height=max(380, 58 * len(teams)),
        margin={"l": 95, "r": 24, "t": 28, "b": 42},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#FFF8F2", "family": "Manrope, Inter, system-ui, sans-serif"},
        xaxis={
            "title": "Stage reach frequency (%)",
            "range": [0, 100],
            "gridcolor": "rgba(255, 218, 184, 0.14)",
        },
        yaxis={"autorange": "reversed"},
        legend={"orientation": "h", "y": -0.15, "x": 0.5, "xanchor": "center"},
    )
    return figure


def _render_stage_probabilities(result: Mapping[str, Any]) -> None:
    section_title(
        "Stage-reach probabilities",
        eyebrow="Real tournament appearance frequencies",
        description="Stage counts are aggregated from the same detailed tournament runs as the championship distribution.",
    )
    stages = result.get("stage_probabilities")
    if not stages:
        st.info(
            "The current Monte Carlo backend exposes championship outcomes but does not expose stage-by-stage appearance counts."
        )
        return
    with glass_card("mc-stage-chart-card"):
        with st.container(key="mc-stage-chart", border=False):
            st.plotly_chart(
                _build_stage_chart(stages),
                width="stretch",
                config={"displayModeBar": False, "responsive": True},
            )
        stage_frame = pd.DataFrame(stages)
        for column in ("Quarter-finals", "Semi-finals", "Final", "Champion"):
            stage_frame[column] = stage_frame[column].map(_format_probability)
        st.dataframe(stage_frame, width="stretch", hide_index=True)


def render_monte_carlo_analysis_page() -> None:
    """Render Monte Carlo Analysis only inside the existing custom route branch."""
    _initialize_monte_carlo_state()
    apply_theme()
    _render_monte_carlo_styles()
    try:
        overview = get_monte_carlo_overview()
    except MonteCarloAnalysisError as error:
        page_header("Monte Carlo Analysis", eyebrow="FIFA World Cup 2026 forecasting")
        st.error(str(error))
        return

    _render_header()
    for warning in overview.get("warnings", []):
        st.warning(str(warning))
    simulation_count, seed, clicked = _render_setup(overview)
    if simulation_count is None or (
        seed is None and str(st.session_state.monte_carlo_seed).strip()
    ):
        return

    try:
        signature = monte_carlo_signature(simulation_count, seed)
    except MonteCarloAnalysisError as error:
        st.error(str(error))
        return

    if st.session_state.monte_carlo_signature != signature:
        st.session_state.monte_carlo_error = None

    if clicked:
        _run_analysis(simulation_count, seed, signature)

    if (
        st.session_state.monte_carlo_error
        and st.session_state.monte_carlo_signature == signature
    ):
        st.error(st.session_state.monte_carlo_error)

    result = st.session_state.monte_carlo_result
    if result and not is_monte_carlo_result_current(result, signature):
        st.info(
            "Analysis settings changed. Run Monte Carlo Analysis to build current frequency estimates."
        )
        result = None

    if not result:
        return

    with animated_container("mc-results", animation="fade_up"):
        _render_summary(result)
        _render_rankings(result)
        _render_top_contenders(result)
        _render_stage_probabilities(result)
