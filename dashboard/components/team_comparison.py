"""Render a data-grounded national-team comparison dashboard page.

Purpose:
    Help users compare two teams through real engineered indicators and the
    optional trained-model outlook already supplied by the project.
Responsibility:
    Own controls, session presentation, and visualization while keeping the
    comparison calculation and inference in the service layer.
Inputs:
    Team selections, Streamlit state, and normalized comparison records from
    ``services.team_comparison_service``.
Outputs:
    Comparative metric cards, feature views, historic context, and real model
    outlook rendered in the active Streamlit page.
Collaboration:
    Invoked by the Team Comparison view and reuses service-backed selectors and
    shared UI primitives.
"""

from __future__ import annotations

import traceback
from html import escape
import re
from typing import Any, Mapping

import plotly.graph_objects as go
import streamlit as st

from components.team_selector import render_team_flag
from services.team_comparison_service import (
    FEATURE_CONFIG,
    FeatureMetadata,
    TeamComparisonError,
    build_team_comparison,
    comparison_signature,
    format_feature_value,
    get_comparison_teams,
    get_snapshot_version,
    is_comparison_current,
)
from ui import (
    animated_container,
    glass_card,
    gradient_button,
    metric_card,
    page_header,
    section_title,
)
from ui.theme import apply_theme


def _initialize_comparison_state() -> None:
    """Initialize only Team Comparison session-state keys."""
    defaults: dict[str, Any] = {
        "comparison_team_a": None,
        "comparison_team_b": None,
        "comparison_result": None,
        "comparison_signature": None,
        "comparison_error": None,
        "comparison_running": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_comparison_styles() -> None:
    """Add page-scoped visual and responsive treatments absent from shared UI."""
    st.markdown(
        """
        <style>
            @keyframes comparison-accent-flow {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            .comparison-team-label,
            .comparison-kicker,
            .comparison-card-label,
            .comparison-strength-label {
                margin: 0 0 0.45rem;
                color: var(--ui-color-accent);
                font-family: var(--ui-type-font-mono);
                font-size: 0.68rem;
                font-weight: 800;
                letter-spacing: 0.13em;
                text-transform: uppercase;
            }

            .comparison-versus {
                display: grid;
                width: 3rem;
                height: 3rem;
                place-items: center;
                margin: 2.25rem auto 0;
                border: 1px solid rgba(214, 151, 71, 0.3);
                border-radius: var(--ui-radius-md);
                background: linear-gradient(135deg, rgba(185, 78, 19, 0.18), rgba(214, 151, 71, 0.13));
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-mono);
                font-size: 0.82rem;
                font-weight: 800;
                letter-spacing: 0.08em;
            }

            .comparison-selector-note,
            .comparison-summary-text,
            .comparison-empty-text {
                margin: 0;
                color: var(--ui-color-text-secondary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.92rem;
                line-height: 1.6;
            }

            .comparison-selector-note { margin: 0.85rem 0 1.2rem; font-size: 0.84rem; }

            .st-key-comparison-team-a-flag [data-testid="stImage"],
            .st-key-comparison-team-b-flag [data-testid="stImage"],
            [class*="st-key-comparison-identity-"][class*="-flag"] [data-testid="stImage"] {
                width: fit-content;
                margin-top: 0.6rem;
                padding: 0.32rem;
                border: 1px solid var(--ui-color-border-subtle);
                border-radius: var(--ui-radius-sm);
                background: rgba(25, 17, 13, 0.5);
            }

            .comparison-identity-team {
                margin: 0.35rem 0 0;
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: clamp(1.3rem, 2.6vw, 2.15rem);
                font-weight: 800;
                letter-spacing: -0.045em;
                line-height: 1.05;
            }

            .comparison-identity-meta {
                margin: 0.62rem 0 0;
                color: var(--ui-color-text-secondary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.82rem;
                line-height: 1.5;
            }

            .comparison-score-panel {
                margin: 1rem auto;
                padding: 0.9rem;
                border: 1px solid var(--ui-color-border-subtle);
                border-radius: var(--ui-radius-md);
                background: var(--ui-glass-background);
                text-align: center;
            }

            .comparison-score-value {
                margin: 0.22rem 0 0;
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: clamp(1.35rem, 2.4vw, 2rem);
                font-weight: 800;
                letter-spacing: -0.05em;
            }

            .comparison-score-caption {
                margin: 0.45rem 0 0;
                color: var(--ui-color-text-muted);
                font-family: var(--ui-type-font-sans);
                font-size: 0.72rem;
                line-height: 1.35;
            }

            .comparison-summary-text { font-size: 1rem; }

            [class*="st-key-ui-glass-comparison-metric-"] {
                position: relative;
                min-height: 11.5rem;
                overflow: hidden;
            }

            [class*="st-key-ui-glass-comparison-metric-"]::before {
                content: "";
                position: absolute;
                top: 0;
                right: 0;
                left: 0;
                height: 3px;
                background: linear-gradient(90deg, var(--ui-color-primary), var(--ui-color-accent), var(--ui-color-primary));
                background-size: 200% 100%;
                animation: comparison-accent-flow 4.2s linear infinite;
            }

            .comparison-card-label { color: var(--ui-color-text-secondary); }
            .comparison-card-values {
                display: grid;
                grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
                gap: 0.75rem;
                margin-top: 0.9rem;
            }

            .comparison-card-side { min-width: 0; }
            .comparison-card-side--b { text-align: right; }
            .comparison-card-team {
                overflow: hidden;
                color: var(--ui-color-text-muted);
                font-family: var(--ui-type-font-sans);
                font-size: 0.68rem;
                font-weight: 700;
                text-overflow: ellipsis;
                white-space: nowrap;
            }

            .comparison-card-value {
                margin-top: 0.2rem;
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: clamp(1.15rem, 2vw, 1.6rem);
                font-weight: 800;
                letter-spacing: -0.035em;
            }

            .comparison-card-footer {
                margin: 0.75rem 0 0;
                color: var(--ui-color-accent);
                font-family: var(--ui-type-font-sans);
                font-size: 0.76rem;
                line-height: 1.42;
            }

            .comparison-card-description {
                margin: 0.42rem 0 0;
                color: var(--ui-color-text-muted);
                font-family: var(--ui-type-font-sans);
                font-size: 0.7rem;
                line-height: 1.35;
            }

            .st-key-comparison-radar-chart [data-testid="stPlotlyChart"] {
                border: 1px solid var(--ui-color-border-subtle);
                border-radius: var(--ui-radius-md);
                background: rgba(25, 17, 13, 0.34);
            }

            .comparison-strength-row {
                display: grid;
                grid-template-columns: minmax(5rem, 0.75fr) minmax(12rem, 2.5fr) minmax(5rem, 0.75fr);
                align-items: center;
                gap: 0.85rem;
                padding: 0.9rem 0;
                border-top: 1px solid var(--ui-color-border-subtle);
            }

            .comparison-strength-row:first-child { border-top: 0; }
            .comparison-strength-value {
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-mono);
                font-size: 0.85rem;
                font-weight: 700;
            }

            .comparison-strength-value--b { text-align: right; }
            .comparison-strength-center { min-width: 0; text-align: center; }
            .comparison-strength-label { margin-bottom: 0.28rem; color: var(--ui-color-text-primary); letter-spacing: 0.04em; }

            .comparison-strength-track {
                display: grid;
                grid-template-columns: 1fr 1fr;
                height: 0.48rem;
                overflow: hidden;
                border-radius: 999px;
                background: var(--ui-color-surface-muted);
            }

            .comparison-strength-side { display: flex; align-items: stretch; }
            .comparison-strength-side--a { justify-content: flex-end; border-right: 1px solid var(--ui-color-border-subtle); }
            .comparison-strength-side--b { justify-content: flex-start; }
            .comparison-strength-fill--a { background: var(--ui-color-primary); }
            .comparison-strength-fill--b { background: var(--ui-color-success); }

            .comparison-strength-favourable {
                display: block;
                margin-top: 0.34rem;
                color: var(--ui-color-text-muted);
                font-family: var(--ui-type-font-sans);
                font-size: 0.68rem;
            }

            .comparison-form-title {
                margin: 0;
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: 1.05rem;
                font-weight: 800;
                letter-spacing: -0.02em;
            }

            .comparison-form-row,
            .comparison-h2h-row {
                display: grid;
                grid-template-columns: auto minmax(0, 1fr) auto;
                align-items: center;
                gap: 0.65rem;
                padding: 0.76rem 0;
                border-top: 1px solid var(--ui-color-border-subtle);
                color: var(--ui-color-text-secondary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.82rem;
            }

            .comparison-form-row:first-of-type,
            .comparison-h2h-row:first-of-type { margin-top: 0.65rem; }
            .comparison-form-status {
                display: inline-grid;
                width: 1.6rem;
                height: 1.6rem;
                place-items: center;
                border-radius: 50%;
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-mono);
                font-size: 0.68rem;
                font-weight: 800;
            }

            .comparison-form-status--W { background: rgba(87, 179, 143, 0.78); }
            .comparison-form-status--D { background: rgba(227, 161, 59, 0.78); }
            .comparison-form-status--L { background: rgba(217, 101, 87, 0.78); }
            .comparison-form-status--U { background: rgba(155, 136, 123, 0.45); }

            .comparison-form-detail, .comparison-h2h-detail { min-width: 0; }
            .comparison-form-detail strong, .comparison-h2h-detail strong { color: var(--ui-color-text-primary); }
            .comparison-form-subdetail, .comparison-h2h-subdetail {
                display: block;
                margin-top: 0.16rem;
                overflow: hidden;
                color: var(--ui-color-text-muted);
                font-size: 0.71rem;
                text-overflow: ellipsis;
                white-space: nowrap;
            }

            .comparison-form-score, .comparison-h2h-score {
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-mono);
                font-size: 0.84rem;
                font-weight: 700;
                text-align: right;
            }

            .comparison-h2h-summary {
                margin: 0 0 0.72rem;
                color: var(--ui-color-text-secondary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.82rem;
            }

            .comparison-outlook-result {
                margin: 0 0 1rem;
                color: var(--ui-color-text-secondary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.9rem;
                line-height: 1.5;
            }

            @media (max-width: 760px) {
                .st-key-ui-glass-comparison-selectors [data-testid="stHorizontalBlock"],
                .st-key-ui-glass-comparison-identity [data-testid="stHorizontalBlock"],
                .st-key-comparison-key-metrics [data-testid="stHorizontalBlock"],
                .st-key-comparison-recent-grid [data-testid="stHorizontalBlock"],
                .st-key-comparison-model-probabilities [data-testid="stHorizontalBlock"] {
                    flex-direction: column;
                }

                .st-key-ui-glass-comparison-selectors [data-testid="stColumn"],
                .st-key-ui-glass-comparison-identity [data-testid="stColumn"],
                .st-key-comparison-key-metrics [data-testid="stColumn"],
                .st-key-comparison-recent-grid [data-testid="stColumn"],
                .st-key-comparison-model-probabilities [data-testid="stColumn"] {
                    width: 100% !important;
                    flex: 1 1 100% !important;
                }

                .comparison-versus { margin: 0.65rem auto; }
                .comparison-score-panel { margin: 0.15rem 0; text-align: left; }
                .comparison-card-values { gap: 0.55rem; }
                .comparison-strength-row { grid-template-columns: 1fr; gap: 0.35rem; }
                .comparison-strength-value--b, .comparison-strength-center { text-align: left; }
                .comparison-form-row, .comparison-h2h-row { grid-template-columns: auto minmax(0, 1fr); }
                .comparison-form-score, .comparison-h2h-score { grid-column: 2; text-align: left; }
            }

            @media (prefers-reduced-motion: reduce) {
                [class*="st-key-ui-glass-comparison-metric-"]::before { animation: none; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    page_header(
        "Team Comparison",
        eyebrow="FIFA World Cup 2026 analytics",
        subtitle=(
            "Compare two national teams through the project’s latest engineered "
            "snapshots, real match history, and optional trained-model outlook."
        ),
    )


def _render_selectors(teams: list[str]) -> tuple[str, str, bool]:
    """Render accessible, duplicate-safe selectors for the two comparison sides."""
    if len(teams) < 2:
        raise TeamComparisonError("At least two teams are required for comparison.")

    if st.session_state.comparison_team_a not in teams:
        st.session_state.comparison_team_a = teams[0]

    section_title(
        "Select two teams",
        eyebrow="Comparison setup",
        description="The page uses the same dynamic simulator/snapshot team source as Match Predictor.",
        compact=True,
    )
    with glass_card("comparison-selectors"):
        team_a_column, versus_column, team_b_column = st.columns((1, 0.3, 1))
        with team_a_column:
            st.markdown(
                '<p class="comparison-team-label">Team A</p>', unsafe_allow_html=True
            )
            team_a = st.selectbox(
                "Team A",
                teams,
                key="comparison_team_a",
                label_visibility="collapsed",
            )
            with st.container(key="comparison-team-a-flag", border=False):
                render_team_flag(team_a)

        team_b_options = [
            team for team in teams if team.casefold() != team_a.casefold()
        ]
        if st.session_state.comparison_team_b not in team_b_options:
            st.session_state.comparison_team_b = team_b_options[0]

        with versus_column:
            st.markdown(
                '<div class="comparison-versus" aria-label="versus">VS</div>',
                unsafe_allow_html=True,
            )

        with team_b_column:
            st.markdown(
                '<p class="comparison-team-label">Team B</p>', unsafe_allow_html=True
            )
            team_b = st.selectbox(
                "Team B",
                team_b_options,
                key="comparison_team_b",
                label_visibility="collapsed",
            )
            with st.container(key="comparison-team-b-flag", border=False):
                render_team_flag(team_b)

        st.markdown(
            """
            <p class="comparison-selector-note">
                The feature-based view is a transparent indicator comparison. The trained
                three-outcome model is kept separate and is requested only after comparison.
            </p>
            """,
            unsafe_allow_html=True,
        )
        clicked = gradient_button(
            "Compare Teams",
            key="comparison-run-button",
            width="content",
            disabled=st.session_state.comparison_running,
        )
    return team_a, team_b, clicked


def _run_comparison(team_a: str, team_b: str, signature: str) -> None:
    """Run one real comparison and save only normalized display-safe output."""
    st.session_state.comparison_running = True
    st.session_state.comparison_error = None
    try:
        with st.status(
            "Building comparison from real project data…", expanded=False
        ) as status:
            status.update(
                label="Loading engineered snapshots and match history…", state="running"
            )
            result = build_team_comparison(team_a, team_b, include_model_outlook=True)
            st.session_state.comparison_result = result
            st.session_state.comparison_signature = result["signature"]
            status.update(label="Team comparison complete", state="complete")
    except TeamComparisonError as error:
        st.session_state.comparison_result = None
        st.session_state.comparison_signature = signature
        st.session_state.comparison_error = str(error)
    except Exception as error:
        st.error(f"{type(error).__name__}: {error}")
        st.code(traceback.format_exc())
        raise
    finally:
        st.session_state.comparison_running = False


def _format_score(value: Any) -> str:
    """Format a feature-based index for display, never as a probability."""
    try:
        return f"{float(value) * 100:.0f}"
    except (TypeError, ValueError):
        return "—"


def _favourable_text(metric: Mapping[str, Any], team_a: str, team_b: str) -> str:
    side = metric.get("favorable_side")
    if side == "team_a":
        return f"Favourable: {team_a}"
    if side == "team_b":
        return f"Favourable: {team_b}"
    return "Favourable: level"


def _format_difference(metric: Mapping[str, Any]) -> str:
    metadata = _metric_metadata(metric)
    try:
        difference = abs(float(metric["value_a"]) - float(metric["value_b"]))
    except (TypeError, ValueError):
        return "Unavailable"
    return format_feature_value(difference, metadata)


def _metric_metadata(metric: Mapping[str, Any]) -> FeatureMetadata:
    """Recover display metadata using the centralized service feature config values."""
    return FEATURE_CONFIG[str(metric["key"])]


def _render_identity_header(result: Mapping[str, Any]) -> None:
    team_a = result["team_a"]
    team_b = result["team_b"]
    verdict = result["verdict"]
    section_title("Team snapshots", eyebrow="Latest available engineered context")
    with glass_card("comparison-identity"):
        column_a, score_column, column_b = st.columns((1, 1.1, 1))
        with column_a:
            with st.container(key="comparison-identity-a-flag", border=False):
                render_team_flag(str(team_a["name"]), width=74)
            st.markdown(
                '<p class="comparison-kicker">Team A</p>', unsafe_allow_html=True
            )
            st.markdown(
                f'<h2 class="comparison-identity-team">{escape(str(team_a["name"]))}</h2>',
                unsafe_allow_html=True,
            )
            _render_identity_meta(team_a)
        with score_column:
            if (
                verdict.get("score_a") is not None
                and verdict.get("score_b") is not None
            ):
                st.markdown(
                    f"""
                    <div class="comparison-score-panel">
                        <p class="comparison-kicker">Feature-based index</p>
                        <p class="comparison-score-value">{_format_score(verdict["score_a"])} · {_format_score(verdict["score_b"])}</p>
                        <p class="comparison-score-caption">Equal-weight normalized indicator index, not win probability</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with column_b:
            with st.container(key="comparison-identity-b-flag", border=False):
                render_team_flag(str(team_b["name"]), width=74)
            st.markdown(
                '<p class="comparison-kicker">Team B</p>', unsafe_allow_html=True
            )
            st.markdown(
                f'<h2 class="comparison-identity-team">{escape(str(team_b["name"]))}</h2>',
                unsafe_allow_html=True,
            )
            _render_identity_meta(team_b)


def _render_identity_meta(team: Mapping[str, Any]) -> None:
    meta_lines = []
    if team.get("snapshot_date"):
        meta_lines.append(f"Snapshot: {escape(str(team['snapshot_date']))}")
    if team.get("strength_label"):
        meta_lines.append(escape(str(team["strength_label"])))
    if meta_lines:
        st.markdown(
            f'<p class="comparison-identity-meta">{"<br>".join(meta_lines)}</p>',
            unsafe_allow_html=True,
        )


def _render_summary(result: Mapping[str, Any]) -> None:
    team_a_name = str(result["team_a"]["name"])
    team_b_name = str(result["team_b"]["name"])
    verdict = result["verdict"]
    section_title(
        "Feature-based comparison",
        eyebrow="Transparent indicator summary",
        description="A deterministic equal-weight comparison of the available engineered features—not a model prediction.",
    )
    with glass_card("comparison-summary"):
        st.markdown(
            f'<p class="comparison-summary-text">{escape(str(verdict["summary"]))}</p>',
            unsafe_allow_html=True,
        )
        if verdict.get("score_a") is not None and verdict.get("score_b") is not None:
            st.caption(
                f"Index: {team_a_name} {_format_score(verdict['score_a'])} · "
                f"{team_b_name} {_format_score(verdict['score_b'])}"
            )


def _render_metric_card(metric: Mapping[str, Any], team_a: str, team_b: str) -> None:
    metadata = _metric_metadata(metric)
    key = str(metric["key"])
    with glass_card(f"comparison-metric-{key}"):
        st.markdown(
            f"""
            <p class="comparison-card-label">{escape(str(metric["label"]))}</p>
            <div class="comparison-card-values">
                <div class="comparison-card-side">
                    <div class="comparison-card-team">{escape(team_a)}</div>
                    <div class="comparison-card-value">{escape(format_feature_value(metric["value_a"], metadata))}</div>
                </div>
                <div class="comparison-card-side comparison-card-side--b">
                    <div class="comparison-card-team">{escape(team_b)}</div>
                    <div class="comparison-card-value">{escape(format_feature_value(metric["value_b"], metadata))}</div>
                </div>
            </div>
            <p class="comparison-card-footer">{escape(_favourable_text(metric, team_a, team_b))} · Difference: {escape(_format_difference(metric))}</p>
            <p class="comparison-card-description">{escape(str(metric["description"]))}</p>
            """,
            unsafe_allow_html=True,
        )


def _render_key_metrics(result: Mapping[str, Any]) -> None:
    metrics = list(result.get("key_metrics", []))
    if not metrics:
        return
    team_a = str(result["team_a"]["name"])
    team_b = str(result["team_b"]["name"])
    section_title(
        "Key engineered indicators",
        eyebrow="Raw snapshot values",
        description="The favourable side respects each metric’s configured direction; raw values remain visible.",
    )
    for start in range(0, len(metrics), 3):
        with st.container(key=f"comparison-key-metrics-{start}", border=False):
            columns = st.columns(3)
            for column, metric in zip(columns, metrics[start : start + 3]):
                with column:
                    _render_metric_card(metric, team_a, team_b)


def _build_radar_chart(result: Mapping[str, Any]) -> go.Figure | None:
    metrics = list(result.get("radar_metrics", []))
    if len(metrics) < 3:
        return None
    labels = [str(metric["label"]) for metric in metrics]
    values_a = [float(metric["normalized_a"]) * 100 for metric in metrics]
    values_b = [float(metric["normalized_b"]) * 100 for metric in metrics]
    labels.append(labels[0])
    values_a.append(values_a[0])
    values_b.append(values_b[0])

    figure = go.Figure()
    figure.add_trace(
        go.Scatterpolar(
            r=values_a,
            theta=labels,
            fill="toself",
            fillcolor="rgba(185, 78, 19, 0.18)",
            line={"color": "#B94E13", "width": 2},
            marker={"color": "#B94E13", "size": 5},
            name=str(result["team_a"]["name"]),
            hovertemplate="%{theta}: %{r:.0f}/100<extra>%{fullData.name}</extra>",
        )
    )
    figure.add_trace(
        go.Scatterpolar(
            r=values_b,
            theta=labels,
            fill="toself",
            fillcolor="rgba(87, 179, 143, 0.15)",
            line={"color": "#57B38F", "width": 2},
            marker={"color": "#57B38F", "size": 5},
            name=str(result["team_b"]["name"]),
            hovertemplate="%{theta}: %{r:.0f}/100<extra>%{fullData.name}</extra>",
        )
    )
    figure.update_layout(
        height=430,
        margin={"l": 34, "r": 34, "t": 28, "b": 28},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#FFF8F2", "family": "Manrope, Inter, system-ui, sans-serif"},
        legend={"orientation": "h", "y": -0.12, "x": 0.5, "xanchor": "center"},
        polar={
            "bgcolor": "rgba(25, 17, 13, 0.42)",
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
                "showticklabels": False,
                "gridcolor": "rgba(255, 218, 184, 0.14)",
                "linecolor": "rgba(255, 218, 184, 0.14)",
            },
            "angularaxis": {
                "gridcolor": "rgba(255, 218, 184, 0.14)",
                "linecolor": "rgba(255, 218, 184, 0.14)",
                "tickfont": {"size": 11, "color": "#FFF8F2"},
            },
        },
    )
    return figure


def _render_radar(result: Mapping[str, Any]) -> None:
    section_title(
        "Normalized feature radar",
        eyebrow="Comparable scales",
        description="Each axis uses the same direction-aware two-team normalization; the raw values are shown above and below.",
    )
    figure = _build_radar_chart(result)
    with glass_card("comparison-radar"):
        if figure is None:
            st.info(
                "A radar chart needs at least three comparable engineered features; "
                "the current snapshots do not provide enough valid values."
            )
        else:
            with st.container(key="comparison-radar-chart", border=False):
                st.plotly_chart(
                    figure,
                    width="stretch",
                    config={"displayModeBar": False, "responsive": True},
                )


def _render_strength_bars(result: Mapping[str, Any]) -> None:
    team_a = str(result["team_a"]["name"])
    team_b = str(result["team_b"]["name"])
    section_title(
        "Side-by-side strengths",
        eyebrow="Direction-aware comparison",
        description="Bar widths are normalized for visual comparison only; the displayed values are the original engineered metrics.",
    )
    with glass_card("comparison-strength-bars"):
        for metric in result.get("metrics", []):
            metadata = _metric_metadata(metric)
            width_a = max(0.0, min(100.0, float(metric["normalized_a"]) * 100))
            width_b = max(0.0, min(100.0, float(metric["normalized_b"]) * 100))
            st.markdown(
                f"""
                <div class="comparison-strength-row">
                    <span class="comparison-strength-value">{escape(format_feature_value(metric["value_a"], metadata))}</span>
                    <div class="comparison-strength-center">
                        <span class="comparison-strength-label">{escape(str(metric["label"]))}</span>
                        <div class="comparison-strength-track" aria-label="{escape(str(metric["label"]))} comparison">
                            <div class="comparison-strength-side comparison-strength-side--a">
                                <span class="comparison-strength-fill--a" style="width: {width_a:.2f}%"></span>
                            </div>
                            <div class="comparison-strength-side comparison-strength-side--b">
                                <span class="comparison-strength-fill--b" style="width: {width_b:.2f}%"></span>
                            </div>
                        </div>
                        <span class="comparison-strength-favourable">{escape(_favourable_text(metric, team_a, team_b))}</span>
                    </div>
                    <span class="comparison-strength-value comparison-strength-value--b">{escape(format_feature_value(metric["value_b"], metadata))}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_form_records(
    team: Mapping[str, Any], records: list[Mapping[str, Any]]
) -> None:
    team_name = str(team["name"])
    safe_key = re.sub(r"[^a-z0-9-]+", "-", team_name.casefold()).strip("-")
    with st.container(key=f"comparison-form-{safe_key or 'team'}", border=False):
        st.markdown(
            f'<h3 class="comparison-form-title">{escape(team_name)}</h3>',
            unsafe_allow_html=True,
        )
        if not records:
            st.markdown(
                '<p class="comparison-empty-text">No recent match records were available for this team.</p>',
                unsafe_allow_html=True,
            )
            return
        for record in records:
            result = str(record.get("result") or "U")
            status = result if result in {"W", "D", "L"} else "U"
            details = [
                value
                for value in (
                    record.get("date"),
                    record.get("venue"),
                    record.get("tournament"),
                )
                if value
            ]
            opponent = escape(str(record.get("opponent") or "Opponent unavailable"))
            st.markdown(
                f"""
                <div class="comparison-form-row">
                    <span class="comparison-form-status comparison-form-status--{status}">{escape(result if result != "U" else "—")}</span>
                    <div class="comparison-form-detail">
                        <strong>vs {opponent}</strong>
                        <span class="comparison-form-subdetail">{escape(" · ".join(str(value) for value in details))}</span>
                    </div>
                    <span class="comparison-form-score">{escape(str(record.get("score") or "Score unavailable"))}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_recent_form(result: Mapping[str, Any]) -> None:
    section_title(
        "Recent form",
        eyebrow="Latest five recorded matches",
        description="Match rows are oriented around each selected team and sorted newest first.",
    )
    records = result.get("recent_form", {})
    with st.container(key="comparison-recent-grid", border=False):
        column_a, column_b = st.columns(2)
        with column_a:
            with glass_card("comparison-recent-a"):
                _render_form_records(result["team_a"], list(records.get("team_a", [])))
        with column_b:
            with glass_card("comparison-recent-b"):
                _render_form_records(result["team_b"], list(records.get("team_b", [])))


def _render_head_to_head(result: Mapping[str, Any]) -> None:
    team_a = str(result["team_a"]["name"])
    team_b = str(result["team_b"]["name"])
    history = result.get("head_to_head", {})
    section_title(
        "Head-to-head history",
        eyebrow="Recorded historical meetings",
        description="Home and away source rows are normalized to Team A before the totals are calculated.",
    )
    with glass_card("comparison-head-to-head"):
        meetings = int(history.get("meetings", 0) or 0)
        if meetings == 0:
            st.markdown(
                '<p class="comparison-empty-text">No recorded head-to-head meetings were found in the available dataset.</p>',
                unsafe_allow_html=True,
            )
            return

        data_range = history.get("data_range", {})
        if data_range.get("start") and data_range.get("end"):
            st.markdown(
                f'<p class="comparison-h2h-summary">{meetings} recorded meeting(s) from {escape(str(data_range["start"]))} to {escape(str(data_range["end"]))}.</p>',
                unsafe_allow_html=True,
            )
        totals = st.columns(3)
        with totals[0]:
            metric_card(
                f"{team_a} wins",
                int(history.get("team_a_wins", 0)),
                caption="Recorded meetings",
                trend="positive",
            )
        with totals[1]:
            metric_card(
                "Draws",
                int(history.get("draws", 0)),
                caption="Recorded meetings",
                trend="neutral",
            )
        with totals[2]:
            metric_card(
                f"{team_b} wins",
                int(history.get("team_b_wins", 0)),
                caption="Recorded meetings",
                trend="positive",
            )

        for record in history.get("records", []):
            result_code = str(record.get("result") or "U")
            status = result_code if result_code in {"W", "D", "L"} else "U"
            details = [
                value
                for value in (
                    record.get("date"),
                    record.get("team_a_venue"),
                    record.get("tournament"),
                )
                if value
            ]
            st.markdown(
                f"""
                <div class="comparison-h2h-row">
                    <span class="comparison-form-status comparison-form-status--{status}">{escape(result_code if result_code != "U" else "—")}</span>
                    <div class="comparison-h2h-detail">
                        <strong>{escape(team_a)} · {escape(team_b)}</strong>
                        <span class="comparison-h2h-subdetail">{escape(" · ".join(str(value) for value in details))}</span>
                    </div>
                    <span class="comparison-h2h-score">{escape(str(record.get("score") or "Score unavailable"))}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_model_outlook(result: Mapping[str, Any]) -> None:
    team_a = str(result["team_a"]["name"])
    team_b = str(result["team_b"]["name"])
    section_title(
        "Model-based match outlook",
        eyebrow="Existing trained three-outcome model",
        description="This is kept distinct from the feature-based comparison and reuses the Match Predictor service.",
    )
    with glass_card("comparison-model-outlook"):
        outlook = result.get("model_outlook")
        if not outlook:
            message = result.get("model_outlook_error") or (
                "The trained predictor did not return a match outlook for this comparison."
            )
            st.info(str(message))
            return

        outcome = str(outlook.get("predicted_outcome"))
        predicted = {
            "home_win": f"{team_a} win",
            "draw": "Draw",
            "away_win": f"{team_b} win",
        }.get(outcome, "Outcome unavailable")
        st.markdown(
            f'<p class="comparison-outlook-result">Most likely outcome: <strong>{escape(predicted)}</strong>.</p>',
            unsafe_allow_html=True,
        )
        with st.container(key="comparison-model-probabilities", border=False):
            columns = st.columns(3)
            probability_cards = (
                (f"{team_a} win", outlook.get("team_a_win_probability"), "positive"),
                ("Draw", outlook.get("draw_probability"), "neutral"),
                (f"{team_b} win", outlook.get("team_b_win_probability"), "positive"),
            )
            for column, (label, probability, trend) in zip(columns, probability_cards):
                with column:
                    value = "Unavailable"
                    try:
                        value = f"{float(probability):.1%}"
                    except (TypeError, ValueError):
                        pass
                    metric_card(
                        label,
                        value,
                        caption="Trained model probability",
                        trend=trend,
                    )


def render_team_comparison_page() -> None:
    """Render Team Comparison only inside app.py's existing route branch."""
    _initialize_comparison_state()
    apply_theme()
    _render_comparison_styles()
    _render_header()

    try:
        teams = get_comparison_teams()
        team_a, team_b, clicked = _render_selectors(teams)
        signature = comparison_signature(team_a, team_b, get_snapshot_version())
    except (TeamComparisonError, ValueError) as error:
        st.error(str(error))
        return

    if st.session_state.comparison_signature != signature:
        st.session_state.comparison_error = None

    saved_result = st.session_state.comparison_result
    if clicked:
        if is_comparison_current(saved_result, signature):
            st.info(
                "This comparison already reflects the current teams and snapshot version."
            )
        else:
            _run_comparison(team_a, team_b, signature)
            saved_result = st.session_state.comparison_result

    if (
        st.session_state.comparison_error
        and st.session_state.comparison_signature == signature
    ):
        st.error(st.session_state.comparison_error)

    if saved_result and not is_comparison_current(saved_result, signature):
        st.info(
            "Team selection changed. Select Compare Teams to build a current comparison."
        )
        saved_result = None

    if not saved_result:
        return

    with animated_container("comparison-results", animation="fade_up"):
        _render_identity_header(saved_result)
        _render_summary(saved_result)
        _render_key_metrics(saved_result)
        _render_radar(saved_result)
        _render_strength_bars(saved_result)
        _render_recent_form(saved_result)
        _render_head_to_head(saved_result)
        _render_model_outlook(saved_result)
