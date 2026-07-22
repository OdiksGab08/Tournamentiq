"""Visualize normalized trained-model match probabilities for dashboard pages.

Purpose:
    Present the same real home-win, draw, and away-win probabilities as both a
    compact distribution and an accessible Plotly chart.
Responsibility:
    Validate presentation-bound probability values and create visual output;
    it never performs feature construction or model inference.
Inputs:
    Normalized result mappings from the shared match-prediction service.
Outputs:
    Streamlit HTML and Plotly probability visualizations.
Collaboration:
    Used by ``components.match_predictor`` alongside result and evidence
    renderers after the canonical prediction workflow completes.
"""

from __future__ import annotations

from html import escape
from typing import Any, Mapping

import plotly.graph_objects as go
import streamlit as st


def _probability_rows(result: Mapping[str, Any]) -> list[tuple[str, float, str]]:
    """Read normalized probabilities from the dashboard service result."""
    home_team = str(result["home_team"])
    away_team = str(result["away_team"])
    rows = [
        (f"{home_team} win", float(result["home_win_probability"]), "#B94E13"),
        ("Draw", float(result["draw_probability"]), "#D69747"),
        (f"{away_team} win", float(result["away_win_probability"]), "#57B38F"),
    ]
    if any(probability < 0 or probability > 1 for _, probability, _ in rows):
        raise ValueError("Probability values must be normalized between 0 and 1.")
    return rows


def render_probability_chart(result: Mapping[str, Any]) -> None:
    """Render an accessible, responsive Plotly chart from real model probabilities."""
    rows = _probability_rows(result)
    labels, probabilities, colors = zip(*rows)
    figure = go.Figure(
        go.Bar(
            x=probabilities,
            y=labels,
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            text=[f"{probability:.1%}" for probability in probabilities],
            textposition="auto",
            textfont={"color": "#FFF8F2", "size": 13},
            hovertemplate="%{y}<br>Model probability: %{x:.2%}<extra></extra>",
        )
    )
    figure.update_layout(
        height=280,
        margin={"l": 12, "r": 12, "t": 16, "b": 12},
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font={"color": "#FFF8F2", "family": "Manrope, Inter, sans-serif"},
        showlegend=False,
    )
    figure.update_xaxes(
        range=[0, 1],
        tickformat=".0%",
        gridcolor="rgba(255, 218, 184, 0.14)",
        zeroline=False,
    )
    figure.update_yaxes(autorange="reversed", gridcolor="rgba(0, 0, 0, 0)")
    with st.container(key="match-outcome-chart", border=False):
        st.plotly_chart(
            figure,
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )


def render_probability_distribution(result: Mapping[str, Any]) -> None:
    """Render a segmented bar and text legend from the same real probabilities."""
    rows = _probability_rows(result)
    segments = "".join(
        (
            f'<span class="match-distribution__segment match-distribution__segment--{index}" '
            f'style="flex-grow: {probability:.8f}"></span>'
        )
        for index, (_, probability, _) in enumerate(rows)
    )
    legend = "".join(
        (
            f'<span class="match-distribution__legend-item">{escape(label)} <strong>{probability:.1%}</strong></span>'
        )
        for label, probability, _ in rows
    )
    st.markdown(
        f"""
        <div class="match-distribution" role="img" aria-label="Home win, draw, and away win probability distribution">
            <div class="match-distribution__bar">{segments}</div>
            <div class="match-distribution__legend">{legend}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
