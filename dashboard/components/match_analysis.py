"""Render supporting feature evidence for canonical match predictions.

Purpose:
    Present factual feature evidence and recent prediction records for a saved
    trained-model prediction.
Responsibility:
    Transform normalized prediction-result mappings into accessible Streamlit
    sections without calculating probabilities or altering model state.
Inputs:
    Result mappings produced by ``services.match_prediction_service`` and
    recent-result records stored in Streamlit session state.
Outputs:
    Evidence cards and recent prediction history elements.
Collaboration:
    Called by ``components.match_predictor`` after the shared inference flow
    completes; relies on ``ui`` primitives for presentation.
"""

from __future__ import annotations

from html import escape
from typing import Any, Mapping, Sequence

import streamlit as st

from ui import glass_card, section_title


FEATURE_SPECS = (
    ("recent_form", "Recent form", "points", True),
    ("attack_strength", "Attack strength", "decimal", True),
    ("defense_strength", "Defence strength", "decimal", False),
    ("goal_difference", "Goal difference", "decimal", True),
    ("competition_strength", "Competition strength", "decimal", True),
    ("world_cup_experience", "World Cup experience", "integer", True),
    ("clean_sheet_rate", "Clean-sheet rate", "percentage", True),
)


def _number(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric == numeric else None


def _format_value(value: Any, value_type: str) -> str:
    numeric = _number(value)
    if numeric is None:
        return "—"
    if value_type == "percentage":
        return f"{numeric:.1%}" if 0 <= numeric <= 1 else f"{numeric:.1f}%"
    if value_type == "decimal":
        return f"{numeric:.2f}"
    return f"{numeric:,.0f}"


def _relative_value_share(home_value: float, away_value: float) -> float:
    total = abs(home_value) + abs(away_value)
    if total == 0:
        return 50.0
    return min(max(abs(home_value) / total * 100, 0), 100)


def render_feature_evidence(result: Mapping[str, Any]) -> None:
    """Compare only interpretable fields from the exact feature vector used."""
    snapshot = result.get("feature_snapshot") or {}
    home_snapshot = snapshot.get("home") or {}
    away_snapshot = snapshot.get("away") or {}
    available = []
    for key, label, value_type, higher_is_better in FEATURE_SPECS:
        home_value = _number(home_snapshot.get(key))
        away_value = _number(away_snapshot.get(key))
        if home_value is not None and away_value is not None:
            available.append(
                (key, label, value_type, higher_is_better, home_value, away_value)
            )

    section_title(
        "Supporting Feature Evidence",
        eyebrow="Exact model inputs",
        description="These comparisons are taken from the feature row built for this match. They are contextual evidence, not per-match causal attribution.",
    )
    with glass_card("match-feature-evidence"):
        if not available:
            st.info(
                "The saved predictor returned outcome probabilities, but no per-match feature explanation was exposed by the current backend."
            )
            return

        st.markdown(
            f"""
            <div class="match-evidence-header">
                <span>{escape(str(result["home_team"]))}</span>
                <span>Recorded comparison</span>
                <span>{escape(str(result["away_team"]))}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for _, label, value_type, _, home_value, away_value in available:
            home_share = _relative_value_share(home_value, away_value)
            difference = home_value - away_value
            st.markdown(
                f"""
                <div class="match-evidence-row">
                    <span class="match-evidence-value">{_format_value(home_value, value_type)}</span>
                    <div class="match-evidence-center">
                        <span class="match-evidence-label">{escape(label)}</span>
                        <div class="match-evidence-bar" aria-label="Relative recorded values">
                            <span class="match-evidence-bar__home" style="width: {home_share:.2f}%"></span>
                            <span class="match-evidence-bar__away" style="width: {100 - home_share:.2f}%"></span>
                        </div>
                        <span class="match-evidence-difference">Home–away difference: {difference:+.2f}</span>
                    </div>
                    <span class="match-evidence-value match-evidence-value--away">{_format_value(away_value, value_type)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        head_to_head = snapshot.get("head_to_head") or {}
        h2h_matches = _number(head_to_head.get("h2h_matches"))
        home_rate = _number(head_to_head.get("home_h2h_win_rate"))
        away_rate = _number(head_to_head.get("away_h2h_win_rate"))
        if (
            h2h_matches is not None
            and h2h_matches > 0
            and home_rate is not None
            and away_rate is not None
        ):
            st.caption(
                "Head-to-head record in the exact feature row: "
                f"{int(h2h_matches)} matches, home-side win rate {_format_value(home_rate, 'percentage')}, "
                f"away-side win rate {_format_value(away_rate, 'percentage')}."
            )


def render_recent_predictions(history: Sequence[Mapping[str, Any]]) -> None:
    """Render the latest in-session results without persisting them to disk."""
    if not history:
        return

    section_title("Recent predictions")
    with glass_card("match-history"):
        for item in history[:5]:
            st.markdown(
                f"""
                <div class="match-history-row">
                    <strong>{escape(str(item["home_team"]))} vs {escape(str(item["away_team"]))}</strong>
                    <span>Most likely: {escape(str(item["outcome_label"]))}</span>
                    <span>{float(item["home_win_probability"]):.1%} / {float(item["draw_probability"]):.1%} / {float(item["away_win_probability"]):.1%}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
