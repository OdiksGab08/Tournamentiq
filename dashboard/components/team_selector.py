"""Provide reusable team-selection and local-flag helpers for match views.

Purpose:
    Offer consistent, accessible Home and Match Predictor team controls backed
    by verified available-team names.
Responsibility:
    Keep team selectors distinct, initialize page-specific widget state, and
    render local flag assets with safe fallbacks.
Inputs:
    Ordered team-name sequences, widget-key prefixes, and local flag assets.
Outputs:
    Selected home/away team names and corresponding Streamlit flag elements.
Collaboration:
    Used by ``components.match_predictor``; delegates flag resolution to
    ``ui.flags`` and performs no model or data-service work.
"""

from __future__ import annotations

from typing import Sequence

import streamlit as st

from ui.flags import get_flag_path, render_flag


def render_team_flag(
    team: str,
    *,
    width: int = 52,
    fallback: str = "Flag unavailable",
) -> None:
    """Render a local flag or a clean accessible fallback when unavailable."""
    render_flag(team, width=width, fallback=fallback)


def render_team_selector(
    teams: Sequence[str], *, key_prefix: str = "match"
) -> tuple[str, str]:
    """Render accessible selectors that never offer the same team twice.

    Args:
        teams: Verified team names available to the trained prediction backend.
        key_prefix: A page-specific Streamlit widget-key prefix.  The default
            preserves the Match Predictor page's existing state keys, while
            other views can retain independent input selections.

    Returns:
        The selected home and away team names.

    Raises:
        ValueError: If fewer than two teams are available for a matchup.
    """
    options = list(teams)
    if len(options) < 2:
        raise ValueError("At least two teams are required for match prediction.")

    home_key = f"{key_prefix}_home_team"
    away_key = f"{key_prefix}_away_team"

    if st.session_state.get(home_key) not in options:
        st.session_state[home_key] = options[0]

    home_column, versus_column, away_column = st.columns((1, 0.3, 1))
    with home_column:
        st.markdown('<p class="match-team-label">Home team</p>', unsafe_allow_html=True)
        home_team = st.selectbox(
            "Home team",
            options,
            key=home_key,
            label_visibility="collapsed",
        )
        with st.container(key=f"{key_prefix}-home-flag", border=False):
            render_team_flag(home_team)

    away_options = [team for team in options if team.casefold() != home_team.casefold()]
    if st.session_state.get(away_key) not in away_options:
        st.session_state[away_key] = away_options[0]

    with versus_column:
        st.markdown(
            '<div class="match-versus" aria-label="versus">VS</div>',
            unsafe_allow_html=True,
        )

    with away_column:
        st.markdown('<p class="match-team-label">Away team</p>', unsafe_allow_html=True)
        away_team = st.selectbox(
            "Away team",
            away_options,
            key=away_key,
            label_visibility="collapsed",
        )
        with st.container(key=f"{key_prefix}-away-flag", border=False):
            render_team_flag(away_team)

    return home_team, away_team


__all__ = ["get_flag_path", "render_team_flag", "render_team_selector"]
