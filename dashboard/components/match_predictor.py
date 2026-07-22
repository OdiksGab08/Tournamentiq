"""Provide the shared Streamlit workspace for real match-outcome predictions.

Purpose:
    Render and coordinate the Home and Match Predictor interfaces that invoke
    the existing trained three-outcome match model.
Responsibility:
    Manage per-session prediction state, validate user selections through the
    service adapter, and render canonical results without duplicating ML logic.
Inputs:
    Team selections, Streamlit session state, and normalized results from
    ``services.match_prediction_service``.
Outputs:
    Prediction controls, running/error feedback, probability views, evidence,
    and optional in-session history rendered in the active page.
Collaboration:
    Reuses team selectors, analysis/chart components, and the cached service
    adapter; Home supplies alternate widget keys through the public workspace.
"""

from __future__ import annotations

from html import escape
from typing import Any, Mapping

import streamlit as st

from components.match_analysis import (
    render_feature_evidence,
    render_recent_predictions,
)
from components.match_probability_chart import (
    render_probability_chart,
    render_probability_distribution,
)
from components.team_selector import render_team_flag, render_team_selector
from services.match_prediction_service import (
    MatchPredictionError,
    get_available_teams,
    is_result_current,
    matchup_signature,
    predict_match,
)
from ui import (
    apply_theme,
    glass_card,
    gradient_button,
    page_header,
    section_title,
)


def initialize_match_prediction_state() -> None:
    """Initialize shared per-session state for trained match predictions.

    The Home and Match Predictor pages intentionally share completed results
    and history, while their selector widget keys remain page-specific.  This
    keeps one source of truth for a real model run without storing state on
    the module or re-running inference during ordinary Streamlit reruns.
    """
    defaults = {
        "match_prediction_result": None,
        "match_prediction_error": None,
        "match_prediction_running": False,
        "match_prediction_signature": None,
        "match_prediction_history": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _outcome_label(result: Mapping[str, Any]) -> str:
    outcome = result.get("predicted_outcome")
    if outcome == "home_win":
        return f"{result['home_team']} win"
    if outcome == "away_win":
        return f"{result['away_team']} win"
    return "Draw"


def _add_to_history(result: Mapping[str, Any]) -> None:
    entry = {
        "home_team": result["home_team"],
        "away_team": result["away_team"],
        "outcome_label": _outcome_label(result),
        "home_win_probability": result["home_win_probability"],
        "draw_probability": result["draw_probability"],
        "away_win_probability": result["away_win_probability"],
        "signature": result["signature"],
    }
    existing = st.session_state.match_prediction_history
    filtered = [
        item for item in existing if item.get("signature") != entry["signature"]
    ]
    st.session_state.match_prediction_history = [entry, *filtered][:5]


def run_match_prediction(home_team: str, away_team: str, signature: str) -> None:
    """Call the real backend once and retain its normalized result in session state."""
    st.session_state.match_prediction_running = True
    st.session_state.match_prediction_error = None

    try:
        with st.status(
            "Generating trained-model probabilities…", expanded=True
        ) as status:
            st.write("Building the existing match feature row.")
            result = predict_match(home_team, away_team)
            result["signature"] = signature
            st.session_state.match_prediction_result = result
            st.session_state.match_prediction_signature = signature
            _add_to_history(result)
            status.update(label="Match prediction complete", state="complete")
    except MatchPredictionError as error:
        st.session_state.match_prediction_error = str(error)
    except Exception:
        st.session_state.match_prediction_error = (
            "The trained match predictor could not complete this request. "
            "Please confirm the model and source datasets are available."
        )
    finally:
        st.session_state.match_prediction_running = False


def apply_match_prediction_styles() -> None:
    """Apply shared visual treatments for match-prediction controls and results."""
    st.markdown(
        """
        <style>
            @keyframes match-gradient-flow {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            .match-setup-note {
                margin: 0.8rem 0 1.25rem;
                color: var(--ui-color-text-secondary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.88rem;
                line-height: 1.5;
            }

            .match-team-label {
                margin: 0 0 0.45rem;
                color: var(--ui-color-accent);
                font-family: var(--ui-type-font-mono);
                font-size: 0.7rem;
                font-weight: 800;
                letter-spacing: 0.13em;
                text-transform: uppercase;
            }

            .match-versus {
                display: grid;
                width: 3rem;
                height: 3rem;
                place-items: center;
                margin: 2.3rem auto 0;
                border: 1px solid rgba(214, 151, 71, 0.3);
                border-radius: var(--ui-radius-md);
                background: linear-gradient(135deg, rgba(185, 78, 19, 0.18), rgba(214, 151, 71, 0.13));
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-mono);
                font-size: 0.82rem;
                font-weight: 800;
                letter-spacing: 0.08em;
            }

            .st-key-match-home-flag [data-testid="stImage"],
            .st-key-match-away-flag [data-testid="stImage"],
            .st-key-home-match-home-flag [data-testid="stImage"],
            .st-key-home-match-away-flag [data-testid="stImage"] {
                width: fit-content;
                margin-top: 0.65rem;
                padding: 0.34rem;
                border: 1px solid var(--ui-color-border-subtle);
                border-radius: var(--ui-radius-sm);
                background: rgba(25, 17, 13, 0.5);
            }

            .match-result-team {
                margin: 0.35rem 0 0;
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: clamp(1.25rem, 2.3vw, 2rem);
                font-weight: 800;
                letter-spacing: -0.04em;
                line-height: 1.05;
            }

            .match-result-side-label,
            .match-result-kicker {
                margin: 0;
                color: var(--ui-color-accent);
                font-family: var(--ui-type-font-mono);
                font-size: 0.68rem;
                font-weight: 800;
                letter-spacing: 0.15em;
                text-transform: uppercase;
            }

            .match-result-outcome {
                margin: 0.55rem 0 0;
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: clamp(1.3rem, 2.5vw, 2rem);
                font-weight: 800;
                letter-spacing: -0.045em;
                line-height: 1.05;
                text-align: center;
            }

            [class*="st-key-match-probability-"] {
                position: relative;
                min-height: 9.5rem;
                padding: 1.15rem;
                overflow: hidden;
                border: 1px solid var(--ui-color-border-subtle);
                border-radius: var(--ui-radius-md);
                background: var(--ui-glass-background);
                box-shadow: inset 0 1px 0 rgba(255, 248, 242, 0.045);
                backdrop-filter: blur(10px) saturate(110%);
                -webkit-backdrop-filter: blur(10px) saturate(110%);
            }

            [class*="st-key-match-probability-"]::before {
                content: "";
                position: absolute;
                top: 0;
                right: 0;
                left: 0;
                height: 3px;
                background: linear-gradient(90deg, var(--ui-color-primary), var(--ui-color-accent), var(--ui-color-primary));
                background-size: 200% 100%;
                animation: match-gradient-flow 4.2s linear infinite;
            }

            [class*="st-key-match-probability-"][class*="leading"] {
                border-color: rgba(214, 151, 71, 0.58);
                box-shadow: var(--ui-shadow-sm), inset 0 1px 0 rgba(255, 248, 242, 0.07);
            }

            .match-probability-label {
                color: var(--ui-color-text-secondary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.06em;
                text-transform: uppercase;
            }

            .match-probability-value {
                margin-top: 1.25rem;
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: clamp(2rem, 4vw, 3rem);
                font-weight: 800;
                letter-spacing: -0.06em;
                line-height: 1;
            }

            .match-probability-state {
                margin-top: 0.55rem;
                color: var(--ui-color-accent);
                font-family: var(--ui-type-font-sans);
                font-size: 0.78rem;
            }

            .match-distribution {
                padding: 1rem 0.1rem 0.25rem;
            }

            .match-distribution__bar {
                display: flex;
                width: 100%;
                min-height: 1rem;
                overflow: hidden;
                border: 1px solid var(--ui-color-border-subtle);
                border-radius: 999px;
                background: var(--ui-color-surface-muted);
            }

            .match-distribution__segment--0 { background: var(--ui-color-primary); }
            .match-distribution__segment--1 { background: var(--ui-color-accent); }
            .match-distribution__segment--2 { background: var(--ui-color-success); }

            .match-distribution__legend {
                display: flex;
                flex-wrap: wrap;
                gap: 0.75rem 1.1rem;
                margin-top: 0.7rem;
                color: var(--ui-color-text-secondary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.82rem;
            }

            .match-distribution__legend-item strong {
                color: var(--ui-color-text-primary);
            }

            .st-key-match-outcome-chart [data-testid="stPlotlyChart"] {
                border: 1px solid var(--ui-color-border-subtle);
                border-radius: var(--ui-radius-md);
                background: rgba(25, 17, 13, 0.34);
            }

            .match-evidence-header,
            .match-evidence-row {
                display: grid;
                grid-template-columns: minmax(4rem, 0.75fr) minmax(11rem, 2.6fr) minmax(4rem, 0.75fr);
                align-items: center;
                gap: 1rem;
            }

            .match-evidence-header {
                margin: 0 0 0.8rem;
                color: var(--ui-color-text-muted);
                font-family: var(--ui-type-font-mono);
                font-size: 0.67rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-align: center;
                text-transform: uppercase;
            }

            .match-evidence-row {
                padding: 0.85rem 0;
                border-top: 1px solid var(--ui-color-border-subtle);
            }

            .match-evidence-value {
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-mono);
                font-size: 0.92rem;
                font-weight: 700;
            }

            .match-evidence-value--away { text-align: right; }

            .match-evidence-center { min-width: 0; }

            .match-evidence-label {
                display: block;
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.83rem;
                font-weight: 700;
                text-align: center;
            }

            .match-evidence-bar {
                display: flex;
                height: 0.42rem;
                margin: 0.45rem 0;
                overflow: hidden;
                border-radius: 999px;
                background: var(--ui-color-surface-muted);
            }

            .match-evidence-bar__home { background: var(--ui-color-primary); }
            .match-evidence-bar__away { background: var(--ui-color-success); }

            .match-evidence-difference {
                display: block;
                color: var(--ui-color-text-muted);
                font-family: var(--ui-type-font-sans);
                font-size: 0.7rem;
                text-align: center;
            }

            .match-history-row {
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                justify-content: space-between;
                gap: 0.55rem 1rem;
                padding: 0.8rem 0;
                border-top: 1px solid var(--ui-color-border-subtle);
                color: var(--ui-color-text-secondary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.82rem;
            }

            .match-history-row:first-child { border-top: 0; }
            .match-history-row strong { color: var(--ui-color-text-primary); }

            @media (max-width: 760px) {
                .st-key-ui-glass-match-setup [data-testid="stHorizontalBlock"],
                .st-key-ui-glass-match-result-hero [data-testid="stHorizontalBlock"],
                .st-key-match-outcome-grid [data-testid="stHorizontalBlock"] {
                    flex-direction: column;
                }

                .st-key-ui-glass-match-setup [data-testid="stColumn"],
                .st-key-ui-glass-match-result-hero [data-testid="stColumn"],
                .st-key-match-outcome-grid [data-testid="stColumn"] {
                    width: 100% !important;
                    flex: 1 1 100% !important;
                }

                .match-versus {
                    margin: 0.65rem auto;
                }

                .match-result-outcome { text-align: left; }

                .match-evidence-header { display: none; }

                .match-evidence-row {
                    grid-template-columns: 1fr;
                    gap: 0.35rem;
                }

                .match-evidence-value,
                .match-evidence-value--away { text-align: left; }

                .match-evidence-label,
                .match-evidence-difference { text-align: left; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    page_header(
        "Match Predictor",
        eyebrow="FIFA World Cup 2026 model",
        subtitle="Compare two national teams using the trained three-outcome World Cup match model.",
    )


def _render_match_setup(
    teams: list[str],
    *,
    selector_key_prefix: str,
    button_key: str,
    card_key: str,
    title: str,
    eyebrow: str,
    description: str,
    button_label: str,
    compact_heading: bool,
) -> tuple[str, str, bool]:
    """Render one configurable, backend-compatible match-selection form."""
    section_title(
        title,
        eyebrow=eyebrow,
        description=description,
        compact=compact_heading,
    )
    with glass_card(card_key):
        home_team, away_team = render_team_selector(
            teams, key_prefix=selector_key_prefix
        )
        st.markdown(
            """
            <p class="match-setup-note">
                Venue and competition-stage controls are intentionally omitted because the current backend fixes those inputs internally.
            </p>
            """,
            unsafe_allow_html=True,
        )
        clicked = gradient_button(
            button_label,
            key=button_key,
            width="content",
            disabled=st.session_state.match_prediction_running,
        )
    return home_team, away_team, clicked


def _render_result_hero(result: Mapping[str, Any]) -> None:
    section_title("Matchup result", eyebrow="Prediction result")
    with glass_card("match-result-hero"):
        home_column, outcome_column, away_column = st.columns((1, 1.25, 1))
        with home_column:
            render_team_flag(str(result["home_team"]), width=76)
            st.markdown(
                '<p class="match-result-side-label">Home team</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<h2 class="match-result-team">{escape(str(result["home_team"]))}</h2>',
                unsafe_allow_html=True,
            )
        with outcome_column:
            st.markdown(
                '<p class="match-result-kicker">Most likely outcome</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="match-result-outcome">{escape(_outcome_label(result))}</div>',
                unsafe_allow_html=True,
            )
        with away_column:
            render_team_flag(str(result["away_team"]), width=76)
            st.markdown(
                '<p class="match-result-side-label">Away team</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<h2 class="match-result-team">{escape(str(result["away_team"]))}</h2>',
                unsafe_allow_html=True,
            )


def _render_probability_cards(result: Mapping[str, Any]) -> None:
    section_title("Outcome probabilities", eyebrow="Real model output")
    cards = (
        (
            "home_win",
            f"{result['home_team']} win",
            float(result["home_win_probability"]),
        ),
        ("draw", "Draw", float(result["draw_probability"])),
        (
            "away_win",
            f"{result['away_team']} win",
            float(result["away_win_probability"]),
        ),
    )
    with st.container(key="match-outcome-grid", border=False):
        columns = st.columns(3)
        for column, (outcome, label, probability) in zip(columns, cards):
            leading = (
                "leading" if result["predicted_outcome"] == outcome else "standard"
            )
            with column:
                with st.container(
                    key=f"match-probability-{outcome}-{leading}",
                    border=False,
                ):
                    state = (
                        "Highest model probability"
                        if leading == "leading"
                        else "Model probability"
                    )
                    st.markdown(
                        f"""
                        <div class="match-probability-label">{escape(label)}</div>
                        <div class="match-probability-value">{probability:.1%}</div>
                        <div class="match-probability-state">{state}</div>
                        """,
                        unsafe_allow_html=True,
                    )

    with glass_card("match-probability-visualization"):
        render_probability_distribution(result)
        render_probability_chart(result)


def render_match_prediction_results(result: Mapping[str, Any]) -> None:
    """Render the canonical trained-model result and its supporting evidence.

    Args:
        result: A normalized result returned by
            :func:`services.match_prediction_service.predict_match` and stored
            by :func:`run_match_prediction`.
    Returns:
        ``None``. The supplied real result is rendered in the active Streamlit
        page.
    """
    _render_result_hero(result)
    _render_probability_cards(result)
    render_feature_evidence(result)


def _current_match_prediction(signature: str) -> Mapping[str, Any] | None:
    """Report any stored service error and return only a current result."""
    if st.session_state.match_prediction_error:
        st.error(st.session_state.match_prediction_error)

    saved_result = st.session_state.match_prediction_result
    if saved_result and not is_result_current(saved_result, signature):
        st.info(
            "Team selection changed. Select Predict Match to generate a result for this matchup."
        )
        return None
    return saved_result


def render_match_prediction_workspace(
    *,
    selector_key_prefix: str = "match",
    button_key: str = "match-predict-button",
    card_key: str = "match-setup",
    setup_title: str = "Set up a matchup",
    setup_eyebrow: str = "Match configuration",
    setup_description: str = (
        "Select two teams from the processed snapshot dataset. The existing "
        "feature builder applies its trained neutral World Cup context."
    ),
    button_label: str = "Predict Match",
    show_history: bool = True,
    compact_setup_heading: bool = False,
) -> None:
    """Run the complete canonical match-prediction workspace on any page.

    Args:
        selector_key_prefix: Page-specific widget-key prefix for team inputs.
        button_key: Unique Streamlit key for the submit control.
        card_key: Presentation key for the setup card.
        setup_title: Visible selection-section title.
        setup_eyebrow: Visible selection-section eyebrow text.
        setup_description: Factual description of the fixed model context.
        button_label: Visible prediction-control label.
        show_history: Whether to render shared in-session prediction history.
        compact_setup_heading: Whether to reduce the setup-heading gap after a
            page header.

    Returns:
        ``None``. Validation errors, real-model status, and current results
        render in-place below the setup controls.

    Notes:
        This is intentionally the single UI orchestration path for Home and
        Match Predictor. It uses the existing prediction service and never
        substitutes a simulation or placeholder result.
    """
    initialize_match_prediction_state()

    try:
        teams = get_available_teams()
    except MatchPredictionError as error:
        st.error(str(error))
        return

    try:
        home_team, away_team, clicked = _render_match_setup(
            teams,
            selector_key_prefix=selector_key_prefix,
            button_key=button_key,
            card_key=card_key,
            title=setup_title,
            eyebrow=setup_eyebrow,
            description=setup_description,
            button_label=button_label,
            compact_heading=compact_setup_heading,
        )
        signature = matchup_signature(home_team, away_team)
    except (MatchPredictionError, ValueError) as error:
        st.error(str(error))
        return

    if clicked:
        run_match_prediction(home_team, away_team, signature)

    saved_result = _current_match_prediction(signature)
    if saved_result:
        render_match_prediction_results(saved_result)

    if show_history:
        render_recent_predictions(st.session_state.match_prediction_history)


def render_match_predictor_page() -> None:
    """Render the Match Predictor only inside app.py's existing route branch."""
    apply_theme()
    apply_match_prediction_styles()
    _render_header()

    render_match_prediction_workspace(compact_setup_heading=True)
