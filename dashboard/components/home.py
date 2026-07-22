"""Render the TournamentIQ Home page from verified dashboard services.

Purpose:
    Present the product introduction, compact real match-prediction entry
    point, and platform coverage metrics on the default dashboard route.
Responsibility:
    Compose Home-only presentation while delegating trained inference and
    artifact reads to shared components and services.
Inputs:
    Streamlit session state, local image assets, match-prediction workspace,
    and artifact-backed platform statistics.
Outputs:
    The rendered Home interface and, after user action, canonical prediction
    results beneath the Home controls.
Collaboration:
    Reuses ``components.match_predictor`` for all prediction behavior and
    ``services.platform_statistics_service`` for non-fabricated metrics.
"""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

import streamlit as st

from components.match_predictor import (
    apply_match_prediction_styles,
    render_match_prediction_workspace,
)
from services.platform_statistics_service import load_platform_statistics
from ui import apply_theme, metric_card, render_html, section_title

DASHBOARD_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = DASHBOARD_ROOT / "assets" / "images"

FIFA_LOGO = IMAGE_ROOT / "fifa_logo.png"
HERO_BANNER = IMAGE_ROOT / "hero_banner.jpg"


def _format_integer(value: Any) -> str | None:
    """Format a verified integer value."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not isfinite(number) or number < 0:
        return None

    return f"{number:,.0f}"


def _render_home_styles() -> None:
    """Render styles used only by the Home page."""
    render_html("""
        <style>
            .st-key-home-hero {
                overflow: hidden;
                margin: 0.75rem 0 2.5rem;
                padding: clamp(0.7rem, 1.4vw, 1rem);
                border: 1px solid var(--ui-color-border-subtle);
                border-radius: 0.8rem;
                background:
                    radial-gradient(
                        circle at 85% 0%,
                        rgba(212, 102, 33, 0.19),
                        transparent 37%
                    ),
                    linear-gradient(
                        122deg,
                        rgba(25, 17, 13, 0.98),
                        rgba(16, 12, 9, 0.92)
                    );
                box-shadow: 0 1.4rem 3.6rem rgba(0, 0, 0, 0.28);
            }

            .st-key-home-hero [data-testid="stHorizontalBlock"] {
                align-items: stretch;
                gap: clamp(1rem, 2.3vw, 2.25rem);
            }

            .st-key-home-hero-copy {
                display: flex;
                min-height: 100%;
                flex-direction: column;
                justify-content: center;
                padding: clamp(1.8rem, 4vw, 3.75rem);
                border: 1px solid rgba(255, 229, 204, 0.09);
                border-radius: 0.55rem;
                background:
                    linear-gradient(
                        138deg,
                        rgba(48, 32, 24, 0.5),
                        rgba(25, 17, 13, 0.2)
                    );
                box-shadow:
                    inset 0 1px 0 rgba(255, 242, 230, 0.04);
            }

            .home-hero-eyebrow {
                margin: 0 0 0.8rem;
                color: var(--ui-color-accent-hover);
                font-family: var(--ui-type-font-mono);
                font-size: var(--ui-type-label);
                font-weight: var(--ui-type-weight-bold);
                letter-spacing: 0.14em;
                text-transform: uppercase;
            }

            .home-hero-title {
                max-width: 11ch;
                margin: 0;
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-display);
                font-size: clamp(2.6rem, 5vw, 5.25rem);
                font-weight: var(--ui-type-weight-black);
                letter-spacing: -0.06em;
                line-height: 0.92;
                text-wrap: balance;
            }

            .home-hero-subtitle {
                margin: 1.35rem 0 0;
                color: var(--ui-color-accent);
                font-size: clamp(0.74rem, 1.1vw, 0.9rem);
                font-weight: var(--ui-type-weight-bold);
                letter-spacing: 0.14em;
                line-height: 1.4;
            }

            .home-hero-description {
                margin: 0.95rem 0 0;
                color: var(--ui-color-text-secondary);
                font-size: clamp(0.92rem, 1.25vw, 1.08rem);
                line-height: var(--ui-type-line-height-body);
            }

            .st-key-home-hero-logo [data-testid="stImage"] {
                width: fit-content;
                margin-bottom: 1.15rem;
                padding: 0.5rem;
                border: 1px solid rgba(240, 185, 103, 0.2);
                border-radius: 0.45rem;
                background: rgba(36, 24, 19, 0.58);
            }

            .st-key-home-hero-image [data-testid="stImage"] {
                min-height: 21rem;
                overflow: hidden;
                border: 1px solid rgba(240, 185, 103, 0.17);
                border-radius: 0.55rem;
                box-shadow: 0 1.1rem 2.4rem rgba(0, 0, 0, 0.3);
            }

            .st-key-home-hero-image [data-testid="stImage"] img {
                width: 100%;
                height: clamp(21rem, 35vw, 32rem);
                object-fit: cover;
                filter:
                    saturate(0.94)
                    contrast(1.05)
                    brightness(0.8);
            }

            .st-key-home-platform-stats {
                margin-bottom: 2rem;
            }

            @media (max-width: 800px) {
                .st-key-home-hero [data-testid="stHorizontalBlock"],
                .st-key-home-platform-stats [data-testid="stHorizontalBlock"] {
                    flex-direction: column;
                }

                .st-key-home-hero [data-testid="stColumn"],
                .st-key-home-platform-stats [data-testid="stColumn"] {
                    width: 100% !important;
                    flex: 1 1 100% !important;
                }

                .st-key-home-hero-copy {
                    padding: 1.75rem 1.35rem;
                }
            }

            @media (max-width: 560px) {
                .st-key-home-hero {
                    padding: 0.65rem;
                    border-radius: 0.6rem;
                }

                .home-hero-title {
                    font-size: clamp(2.35rem, 14vw, 3.4rem);
                }

                .st-key-home-hero-image [data-testid="stImage"],
                .st-key-home-hero-image [data-testid="stImage"] img {
                    height: 16rem;
                    min-height: 16rem;
                }
            }
        </style>
        """)


def _render_hero() -> None:
    """Render the Home hero without forecast or simulation actions."""
    with st.container(key="home-hero", border=False):
        copy_column, image_column = st.columns((1.08, 0.92))

        with copy_column:
            with st.container(key="home-hero-copy", border=False):
                if FIFA_LOGO.exists():
                    with st.container(key="home-hero-logo", border=False):
                        st.image(str(FIFA_LOGO), width=72)

                render_html("""
                    <p class="home-hero-eyebrow">
                        Tournament intelligence / 2026
                    </p>

                    <h1 class="home-hero-title">
                        FIFA WORLD CUP 2026
                    </h1>

                    <p class="home-hero-subtitle">
                        AI FOOTBALL ANALYTICS PLATFORM
                    </p>

                    <p class="home-hero-description">
                        Predict &bull; Simulate &bull; Analyze &bull; Visualize
                    </p>
                    """)

        with image_column:
            if HERO_BANNER.exists():
                with st.container(key="home-hero-image", border=False):
                    st.image(
                        str(HERO_BANNER),
                        width="stretch",
                    )


def _load_verified_platform_metrics() -> list[tuple[str, str]]:
    """Return only real platform metrics that have valid values."""
    statistics = load_platform_statistics()

    metrics = [
        (
            "Total Historical Matches",
            _format_integer(statistics.get("total_historical_matches")),
        ),
        (
            "Engineered Features",
            _format_integer(statistics.get("engineered_features")),
        ),
        (
            "Competitions",
            _format_integer(statistics.get("competitions")),
        ),
        (
            "Countries Represented",
            _format_integer(statistics.get("countries_represented")),
        ),
    ]

    return [(label, value) for label, value in metrics if value is not None]


def _render_platform_overview() -> None:
    """Render verified platform statistics only."""
    metrics = _load_verified_platform_metrics()

    if not metrics:
        return

    section_title("Platform Overview")

    with st.container(key="home-platform-stats", border=False):
        for start in range(0, len(metrics), 3):
            row_metrics = metrics[start : start + 3]
            columns = st.columns(len(row_metrics))

            for column, (label, value) in zip(columns, row_metrics):
                with column:
                    metric_card(label, value)


def _render_home_match_prediction() -> None:
    """Render Home's compact entry point to the canonical match predictor.

    The shared workspace performs all validation, inference, state management,
    and result rendering.  Home supplies only page-specific widget keys and
    presentation text so it cannot diverge from the Match Predictor backend.
    """
    render_match_prediction_workspace(
        selector_key_prefix="home-match",
        button_key="home-match-predict-button",
        card_key="home-match-setup",
        setup_title="Predict a match",
        setup_eyebrow="Trained match model",
        setup_description=(
            "Choose two teams to run the same three-outcome prediction used by "
            "the Match Predictor. The trained model applies its fixed neutral "
            "World Cup context."
        ),
        button_label="Predict Match",
        show_history=False,
    )


def render_home_page() -> None:
    """Render the complete Home page."""
    apply_theme()
    apply_match_prediction_styles()
    _render_home_styles()
    _render_hero()
    _render_home_match_prediction()
    _render_platform_overview()


__all__ = ["render_home_page"]
