"""Apply the shared TournamentIQ application frame around routed pages.

Purpose:
    Establish the visual shell consistently before any page renderer executes.
Responsibility:
    Coordinate the global theme, decorative background, and frame-level CSS
    without configuring routing or embedding business functionality.
Inputs:
    The active Streamlit page and reusable UI/layout helpers.
Outputs:
    Shared presentation rules injected into the current dashboard page.
Collaboration:
    Invoked by ``app.py`` and delegates implementation details to ``ui`` and
    ``components.background``.
"""

from __future__ import annotations

import streamlit as st

from components.background import render_background
from ui import apply_theme


def _render_navigation_shell_styles() -> None:
    """Style native top navigation as a restrained premium application bar."""
    st.markdown(
        """
        <style>
            header[data-testid="stHeader"] {
                min-height: 4.15rem;
                border-bottom: 1px solid rgba(255, 211, 168, 0.14);
                background:
                    linear-gradient(
                        105deg,
                        rgba(18, 11, 8, 0.98),
                        rgba(34, 20, 14, 0.96) 60%,
                        rgba(46, 26, 17, 0.94)
                    ) !important;
                box-shadow: 0 0.65rem 1.8rem rgba(7, 4, 2, 0.26);
                backdrop-filter: blur(14px) saturate(118%);
                -webkit-backdrop-filter: blur(14px) saturate(118%);
            }

            [data-testid="stTopNavLinkContainer"] {
                display: flex !important;
                flex-direction: row !important;
                flex-wrap: nowrap !important;
                align-items: center !important;
                justify-content: flex-start !important;
                width: 100% !important;
                min-height: 4.15rem !important;
                gap: 0.28rem !important;
                padding: 0 1.25rem !important;
                margin: 0 !important;
                overflow-x: auto !important;
                overflow-y: hidden !important;
                scrollbar-width: none;
            }

            [data-testid="stTopNavLinkContainer"]::-webkit-scrollbar {
                display: none;
            }

            header[data-testid="stHeader"] [data-testid="stTopNavLinkContainer"]::before {
                content: "TournamentIQ";
                display: inline-flex !important;
                align-items: center;
                justify-content: center;
                flex: 0 0 auto !important;
                min-height: 2.35rem;
                margin-right: 1.2rem;
                padding: 0 0.3rem 0 0.7rem;
                border-left: 2px solid #d69747;
                color: #fff8f2;
                font-family:
                    Inter,
                    ui-sans-serif,
                    system-ui,
                    -apple-system,
                    BlinkMacSystemFont,
                    "Segoe UI",
                    sans-serif;
                font-size: 1.12rem;
                font-weight: 800;
                letter-spacing: -0.04em;
                line-height: 1;
                white-space: nowrap;
                text-shadow: 0 0 1rem rgba(214, 151, 71, 0.12);
            }

            /* Hide duplicated brand inside popovers/dropdowns so TournamentIQ appears once */
            [data-testid="stTopNavPopover"] [data-testid="stTopNavLinkContainer"]::before {
                display: none !important;
            }

            [data-testid="stTopNavLink"],
            [data-testid="stTopNavDropdownLink"] {
                display: inline-flex !important;
                flex: 0 0 auto !important;
                align-items: center !important;
                justify-content: center !important;
                min-height: 2.3rem;
                padding: 0.4rem 0.78rem !important;
                border: 1px solid transparent;
                border-radius: 0.45rem;
                color: #d7c7ba !important;
                font-weight: 650;
                white-space: nowrap !important;
                transition:
                    background-color 150ms ease,
                    border-color 150ms ease,
                    color 150ms ease,
                    box-shadow 150ms ease,
                    transform 150ms ease;
            }

            [data-testid="stTopNavLink"]:hover,
            [data-testid="stTopNavDropdownLink"]:hover {
                border-color: rgba(240, 185, 103, 0.32);
                background: rgba(185, 78, 19, 0.14);
                box-shadow: inset 0 1px 0 rgba(255, 238, 222, 0.05);
                color: #fff8f2 !important;
            }

            [data-testid="stTopNavLink"][aria-current="page"] {
                border-color: rgba(240, 185, 103, 0.46);
                background:
                    linear-gradient(
                        110deg,
                        rgba(185, 78, 19, 0.28),
                        rgba(214, 151, 71, 0.16)
                    );
                box-shadow:
                    inset 0 1px 0 rgba(255, 238, 222, 0.12),
                    0 0 1.1rem rgba(185, 78, 19, 0.16);
                color: #fff8f2 !important;
            }

            [data-testid="stTopNavPopover"] {
                border: 1px solid rgba(255, 211, 168, 0.18);
                border-radius: 0.6rem;
                background: rgba(29, 17, 12, 0.98);
                box-shadow: 0 1rem 2.4rem rgba(7, 4, 2, 0.36);
                backdrop-filter: blur(14px);
                -webkit-backdrop-filter: blur(14px);
            }

            [data-testid="stAppViewContainer"],
            .stApp {
                overflow-x: hidden;
            }

            [data-testid="stMainBlockContainer"] {
                padding-top: clamp(1.35rem, 2.3vw, 2.15rem) !important;
                padding-bottom: clamp(2rem, 4vw, 4rem) !important;
            }

            @media (max-width: 760px) {
                header[data-testid="stHeader"] {
                    min-height: 3.8rem;
                }

                [data-testid="stTopNavLinkContainer"] {
                    min-height: 3.8rem !important;
                    padding: 0 0.65rem !important;
                    gap: 0.2rem !important;
                }

                header[data-testid="stHeader"] [data-testid="stTopNavLinkContainer"]::before {
                    margin-right: 0.65rem;
                    font-size: 1rem;
                }

                [data-testid="stTopNavLink"],
                [data-testid="stTopNavDropdownLink"] {
                    min-height: 2.2rem;
                    padding: 0.38rem 0.55rem !important;
                    font-size: 0.86rem;
                }

                [data-testid="stMainBlockContainer"] {
                    padding-top: 0.9rem !important;
                }
            }

            @media (prefers-reduced-motion: reduce) {
                header[data-testid="stHeader"],
                [data-testid="stTopNavPopover"] {
                    backdrop-filter: none;
                    -webkit-backdrop-filter: none;
                }

                [data-testid="stTopNavLink"],
                [data-testid="stTopNavDropdownLink"] {
                    transition: none;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_global_layout() -> None:
    """Apply the shared theme, atmospheric background, and native navigation frame."""
    apply_theme()
    render_background()
    _render_navigation_shell_styles()


__all__ = ["apply_global_layout"]
