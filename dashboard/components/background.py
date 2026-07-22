"""Render the shared decorative background for TournamentIQ pages.

Purpose:
    Supply a subtle, non-interactive visual backdrop for the dashboard shell.
Responsibility:
    Inject only trusted CSS needed for the background and avoid page-specific
    layout, navigation, or business logic.
Inputs:
    The active Streamlit page and the shared visual token values in this module.
Outputs:
    Background CSS emitted into the current Streamlit response.
Collaboration:
    Called by ``components.layout`` before view renderers add their content.
"""

import base64
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

BACKGROUND = ROOT / "assets" / "images" / "worldcup_logo.jpg"


def render_background() -> None:
    """Apply a dark editorial backdrop without turning content into a giant card.

    The image remains an optional local texture. A layered dark gradient keeps
    text and data readable even when the asset is unavailable.
    """
    encoded: str | None = None
    if BACKGROUND.exists():
        try:
            encoded = base64.b64encode(BACKGROUND.read_bytes()).decode("ascii")
        except OSError:
            # A missing or unreadable decorative image should never block pages.
            encoded = None

    image_layer = (
        f', url("data:image/jpeg;base64,{encoded}")' if encoded is not None else ""
    )
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image:
                    radial-gradient(circle at 8% -12%, rgba(185, 78, 19, 0.23), transparent 34%),
                    radial-gradient(circle at 92% 7%, rgba(214, 151, 71, 0.10), transparent 29%),
                    linear-gradient(132deg, rgba(16, 12, 9, 0.96), rgba(16, 10, 8, 0.985)){image_layer};
                background-position: center;
                background-size: cover;
                background-attachment: fixed;
            }}

            [data-testid="stAppViewContainer"] {{
                background: transparent;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
