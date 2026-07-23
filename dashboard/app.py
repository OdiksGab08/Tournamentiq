"""Run the TournamentIQ Streamlit application and native page router.

Purpose:
    Provide the single executable entry point for the production dashboard.
Responsibility:
    Configure Streamlit once, apply shared layout treatment, and dispatch the
    registered native-navigation page without duplicating page logic.
Inputs:
    Streamlit runtime state and the route declarations from ``navigation``.
Outputs:
    A configured dashboard page rendered by the selected view module.
Collaboration:
    Delegates layout work to ``components.layout`` and route construction to
    ``navigation``; individual views own all page-specific presentation.
"""

from __future__ import annotations

try:  # Supports both ``streamlit run dashboard/app.py`` and module imports.
    from .bootstrap import ensure_project_root
except ImportError:  # Streamlit executes this entry point as a script.
    from bootstrap import ensure_project_root

ensure_project_root()

import streamlit as st

from components.layout import apply_global_layout
from navigation import (
    NavigationError,
    build_page_groups,
    ensure_native_navigation_supported,
)


st.set_page_config(
    page_title="TournamentIQ | Football Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Remove only the obsolete navigation-state key.
# Page-specific prediction and filter state remains untouched.
st.session_state.pop("page", None)

apply_global_layout()

try:
    ensure_native_navigation_supported()

    current_page = st.navigation(
        build_page_groups(),
        position="top",
    )
except NavigationError as error:
    st.error(f"Navigation could not be initialized: {error}")
    st.stop()

current_page.run()
