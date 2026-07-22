"""Resolve and render trusted local national-team flag assets.

Purpose:
    Keep flag lookup consistent across Home, Match Predictor, comparison, and
    tournament presentation without network requests or duplicated mappings.
Responsibility:
    Map canonical team names to local image/SVG assets and provide accessible
    fallback content when a flag is unavailable.
Inputs:
    Canonical national-team names and optional visual dimensions/fallback text.
Outputs:
    Existing local flag paths or Streamlit image/fallback elements.
Collaboration:
    Uses ``ui.media`` for SVG rendering and is called by team-facing dashboard
    components; it never changes team data or model selections.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from .media import render_svg_image


DASHBOARD_ROOT = Path(__file__).resolve().parents[1]
FLAG_ROOT = DASHBOARD_ROOT / "assets" / "flags"

COUNTRY_FLAG_CODES: dict[str, str] = {
    "Argentina": "ARG",
    "Australia": "AUS",
    "Austria": "AUT",
    "Belgium": "BEL",
    "Brazil": "BRA",
    "Canada": "CAN",
    "Chile": "CHI",
    "Colombia": "COL",
    "Croatia": "CRO",
    "Czech Republic": "CZE",
    "Denmark": "DEN",
    "Ecuador": "ECU",
    "Egypt": "EGY",
    "England": "ENG",
    "France": "FRA",
    "Germany": "GER",
    "Ghana": "GHA",
    "Iran": "IRN",
    "Japan": "JPN",
    "Mexico": "MEX",
    "Morocco": "MAR",
    "Netherlands": "NED",
    "Nigeria": "NGA",
    "Norway": "NOR",
    "Paraguay": "PAR",
    "Peru": "PER",
    "Poland": "POL",
    "Portugal": "POR",
    "Qatar": "QAT",
    "Senegal": "SEN",
    "Serbia": "SRB",
    "South Korea": "KOR",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "SUI",
    "Tunisia": "TUN",
    "Turkey": "TUR",
    "United States": "USA",
    "Uruguay": "URU",
}


def get_flag_path(team: str) -> Path | None:
    """Return the correct local asset for a canonical national-team name."""
    code = COUNTRY_FLAG_CODES.get(team)
    if not code:
        return None
    for suffix in (".svg", ".png"):
        path = FLAG_ROOT / f"{code}{suffix}"
        if path.exists():
            return path
    return None


def render_flag(
    team: str, *, width: int = 52, fallback: str = "Flag unavailable"
) -> None:
    """Render a local flag safely, including the dedicated England asset."""
    path = get_flag_path(team)
    if path is None:
        st.caption(fallback)
        return
    if path.suffix.casefold() == ".svg":
        if render_svg_image(
            path, width=width, label=f"{team} flag", css_class="ui-team-flag"
        ):
            return
        st.caption(fallback)
        return
    st.image(str(path), width=width)


__all__ = ["COUNTRY_FLAG_CODES", "FLAG_ROOT", "get_flag_path", "render_flag"]
