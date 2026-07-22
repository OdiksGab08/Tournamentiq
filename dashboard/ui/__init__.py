"""Expose the reusable visual primitives used throughout TournamentIQ.

Purpose:
    Offer a concise import surface for shared theme, layout, media, and flag
    helpers without coupling pages to individual implementation modules.
Responsibility:
    Re-export retained UI primitives only; it contains no page logic, data
    access, or application state.
Inputs:
    Imports from dashboard components and views that need shared presentation.
Outputs:
    Importable UI helpers, theme tokens, and trusted media-rendering utilities.
Collaboration:
    Used by all dashboard components; implementation remains split across
    ``ui.components``, ``ui.theme``, ``ui.flags``, and ``ui.media``.
"""

from .components import (
    animated_container,
    glass_card,
    gradient_button,
    metric_card,
    page_header,
    render_html,
    section_title,
)
from .flags import get_flag_path, render_flag
from .media import load_svg_markup, render_svg_icon, render_svg_image, svg_data_uri
from .theme import THEME, apply_theme, get_theme_css

__all__ = [
    "THEME",
    "animated_container",
    "apply_theme",
    "get_theme_css",
    "glass_card",
    "get_flag_path",
    "gradient_button",
    "load_svg_markup",
    "metric_card",
    "page_header",
    "render_flag",
    "render_html",
    "render_svg_icon",
    "render_svg_image",
    "section_title",
    "svg_data_uri",
]
