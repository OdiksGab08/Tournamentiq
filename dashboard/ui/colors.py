"""Define warm, high-contrast semantic color tokens for TournamentIQ.

Purpose:
    Keep dashboard colors consistent across surfaces, text, controls, charts,
    and feedback states.
Responsibility:
    Store immutable palette values and translate them into CSS custom
    properties without applying CSS directly.
Inputs:
    Optional ``ColorPalette`` instances supplied by the shared theme.
Outputs:
    Immutable palette records and CSS variable declarations.
Collaboration:
    ``ui.theme`` combines these tokens with typography, spacing, and glass
    settings; components consume the resulting CSS variables.
"""

from dataclasses import asdict, dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class ColorPalette:
    """Semantic palette shared by dashboard surfaces, controls, and charts."""

    canvas: str = "#100C09"
    surface: str = "#19110D"
    surface_elevated: str = "#241813"
    surface_muted: str = "#302018"
    primary: str = "#B94E13"
    primary_hover: str = "#D46621"
    accent: str = "#D69747"
    accent_hover: str = "#F0B967"
    success: str = "#57B38F"
    warning: str = "#E3A13B"
    danger: str = "#D96557"
    text_primary: str = "#FFF8F2"
    text_secondary: str = "#D8C7BA"
    text_muted: str = "#9B887B"
    border: str = "#4E362A"
    border_subtle: str = "rgba(255, 218, 184, 0.14)"


COLORS: Final = ColorPalette()


def color_css_variables(palette: ColorPalette = COLORS) -> str:
    """Return CSS custom properties for the configured color palette."""
    return "\n".join(
        f"--ui-color-{name.replace('_', '-')}: {value};"
        for name, value in asdict(palette).items()
    )
