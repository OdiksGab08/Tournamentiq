"""Define typography tokens for consistent TournamentIQ information hierarchy.

Purpose:
    Centralize readable font stacks and responsive type-scale values for all
    dashboard pages and reusable UI primitives.
Responsibility:
    Store immutable typography settings and translate them into CSS variables;
    it does not render text or decide page-specific hierarchy.
Inputs:
    Optional ``Typography`` token instances supplied by the shared theme.
Outputs:
    Immutable typography records and CSS custom-property declarations.
Collaboration:
    ``ui.theme`` combines these values with color and layout tokens; components
    consume the variables through the shared stylesheet.
"""

from dataclasses import asdict, dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class Typography:
    """Font stacks and a compact, data-first type scale."""

    font_sans: str = (
        "Manrope, Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "
        '"Segoe UI", sans-serif'
    )
    font_display: str = (
        '"DM Sans", Manrope, Inter, ui-sans-serif, system-ui, -apple-system, '
        'BlinkMacSystemFont, "Segoe UI", sans-serif'
    )
    font_mono: str = '"IBM Plex Mono", "SFMono-Regular", Consolas, monospace'
    display: str = "clamp(2.25rem, 4.2vw, 4.75rem)"
    h1: str = "clamp(1.9rem, 3vw, 3rem)"
    h2: str = "clamp(1.3rem, 1.9vw, 1.8rem)"
    h3: str = "1.08rem"
    body: str = "0.95rem"
    small: str = "0.78rem"
    label: str = "0.68rem"
    line_height_tight: str = "1.08"
    line_height_body: str = "1.6"
    weight_regular: str = "400"
    weight_medium: str = "500"
    weight_semibold: str = "600"
    weight_bold: str = "700"
    weight_black: str = "800"


TYPE: Final = Typography()


def typography_css_variables(typography: Typography = TYPE) -> str:
    """Return CSS custom properties for the shared type scale."""
    return "\n".join(
        f"--ui-type-{name.replace('_', '-')}: {value};"
        for name, value in asdict(typography).items()
    )
