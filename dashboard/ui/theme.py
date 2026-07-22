"""Define and apply the shared TournamentIQ visual theme.

Purpose:
    Assemble color, typography, spacing, radius, shadow, glass, and animation
    tokens into one reusable stylesheet for dashboard presentation.
Responsibility:
    Hold immutable theme configuration and emit CSS only when a page or UI
    primitive explicitly requests it.
Inputs:
    Optional immutable ``UITheme`` instances and the active Streamlit page.
Outputs:
    CSS variable declarations and injected theme stylesheet markup.
Collaboration:
    Combines ``ui.colors``, ``ui.typography``, and ``ui.animations`` for use by
    ``ui.components`` and global layout helpers.
"""

from dataclasses import asdict, dataclass
from typing import Final

import streamlit as st

from .animations import ANIMATION_CSS
from .colors import COLORS, ColorPalette, color_css_variables
from .typography import TYPE, Typography, typography_css_variables


@dataclass(frozen=True, slots=True)
class SpacingScale:
    """Store immutable spacing tokens used by layout-oriented UI primitives.

    Attributes:
        xxs: Extra-extra-small spacing value.
        xs: Extra-small spacing value.
        sm: Small spacing value.
        md: Default spacing value.
        lg: Large spacing value.
        xl: Extra-large spacing value.
        xxl: Extra-extra-large spacing value.
    """

    xxs: str = "0.25rem"
    xs: str = "0.45rem"
    sm: str = "0.7rem"
    md: str = "1rem"
    lg: str = "1.35rem"
    xl: str = "2.15rem"
    xxl: str = "3.35rem"


@dataclass(frozen=True, slots=True)
class RadiusScale:
    """Store immutable border-radius tokens for dashboard surfaces and controls.

    Attributes:
        sm: Small corner radius.
        md: Default corner radius.
        lg: Large corner radius.
        xl: Extra-large corner radius.
        pill: Fully rounded radius for badges and pill-shaped controls.
    """

    sm: str = "0.375rem"
    md: str = "0.625rem"
    lg: str = "0.875rem"
    xl: str = "1.125rem"
    pill: str = "999px"


@dataclass(frozen=True, slots=True)
class ShadowScale:
    """Store immutable elevation and glow shadow tokens for the shared theme.

    Attributes:
        sm: Low-elevation shadow.
        md: Standard card-elevation shadow.
        lg: High-elevation shadow.
        glow: Accent glow used by primary interactive elements.
    """

    sm: str = "0 0.7rem 1.65rem rgba(7, 4, 2, 0.24)"
    md: str = "0 1.1rem 2.75rem rgba(7, 4, 2, 0.32)"
    lg: str = "0 1.8rem 4.5rem rgba(7, 4, 2, 0.4)"
    glow: str = "0 0.8rem 2.1rem rgba(185, 78, 19, 0.18)"


@dataclass(frozen=True, slots=True)
class GlassTokens:
    """Store immutable translucent-surface tokens for limited glass treatments.

    Attributes:
        background: Standard translucent surface fill.
        background_strong: Higher-contrast translucent surface fill.
        border: Subtle glass-surface border color.
        blur: Backdrop blur strength.
        saturation: Backdrop saturation multiplier.
    """

    background: str = "rgba(29, 18, 14, 0.78)"
    background_strong: str = "rgba(23, 14, 11, 0.92)"
    border: str = "rgba(255, 218, 184, 0.14)"
    blur: str = "12px"
    saturation: str = "118%"


@dataclass(frozen=True, slots=True)
class UITheme:
    """Aggregate immutable token groups consumed by shared theme generation.

    Attributes:
        colors: Semantic color palette.
        typography: Font and type-scale tokens.
        spacing: Layout spacing tokens.
        radius: Border-radius tokens.
        shadows: Elevation and glow-shadow tokens.
        glass: Translucent-surface tokens.
    """

    colors: ColorPalette = COLORS
    typography: Typography = TYPE
    spacing: SpacingScale = SpacingScale()
    radius: RadiusScale = RadiusScale()
    shadows: ShadowScale = ShadowScale()
    glass: GlassTokens = GlassTokens()


THEME: Final = UITheme()


def _token_variables(theme: UITheme) -> str:
    spacing = "\n".join(
        f"--ui-space-{name.replace('_', '-')}: {value};"
        for name, value in asdict(theme.spacing).items()
    )
    radius = "\n".join(
        f"--ui-radius-{name.replace('_', '-')}: {value};"
        for name, value in asdict(theme.radius).items()
    )
    shadows = "\n".join(
        f"--ui-shadow-{name.replace('_', '-')}: {value};"
        for name, value in asdict(theme.shadows).items()
    )
    glass = "\n".join(
        f"--ui-glass-{name.replace('_', '-')}: {value};"
        for name, value in asdict(theme.glass).items()
    )
    return "\n".join(
        (
            color_css_variables(theme.colors),
            typography_css_variables(theme.typography),
            spacing,
            radius,
            shadows,
            glass,
        )
    )


def get_theme_css(theme: UITheme = THEME) -> str:
    """Build CSS used by the reusable UI components.

    The CSS is opt-in: it only takes effect after a page calls ``apply_theme`` or
    uses one of the helpers in ``ui.components``.
    """
    return f"""
    :root {{
        {_token_variables(theme)}
    }}

    html,
    body,
    [data-testid="stAppViewContainer"] {{
        color: var(--ui-color-text-primary);
        font-family: var(--ui-type-font-sans);
    }}

    [data-testid="stMainBlockContainer"] {{
        max-width: 96rem;
    }}

    .ui-page-header {{
        display: flex;
        flex-wrap: wrap;
        align-items: end;
        justify-content: space-between;
        gap: var(--ui-space-lg);
        margin: 0 0 var(--ui-space-xl);
        padding-bottom: var(--ui-space-lg);
        border-bottom: 1px solid var(--ui-color-border-subtle);
        font-family: var(--ui-type-font-display);
    }}

    .ui-page-header__eyebrow,
    .ui-section-title__eyebrow {{
        margin: 0 0 var(--ui-space-xs);
        color: var(--ui-color-accent);
        font-family: var(--ui-type-font-mono);
        font-size: var(--ui-type-label);
        font-weight: var(--ui-type-weight-bold);
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }}

    .ui-page-header__title {{
        margin: 0;
        color: var(--ui-color-text-primary);
        font-size: var(--ui-type-h1);
        font-weight: var(--ui-type-weight-black);
        letter-spacing: -0.055em;
        line-height: var(--ui-type-line-height-tight);
    }}

    .ui-page-header__subtitle {{
        max-width: 44rem;
        margin: var(--ui-space-sm) 0 0;
        color: var(--ui-color-text-secondary);
        font-size: var(--ui-type-body);
        line-height: var(--ui-type-line-height-body);
    }}

    .ui-section-title {{
        margin: var(--ui-space-xxl) 0 var(--ui-space-md);
        font-family: var(--ui-type-font-display);
    }}

    .ui-section-title--compact {{
        margin-top: var(--ui-space-lg);
    }}

    .ui-section-title__title {{
        margin: 0;
        color: var(--ui-color-text-primary);
        font-size: var(--ui-type-h2);
        font-weight: var(--ui-type-weight-bold);
        letter-spacing: -0.025em;
        line-height: var(--ui-type-line-height-tight);
    }}

    .ui-section-title__description {{
        margin: var(--ui-space-xs) 0 0;
        color: var(--ui-color-text-secondary);
        font-size: var(--ui-type-body);
        line-height: var(--ui-type-line-height-body);
    }}

    .ui-metric-card {{
        min-height: 9.25rem;
        padding: var(--ui-space-lg);
        border: 1px solid var(--ui-glass-border);
        border-radius: var(--ui-radius-lg);
        background:
            linear-gradient(145deg, rgba(47, 30, 23, 0.88), rgba(24, 15, 12, 0.92));
        box-shadow: inset 0 1px 0 rgba(255, 239, 225, 0.045), var(--ui-shadow-sm);
        font-family: var(--ui-type-font-sans);
        backdrop-filter: blur(var(--ui-glass-blur)) saturate(var(--ui-glass-saturation));
        -webkit-backdrop-filter: blur(var(--ui-glass-blur)) saturate(var(--ui-glass-saturation));
        transition: border-color 180ms ease, transform 180ms ease, box-shadow 180ms ease;
    }}

    .ui-metric-card:hover {{
        border-color: rgba(240, 185, 103, 0.32);
        box-shadow: inset 0 1px 0 rgba(255, 239, 225, 0.06), var(--ui-shadow-md);
        transform: translateY(-2px);
    }}

    .ui-metric-card__topline {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: var(--ui-space-sm);
    }}

    .ui-metric-card__label {{
        color: var(--ui-color-text-secondary);
        font-size: var(--ui-type-small);
        font-weight: var(--ui-type-weight-semibold);
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }}

    .ui-metric-card__icon {{
        display: grid;
        width: 2rem;
        height: 2rem;
        place-items: center;
        border: 1px solid var(--ui-glass-border);
        border-radius: var(--ui-radius-md);
        background: rgba(185, 78, 19, 0.16);
        color: var(--ui-color-accent-hover);
    }}

    .ui-metric-card__value {{
        margin-top: var(--ui-space-lg);
        color: var(--ui-color-text-primary);
        font-size: clamp(1.85rem, 3vw, 2.7rem);
        font-weight: var(--ui-type-weight-black);
        letter-spacing: -0.045em;
        line-height: 1;
    }}

    .ui-metric-card__footer {{
        display: flex;
        flex-wrap: wrap;
        gap: var(--ui-space-sm);
        margin-top: var(--ui-space-sm);
        color: var(--ui-color-text-muted);
        font-size: var(--ui-type-small);
    }}

    .ui-metric-card__delta--positive {{ color: var(--ui-color-success); }}
    .ui-metric-card__delta--negative {{ color: var(--ui-color-danger); }}
    .ui-metric-card__delta--neutral {{ color: var(--ui-color-text-secondary); }}

    [class*="st-key-ui-glass-"] {{
        padding: var(--ui-space-lg);
        border: 1px solid var(--ui-glass-border);
        border-radius: var(--ui-radius-lg);
        background:
            linear-gradient(145deg, var(--ui-glass-background), rgba(19, 12, 10, 0.88));
        box-shadow: inset 0 1px 0 rgba(255, 239, 225, 0.045), var(--ui-shadow-md);
        backdrop-filter: blur(var(--ui-glass-blur)) saturate(var(--ui-glass-saturation));
        -webkit-backdrop-filter: blur(var(--ui-glass-blur)) saturate(var(--ui-glass-saturation));
    }}

    [class*="st-key-ui-animated-"] {{
        animation: ui-fade-up 420ms ease-out both;
    }}

    [class*="st-key-ui-gradient-button-"] {{
        margin: 0;
    }}

    [class*="st-key-ui-gradient-button-"] button {{
        min-height: 2.9rem;
        border: 1px solid rgba(255, 203, 155, 0.34);
        border-radius: var(--ui-radius-md);
        background: linear-gradient(110deg, var(--ui-color-primary), #C85E1B 58%, #D58A36);
        box-shadow: var(--ui-shadow-glow), inset 0 1px 0 rgba(255, 238, 222, 0.2);
        color: #ffffff;
        font-family: var(--ui-type-font-sans);
        font-weight: var(--ui-type-weight-bold);
        letter-spacing: 0.015em;
        transition: background-color 160ms ease, filter 160ms ease, transform 160ms ease, box-shadow 160ms ease;
    }}

    [class*="st-key-ui-gradient-button-"] button:hover:not(:disabled) {{
        filter: brightness(1.06) saturate(1.05);
        transform: translateY(-1px);
        box-shadow: 0 1rem 2.4rem rgba(185, 78, 19, 0.28), inset 0 1px 0 rgba(255, 238, 222, 0.28);
    }}

    [class*="st-key-ui-gradient-button-"] button:focus-visible,
    button:focus-visible,
    [data-testid="stTopNavLink"]:focus-visible,
    [data-testid="stTopNavDropdownLink"]:focus-visible {{
        outline: 2px solid var(--ui-color-accent-hover) !important;
        outline-offset: 2px;
    }}

    [data-baseweb="select"] > div,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input,
    [data-testid="stTextArea"] textarea {{
        border-color: var(--ui-color-border) !important;
        border-radius: var(--ui-radius-sm) !important;
        background: rgba(20, 13, 10, 0.72) !important;
        color: var(--ui-color-text-primary) !important;
    }}

    [data-baseweb="select"] > div:hover,
    [data-testid="stTextInput"] input:hover,
    [data-testid="stNumberInput"] input:hover,
    [data-testid="stDateInput"] input:hover,
    [data-testid="stTextArea"] textarea:hover {{
        border-color: rgba(240, 185, 103, 0.42) !important;
    }}

    [data-testid="stExpander"] {{
        border: 1px solid var(--ui-color-border-subtle);
        border-radius: var(--ui-radius-md);
        background: rgba(25, 16, 12, 0.56);
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid var(--ui-color-border-subtle);
        border-radius: var(--ui-radius-md);
        overflow: hidden;
    }}

    @media (max-width: 720px) {{
        .ui-page-header {{
            align-items: start;
            margin-bottom: var(--ui-space-lg);
        }}

        .ui-metric-card {{
            min-height: 8.5rem;
            padding: var(--ui-space-md);
        }}

        [class*="st-key-ui-glass-"] {{
            padding: var(--ui-space-md);
            border-radius: var(--ui-radius-md);
        }}

        .ui-section-title {{
            margin-top: var(--ui-space-xl);
        }}
    }}

    @media (prefers-reduced-motion: reduce) {{
        .ui-metric-card,
        [class*="st-key-ui-gradient-button-"] button {{
            transition: none;
        }}

        .ui-metric-card:hover,
        [class*="st-key-ui-gradient-button-"] button:hover:not(:disabled) {{
            transform: none;
        }}
    }}

    {ANIMATION_CSS}
    """


def apply_theme(theme: UITheme = THEME) -> None:
    """Inject the shared theme stylesheet into the current Streamlit page."""
    st.markdown(f"<style>{get_theme_css(theme)}</style>", unsafe_allow_html=True)
