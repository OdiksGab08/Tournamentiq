"""Define restrained, accessible animation tokens for shared dashboard UI.

Purpose:
    Centralize the small set of supported motion names and their CSS behavior.
Responsibility:
    Validate animation requests and return CSS-safe class/declaration strings
    without rendering UI elements or adding browser-side state.
Inputs:
    Supported animation-name strings requested by reusable UI helpers.
Outputs:
    CSS keyframes, class names, or animation declarations for trusted styling.
Collaboration:
    ``ui.theme`` injects the keyframes and ``ui.components`` uses returned
    declarations for opt-in animated containers.
"""

from typing import Final


ANIMATION_NAMES: Final = frozenset({"fade_up", "fade_in", "pulse"})

ANIMATION_CSS: Final = """
@keyframes ui-fade-up {
    from {
        opacity: 0;
        transform: translateY(12px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes ui-fade-in {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes ui-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.72; }
}

.ui-animate-fade-up { animation: ui-fade-up 420ms ease-out both; }
.ui-animate-fade-in { animation: ui-fade-in 300ms ease-out both; }
.ui-animate-pulse { animation: ui-pulse 1.8s ease-in-out infinite; }

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: 0.01ms !important;
    }
}
"""


def animation_class(name: str) -> str:
    """Return the CSS class for a supported animation name."""
    normalized = name.strip().lower().replace("-", "_")
    if normalized not in ANIMATION_NAMES:
        supported = ", ".join(sorted(ANIMATION_NAMES))
        raise ValueError(f"Unsupported animation '{name}'. Choose one of: {supported}.")
    return f"ui-animate-{normalized.replace('_', '-')}"


def animation_css_value(name: str) -> str:
    """Return the complete CSS animation declaration for a motion token."""
    normalized = name.strip().lower().replace("-", "_")
    values = {
        "fade_up": "ui-fade-up 420ms ease-out both",
        "fade_in": "ui-fade-in 300ms ease-out both",
        "pulse": "ui-pulse 1.8s ease-in-out infinite",
    }
    if normalized not in values:
        supported = ", ".join(sorted(ANIMATION_NAMES))
        raise ValueError(f"Unsupported animation '{name}'. Choose one of: {supported}.")
    return values[normalized]
