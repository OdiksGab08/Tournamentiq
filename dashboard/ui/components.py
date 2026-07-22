"""Provide reusable, opt-in Streamlit presentation building blocks.

Purpose:
    Give dashboard pages consistent cards, headings, buttons, animation, and
    trusted HTML rendering without duplicating theme-aware Streamlit markup.
Responsibility:
    Render visual primitives only; no helper loads data, invokes models, or
    decides page navigation.
Inputs:
    Trusted display text, Streamlit container context, component keys, and
    optional presentation parameters.
Outputs:
    Styled Streamlit elements, context-managed containers, and button events.
Collaboration:
    Imports tokens from ``ui.theme`` and is used by all dashboard component
    renderers as their shared presentation layer.
"""

from contextlib import contextmanager
from html import escape
import re
from textwrap import dedent
from typing import Iterator, Literal

import streamlit as st

from .animations import animation_css_value
from .theme import apply_theme


MetricTrend = Literal["positive", "negative", "neutral"]


def _scope_key(prefix: str, key: str) -> str:
    """Create predictable, CSS-safe Streamlit container keys."""
    normalized = re.sub(r"[^a-z0-9-]+", "-", key.strip().lower())
    normalized = normalized.strip("-") or "component"
    return f"{prefix}-{normalized}"


def _html(value: object | None) -> str:
    return "" if value is None else escape(str(value))


def render_html(
    markup: str, *, width: Literal["stretch", "content"] = "stretch"
) -> None:
    """Render one complete trusted HTML fragment without Markdown code-block parsing."""
    cleaned = dedent(markup).strip()
    if not cleaned:
        return
    if callable(getattr(st, "html", None)):
        st.html(cleaned, width=width)
    else:  # Compatibility for older Streamlit runtimes.
        st.markdown(cleaned, unsafe_allow_html=True)


def metric_card(
    label: str,
    value: str | int | float,
    *,
    delta: str | None = None,
    caption: str | None = None,
    icon: str | None = None,
    trend: MetricTrend = "neutral",
) -> None:
    """Render a responsive metric card inside the current Streamlit container."""
    if trend not in {"positive", "negative", "neutral"}:
        raise ValueError("trend must be 'positive', 'negative', or 'neutral'.")

    apply_theme()
    icon_markup = (
        f'<span class="ui-metric-card__icon" aria-hidden="true">{_html(icon)}</span>'
        if icon
        else ""
    )
    delta_markup = (
        f'<span class="ui-metric-card__delta--{trend}">{_html(delta)}</span>'
        if delta
        else ""
    )
    caption_markup = f"<span>{_html(caption)}</span>" if caption else ""

    render_html(
        '<div class="ui-metric-card">'
        '<div class="ui-metric-card__topline">'
        f'<span class="ui-metric-card__label">{_html(label)}</span>'
        f"{icon_markup}"
        "</div>"
        f'<div class="ui-metric-card__value">{_html(value)}</div>'
        f'<div class="ui-metric-card__footer">{delta_markup}{caption_markup}</div>'
        "</div>"
    )


@contextmanager
def glass_card(key: str) -> Iterator[None]:
    """Provide a glassmorphism container for arbitrary Streamlit content.

    Example:
        with glass_card("model-summary"):
            st.write("Reusable content")
    """
    apply_theme()
    with st.container(key=_scope_key("ui-glass", key), border=False):
        yield


def page_header(
    title: str,
    *,
    subtitle: str | None = None,
    eyebrow: str | None = None,
) -> None:
    """Render a page-level heading with optional explanatory text."""
    apply_theme()
    eyebrow_markup = (
        f'<p class="ui-page-header__eyebrow">{_html(eyebrow)}</p>' if eyebrow else ""
    )
    subtitle_markup = (
        f'<p class="ui-page-header__subtitle">{_html(subtitle)}</p>' if subtitle else ""
    )
    render_html(
        '<header class="ui-page-header"><div>'
        f"{eyebrow_markup}"
        f'<h1 class="ui-page-header__title">{_html(title)}</h1>'
        f"{subtitle_markup}"
        "</div></header>"
    )


def section_title(
    title: str,
    *,
    description: str | None = None,
    eyebrow: str | None = None,
    compact: bool = False,
) -> None:
    """Render a consistent section title without changing page layout.

    Args:
        title: Visible section heading.
        description: Optional supporting description shown under the heading.
        eyebrow: Optional small category label shown above the heading.
        compact: Whether to reduce the top gap when the section immediately
            follows a page header.
    """
    apply_theme()
    eyebrow_markup = (
        f'<p class="ui-section-title__eyebrow">{_html(eyebrow)}</p>' if eyebrow else ""
    )
    description_markup = (
        f'<p class="ui-section-title__description">{_html(description)}</p>'
        if description
        else ""
    )
    class_name = (
        "ui-section-title ui-section-title--compact" if compact else "ui-section-title"
    )
    render_html(
        f'<section class="{class_name}">'
        f"{eyebrow_markup}"
        f'<h2 class="ui-section-title__title">{_html(title)}</h2>'
        f"{description_markup}"
        "</section>"
    )


def gradient_button(
    label: str,
    *,
    key: str,
    icon: str | None = None,
    help: str | None = None,
    disabled: bool = False,
    width: Literal["stretch", "content"] = "stretch",
) -> bool:
    """Render a native Streamlit button with shared gradient styling."""
    apply_theme()
    scoped_key = _scope_key("ui-gradient-button", key)
    button_label = f"{icon} {label}" if icon else label
    with st.container(key=scoped_key, border=False):
        return st.button(
            button_label,
            key=key,
            help=help,
            disabled=disabled,
            width=width,
        )


@contextmanager
def animated_container(
    key: str,
    *,
    animation: str = "fade_up",
) -> Iterator[None]:
    """Provide an accessible animated container for future page sections."""
    apply_theme()
    scoped_key = _scope_key("ui-animated", key)
    animation_value = animation_css_value(animation)
    st.markdown(
        f"<style>.st-key-{scoped_key} {{ animation: {animation_value}; }}</style>",
        unsafe_allow_html=True,
    )
    with st.container(key=scoped_key, border=False):
        yield
