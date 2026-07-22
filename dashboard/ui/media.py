"""Safely load and render trusted local SVG assets in the Streamlit dashboard.

Purpose:
    Make bundled SVG images and icons available to dashboard components without
    exposing raw XML injection or letting decorative asset failures break pages.
Responsibility:
    Validate local SVG markup, create data URIs, and delegate escaped HTML
    output to the shared rendering primitive.
Inputs:
    Local SVG paths, optional colors, dimensions, labels, and CSS class names.
Outputs:
    Safe SVG markup/data URIs or rendered image/icon elements; unavailable
    assets return safe empty values rather than throwing presentation failures.
Collaboration:
    Called by ``ui.flags`` and page components, and delegates HTML emission to
    ``ui.components.render_html``.
"""

from __future__ import annotations

from base64 import b64encode
from html import escape
from pathlib import Path
import re

from .components import render_html


_SAFE_CLASS = re.compile(r"[^a-zA-Z0-9_-]+")


def load_svg_markup(path: Path, *, color: str | None = None) -> str | None:
    """Load a trusted SVG document from disk.

    Args:
        path: Location of the SVG asset.
        color: Optional replacement for ``currentColor`` tokens in the SVG.

    Returns:
        The stripped SVG markup when the file is readable and begins with an
        ``<svg`` element; otherwise, ``None``.

    Notes:
        The structural check keeps accidental non-SVG files from being injected
        into page markup.
    """
    if not path.exists() or path.suffix.casefold() != ".svg":
        return None
    try:
        markup = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not markup.casefold().startswith("<svg"):
        return None
    return markup.replace("currentColor", color) if color else markup


def svg_data_uri(path: Path, *, color: str | None = None) -> str | None:
    """Encode a trusted SVG as a browser-safe data URI.

    Args:
        path: Location of the SVG asset.
        color: Optional replacement for ``currentColor`` tokens in the SVG.

    Returns:
        A base64-encoded SVG data URI, or ``None`` when the asset is invalid or
        unavailable.

    Notes:
        Encoding prevents raw XML from being emitted as page text.
    """
    markup = load_svg_markup(path, color=color)
    if markup is None:
        return None
    encoded = b64encode(markup.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def render_svg_image(
    path: Path,
    *,
    width: int,
    label: str,
    color: str | None = None,
    css_class: str | None = None,
) -> bool:
    """Render a trusted SVG in an escaped image element.

    Args:
        path: Location of the SVG asset.
        width: Rendered width and height in pixels.
        label: Accessible text for the image.
        color: Optional replacement for ``currentColor`` tokens in the SVG.
        css_class: Optional sanitized CSS class for the image element.

    Returns:
        ``True`` when an image was rendered; ``False`` when the asset could not
        be loaded.

    Notes:
        Attribute values are escaped and CSS classes are sanitized before
        rendering through the shared HTML helper.
    """
    uri = svg_data_uri(path, color=color)
    if uri is None:
        return False
    class_name = (
        f' class="{escape(_SAFE_CLASS.sub("-", css_class).strip("-") or "ui-svg-image")}"'
        if css_class
        else ""
    )
    alt = escape(label)
    render_html(
        f'<img{class_name} src="{uri}" alt="{alt}" '
        f'style="display:block;width:{int(width)}px;height:{int(width)}px;object-fit:contain" />',
        width="content",
    )
    return True


def render_svg_icon(
    path: Path,
    *,
    size: int = 18,
    label: str = "",
    color: str = "#8ee8ff",
    wrapper_class: str | None = None,
) -> bool:
    """Render a compact trusted SVG icon without injecting raw XML.

    Args:
        path: Location of the SVG asset.
        size: Rendered width and height in pixels.
        label: Optional accessible label for the icon.
        color: Replacement for ``currentColor`` tokens in the SVG.
        wrapper_class: Optional sanitized CSS class for the wrapping element.

    Returns:
        ``True`` when an icon was rendered; ``False`` when the asset could not
        be loaded.

    Notes:
        Unlabelled icons are marked decorative for assistive technologies.
    """
    uri = svg_data_uri(path, color=color)
    if uri is None:
        return False
    class_name = (
        f' class="{escape(_SAFE_CLASS.sub("-", wrapper_class).strip("-") or "ui-svg-icon")}"'
        if wrapper_class
        else ""
    )
    accessible_label = escape(label) if label else ""
    aria_hidden = "true" if not label else "false"
    render_html(
        f'<span{class_name} aria-hidden="{aria_hidden}" aria-label="{accessible_label}">'
        f'<img src="{uri}" alt="{accessible_label}" '
        f'style="display:block;width:{int(size)}px;height:{int(size)}px;object-fit:contain" />'
        "</span>",
        width="content",
    )
    return True


__all__ = ["load_svg_markup", "render_svg_icon", "render_svg_image", "svg_data_uri"]
