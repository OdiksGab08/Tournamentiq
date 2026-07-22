"""Render saved model-evaluation artifacts on the Model Insights route.

Purpose:
    Give users a transparent, read-only view of retained model comparison and
    optional held-out diagnostics.
Responsibility:
    Compose UI from normalized service artifacts without fitting, replacing,
    or mutating any trained model or evaluation split.
Inputs:
    Streamlit controls and safe records supplied by
    ``services.model_insights_service``.
Outputs:
    Model comparison, calibration, feature-importance, and diagnostic sections
    rendered only when their source artifacts are available.
Collaboration:
    Acts as the presentation layer for the model-insights service and is called
    by its registered view module.
"""

from __future__ import annotations

from html import escape
from math import isfinite
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.model_insights_service import (
    ModelInsightsError,
    dataframe_csv,
    get_model_insights_overview,
)
from ui import (
    animated_container,
    apply_theme,
    glass_card,
    metric_card,
    page_header,
    render_svg_icon,
    section_title,
)


DASHBOARD_ROOT = Path(__file__).resolve().parents[1]
MODEL_INSIGHTS_ICON = DASHBOARD_ROOT / "assets" / "icons" / "brain.svg"

_CLASS_COLORS = {
    "Home Win": "#B94E13",
    "Draw": "#D69747",
    "Away Win": "#57B38F",
}
_MODEL_COLOR = "#E3A13B"
_METRIC_COLORS = {
    "Accuracy": "#B94E13",
    "Precision": "#D69747",
    "Recall": "#57B38F",
    "F1": "#E3A13B",
}


def _mapping_value(
    mapping: Mapping[str, Any] | None,
    *names: str,
    default: Any = None,
) -> Any:
    """Return the first present mapping value across compatible service keys."""
    if not isinstance(mapping, Mapping):
        return default
    normalized = {
        str(key).casefold().replace("_", " "): value for key, value in mapping.items()
    }
    for name in names:
        if name in mapping:
            return mapping[name]
        value = normalized.get(name.casefold().replace("_", " "))
        if value is not None:
            return value
    return default


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Return a mapping-shaped value without coercing unknown data."""
    return value if isinstance(value, Mapping) else {}


def _artifact_available(value: Any) -> bool:
    """Interpret a service availability value without treating ``'unavailable'`` as true."""
    if isinstance(value, Mapping):
        value = _mapping_value(
            value, "available", "is_available", "status", default=False
        )
    if isinstance(value, str):
        return value.strip().casefold() in {
            "available",
            "present",
            "yes",
            "true",
            "ready",
        }
    return bool(value)


def _safe_key(value: Any) -> str:
    """Create a deterministic Streamlit-safe key from a service label."""
    return re.sub(r"[^a-z0-9-]+", "-", str(value).casefold()).strip("-") or "item"


def _format_number(value: Any, *, decimals: int = 0) -> str:
    """Format a finite numeric value without inventing a fallback number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    if not isfinite(number):
        return "Unavailable"
    return f"{number:,.{decimals}f}"


def _format_percent(value: Any, *, decimals: int = 1) -> str:
    """Format a normalized probability/metric supplied by the service."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    if not isfinite(number):
        return "Unavailable"
    return f"{number:.{decimals}%}"


def _format_duration(value: Any) -> str:
    """Format a measured seconds value for concise metadata display."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    if not isfinite(seconds) or seconds < 0:
        return "Unavailable"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)} min {remainder:.0f} s"


def _format_file_size(value: Any) -> str:
    """Format a genuine artifact size when one is available."""
    try:
        size = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    if not isfinite(size) or size < 0:
        return "Unavailable"
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{size:.0f} {unit}"
        size /= 1024
    return "Unavailable"


def _coerce_frame(value: Any) -> pd.DataFrame:
    """Convert already-normalized tabular service output into a display frame."""
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, Mapping):
        if "records" in value:
            return _coerce_frame(value["records"])
        try:
            return pd.DataFrame(value)
        except (TypeError, ValueError):
            return pd.DataFrame()
    if isinstance(value, (list, tuple)):
        try:
            return pd.DataFrame(value)
        except (TypeError, ValueError):
            return pd.DataFrame()
    return pd.DataFrame()


def _column(frame: pd.DataFrame, *aliases: str) -> str | None:
    """Find a normalized column name without changing a source table."""
    lookup = {
        re.sub(r"[^a-z0-9]+", "", str(column).casefold()): str(column)
        for column in frame.columns
    }
    for alias in aliases:
        match = lookup.get(re.sub(r"[^a-z0-9]+", "", alias.casefold()))
        if match:
            return match
    return None


def _first_frame(mapping: Mapping[str, Any], *names: str) -> pd.DataFrame:
    """Read the first non-empty compatible table from a service payload."""
    for name in names:
        frame = _coerce_frame(_mapping_value(mapping, name))
        if not frame.empty:
            return frame
    return pd.DataFrame()


def _class_rows(value: Any) -> list[tuple[str, str]]:
    """Normalize verified class mapping variants into UI-safe label rows."""
    if isinstance(value, Mapping):
        rows = [(str(key), str(label)) for key, label in value.items()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = []
        for item in value:
            if isinstance(item, Mapping):
                raw_class = _mapping_value(item, "class", "class_id", "value")
                label = _mapping_value(item, "label", "outcome", "name")
                if raw_class is not None and label is not None:
                    rows.append((str(raw_class), str(label)))
            elif isinstance(item, Sequence) and len(item) >= 2:
                rows.append((str(item[0]), str(item[1])))
    else:
        rows = []

    def sort_key(row: tuple[str, str]) -> tuple[int, str]:
        try:
            return int(row[0]), row[1]
        except ValueError:
            return 999, row[0]

    return sorted(rows, key=sort_key)


def _class_labels(
    overview: Mapping[str, Any], diagnostics: Mapping[str, Any] | None = None
) -> list[str]:
    """Return verified display labels without replacing an unknown mapping."""
    for source in (diagnostics or {}, overview):
        mapping = _mapping_value(source, "class_mapping", "classes", "class_labels")
        labels = [label for _, label in _class_rows(mapping)]
        if labels:
            return labels
    return []


def _selected_model(overview: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read the selected production-model record from the overview payload."""
    candidate = _mapping_value(
        overview,
        "selected_model",
        "production_model",
        "model",
    )
    if isinstance(candidate, Mapping):
        return candidate
    return {}


def _selected_model_name(overview: Mapping[str, Any]) -> str:
    selected = _selected_model(overview)
    value = _mapping_value(selected, "name", "model_name", "label")
    if value is None:
        value = _mapping_value(overview, "selected_model_name", "model_name")
    return str(value).strip() if value is not None else "Selected model unavailable"


def _comparison_frame(overview: Mapping[str, Any]) -> pd.DataFrame:
    """Obtain the saved, normalized validation comparison table."""
    return _first_frame(
        overview,
        "comparison",
        "comparison_table",
        "model_comparison",
        "metrics_table",
    )


def _comparison_metric(frame: pd.DataFrame, name: str) -> pd.Series | None:
    """Return a safe numeric comparison metric only when the source provides it."""
    aliases = {
        "accuracy": ("accuracy", "validation_accuracy"),
        "precision": ("precision", "weighted_precision"),
        "recall": ("recall", "weighted_recall"),
        "f1": ("f1", "f1_score", "weighted_f1"),
        "log_loss": ("log_loss", "log loss", "logloss"),
        "training_time": ("training_time", "training time", "fit_time"),
    }
    column = _column(frame, *aliases[name])
    if not column:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    return values if values.notna().any() else None


def _plotly_layout(*, height: int = 360, **extra: Any) -> dict[str, Any]:
    """Return a consistent transparent Plotly layout."""
    layout: dict[str, Any] = {
        "height": height,
        "margin": {"l": 52, "r": 28, "t": 46, "b": 48},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#FFF8F2", "family": "Manrope, Inter, system-ui, sans-serif"},
        "hoverlabel": {"bgcolor": "#241813", "font": {"color": "#FFF8F2"}},
        "legend": {"orientation": "h", "y": 1.1, "x": 0},
    }
    layout.update(extra)
    return layout


def _show_chart(figure: go.Figure, *, key: str) -> None:
    """Render an accessible responsive chart inside the existing glass system."""
    with glass_card(f"model-insights-chart-{key}"):
        st.plotly_chart(
            figure,
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
            key=f"model-insights-plot-{key}",
        )


def _download_frame(
    frame: pd.DataFrame, *, label: str, filename: str, key: str
) -> None:
    """Offer a download only for a currently displayed real artifact table."""
    if frame.empty:
        return
    try:
        encoded = dataframe_csv(frame)
    except (ModelInsightsError, OSError, TypeError, ValueError):
        st.caption("This artifact could not be encoded for download.")
        return
    st.download_button(
        label,
        data=encoded,
        file_name=filename,
        mime="text/csv",
        key=key,
        width="content",
    )


def _initialize_model_insights_state() -> None:
    """Initialize only compact Model Insights state; never persist model/data objects."""
    defaults: dict[str, Any] = {
        "model_insights_confusion_mode": "Counts",
        "model_insights_importance_count": 20,
        "model_insights_diagnostics_requested": False,
        "model_insights_diagnostics_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_model_insights_styles() -> None:
    """Inject small, page-scoped responsive treatments missing from shared UI."""
    st.markdown(
        """
        <style>
            @keyframes model-insights-gradient-flow {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            .model-insights-status,
            .model-insights-copy,
            .model-insights-note,
            .model-insights-empty,
            .model-insights-selection-copy,
            .model-insights-definition,
            .model-insights-transparency {
                color: var(--ui-color-text-secondary);
                font-family: var(--ui-type-font-sans);
                line-height: 1.56;
            }

            .model-insights-status {
                display: inline-flex;
                flex-wrap: wrap;
                align-items: center;
                gap: 0.48rem;
                margin: -0.85rem 0 1.32rem;
                padding: 0.48rem 0.74rem;
                border: 1px solid var(--ui-color-border-subtle);
                border-radius: var(--ui-radius-sm);
                background: color-mix(in srgb, var(--ui-color-surface-elevated) 84%, transparent);
                color: var(--ui-color-text-primary);
                font-size: 0.8rem;
            }

            .model-insights-status__dot {
                width: 0.52rem;
                height: 0.52rem;
                border-radius: 50%;
                background: var(--ui-color-success);
                box-shadow: 0 0 0.45rem color-mix(in srgb, var(--ui-color-success) 42%, transparent);
            }

            .model-insights-status__dot--warning {
                background: var(--ui-color-warning);
                box-shadow: 0 0 0.45rem color-mix(in srgb, var(--ui-color-warning) 38%, transparent);
            }

            .model-insights-header-icon {
                display: inline-grid;
                width: 2rem;
                height: 2rem;
                place-items: center;
                margin: -0.65rem 0 0.55rem 0.45rem;
                border: 1px solid var(--ui-color-border-subtle);
                border-radius: var(--ui-radius-sm);
                background: color-mix(in srgb, var(--ui-color-primary) 13%, transparent);
                color: var(--ui-color-accent);
            }

            .model-insights-header-icon svg { width: 1.1rem; height: 1.1rem; }

            [class*="st-key-ui-glass-model-insights-overview"],
            [class*="st-key-ui-glass-model-insights-selection"],
            [class*="st-key-ui-glass-model-insights-diagnostics"] {
                position: relative;
                overflow: hidden;
            }

            [class*="st-key-ui-glass-model-insights-overview"]::before,
            [class*="st-key-ui-glass-model-insights-selection"]::before,
            [class*="st-key-ui-glass-model-insights-diagnostics"]::before {
                position: absolute;
                top: 0;
                right: 0;
                left: 0;
                height: 3px;
                background: linear-gradient(
                    90deg,
                    var(--ui-color-primary),
                    var(--ui-color-accent),
                    var(--ui-color-warning),
                    var(--ui-color-primary)
                );
                background-size: 220% 100%;
                content: "";
                animation: model-insights-gradient-flow 6.5s ease infinite;
            }

            .model-insights-kicker,
            .model-insights-small-label,
            .model-insights-status-label {
                margin: 0;
                color: var(--ui-color-accent);
                font-family: var(--ui-type-font-mono);
                font-size: 0.68rem;
                font-weight: 800;
                letter-spacing: 0.12em;
                text-transform: uppercase;
            }

            .model-insights-production-name {
                margin: 0.24rem 0 0;
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: clamp(1.45rem, 3vw, 2.25rem);
                font-weight: 850;
                letter-spacing: -0.045em;
                line-height: 1.05;
            }

            .model-insights-production-meta,
            .model-insights-selection-copy,
            .model-insights-note,
            .model-insights-empty,
            .model-insights-definition,
            .model-insights-transparency {
                margin: 0.62rem 0 0;
                font-size: 0.88rem;
            }

            .model-insights-detail-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.68rem;
                margin-top: 0.82rem;
            }

            .model-insights-detail {
                min-width: 0;
                padding: 0.68rem;
                border: 1px solid var(--ui-color-border-subtle);
                border-radius: var(--ui-radius-sm);
                background: color-mix(in srgb, var(--ui-color-surface) 72%, transparent);
            }

            .model-insights-detail__label {
                display: block;
                color: var(--ui-color-text-muted);
                font-family: var(--ui-type-font-mono);
                font-size: 0.65rem;
                font-weight: 800;
                letter-spacing: 0.095em;
                text-transform: uppercase;
            }

            .model-insights-detail__value {
                display: block;
                margin-top: 0.23rem;
                overflow-wrap: anywhere;
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.86rem;
                font-weight: 720;
                line-height: 1.35;
            }

            .model-insights-selection-rule {
                margin: 0.8rem 0 0;
                padding: 0.72rem 0.82rem;
                border-left: 3px solid var(--ui-color-primary);
                border-radius: 0 var(--ui-radius-sm) var(--ui-radius-sm) 0;
                background: color-mix(in srgb, var(--ui-color-primary) 10%, transparent);
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.87rem;
                line-height: 1.52;
            }

            .model-insights-comparison-caption,
            .model-insights-chart-caption {
                margin: 0 0 0.7rem;
                color: var(--ui-color-text-muted);
                font-family: var(--ui-type-font-sans);
                font-size: 0.79rem;
                line-height: 1.48;
            }

            .model-insights-selected-row {
                color: var(--ui-color-accent-hover);
                font-weight: 800;
            }

            .model-insights-class-map {
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin-top: 0.78rem;
            }

            .model-insights-class-chip {
                padding: 0.32rem 0.52rem;
                border: 1px solid var(--ui-color-border-subtle);
                border-radius: var(--ui-radius-sm);
                background: color-mix(in srgb, var(--ui-color-primary) 11%, transparent);
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.75rem;
            }

            .model-insights-empty {
                padding: 0.78rem 0.86rem;
                border: 1px dashed var(--ui-color-border-subtle);
                border-radius: var(--ui-radius-sm);
                background: color-mix(in srgb, var(--ui-color-surface) 68%, transparent);
            }

            .model-insights-status-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.66rem;
            }

            .model-insights-status-card {
                min-width: 0;
                padding: 0.7rem;
                border: 1px solid var(--ui-color-border-subtle);
                border-radius: var(--ui-radius-sm);
                background: color-mix(in srgb, var(--ui-color-surface) 74%, transparent);
            }

            .model-insights-status-card__value {
                display: block;
                margin-top: 0.28rem;
                overflow-wrap: anywhere;
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.82rem;
                font-weight: 720;
            }

            .model-insights-diagnostic-copy { margin: 0 0 0.8rem; }

            .model-insights-limitations {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.68rem;
                margin-top: 0.8rem;
            }

            .model-insights-limitation {
                padding: 0.75rem;
                border: 1px solid color-mix(in srgb, var(--ui-color-warning) 32%, transparent);
                border-radius: var(--ui-radius-sm);
                background: color-mix(in srgb, var(--ui-color-warning) 8%, transparent);
                color: var(--ui-color-text-secondary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.84rem;
                line-height: 1.5;
            }

            @media (max-width: 760px) {
                .st-key-model-insights-kpis [data-testid="stHorizontalBlock"],
                .st-key-model-insights-compare-charts [data-testid="stHorizontalBlock"],
                .st-key-model-insights-feature-grid [data-testid="stHorizontalBlock"],
                .st-key-model-insights-heldout-grid [data-testid="stHorizontalBlock"],
                .st-key-model-insights-export-grid [data-testid="stHorizontalBlock"] {
                    flex-direction: column;
                }

                .st-key-model-insights-kpis [data-testid="stColumn"],
                .st-key-model-insights-compare-charts [data-testid="stColumn"],
                .st-key-model-insights-feature-grid [data-testid="stColumn"],
                .st-key-model-insights-heldout-grid [data-testid="stColumn"],
                .st-key-model-insights-export-grid [data-testid="stColumn"] {
                    width: 100% !important;
                    flex: 1 1 100% !important;
                }

                .model-insights-detail-grid,
                .model-insights-status-grid,
                .model-insights-limitations { grid-template-columns: 1fr; }
            }

            @media (prefers-reduced-motion: reduce) {
                [class*="st-key-ui-glass-model-insights-overview"]::before,
                [class*="st-key-ui-glass-model-insights-selection"]::before,
                [class*="st-key-ui-glass-model-insights-diagnostics"]::before {
                    animation: none;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header(overview: Mapping[str, Any]) -> None:
    """Render the source-aware page identity and verified mapping status."""
    selected_name = _selected_model_name(overview)
    selected = _selected_model(overview)
    context = _as_mapping(_mapping_value(overview, "metric_context", "evaluation"))
    split = (
        _mapping_value(context, "split_label", "split", "dataset_label")
        or "Saved evaluation artifacts"
    )
    mapping = _class_rows(
        _mapping_value(selected, "class_mapping", "classes", default=None)
    )
    if not mapping:
        mapping = _class_rows(
            _mapping_value(overview, "class_mapping", "classes", default=None)
        )
    class_text = ", ".join(f"{value} = {label}" for value, label in mapping)
    page_header(
        "Model Insights",
        eyebrow="Three-class football outcome model",
        subtitle=(
            "Inspect the selected match-outcome model, saved validation comparison "
            "metrics, and optional immutable held-out test diagnostics without retraining."
        ),
    )
    render_svg_icon(
        MODEL_INSIGHTS_ICON,
        size=18,
        wrapper_class="model-insights-header-icon",
    )
    verified = bool(
        _mapping_value(
            selected,
            "class_mapping_verified",
            "classes_verified",
            default=bool(mapping),
        )
    )
    dot_class = "" if verified else " model-insights-status__dot--warning"
    mapping_text = class_text or "class mapping unavailable"
    st.markdown(
        f"""
        <div class="model-insights-status" aria-label="Model artifact status">
            <span class="model-insights-status__dot{dot_class}" aria-hidden="true"></span>
            <strong>{escape(selected_name)}</strong>
            <span>·</span><span>{escape(str(split))}</span>
            <span>·</span><span>{escape(mapping_text)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_production_overview(overview: Mapping[str, Any]) -> None:
    """Render only model metadata actually found by the artifact service."""
    section_title(
        "Production model overview",
        description="The model artifact currently selected by the repository's training and simulator workflow.",
        eyebrow="Selected artifact",
    )
    selected = _selected_model(overview)
    model_name = _selected_model_name(overview)
    details = [
        (
            "Model family",
            _mapping_value(selected, "family", "model_family", "estimator_type"),
        ),
        (
            "Artifact",
            _mapping_value(
                selected, "artifact_name", "artifact", "filename", "model_file"
            ),
        ),
        (
            "Feature count",
            _mapping_value(selected, "feature_count", "input_feature_count"),
        ),
        ("Class count", _mapping_value(selected, "class_count", "n_classes")),
        (
            "Training split rows",
            _mapping_value(selected, "training_dataset_size", "training_rows"),
        ),
        (
            "Validation split rows",
            _mapping_value(selected, "validation_dataset_size", "validation_rows"),
        ),
        (
            "Held-out test rows",
            _mapping_value(selected, "test_dataset_size", "test_rows"),
        ),
        (
            "Model size",
            _format_file_size(
                _mapping_value(selected, "model_size_bytes", "artifact_size_bytes")
            ),
        ),
        (
            "Preprocessing",
            _mapping_value(selected, "preprocessor_status", "pipeline_status"),
        ),
        ("Training date", _mapping_value(selected, "training_date", "fitted_at")),
        (
            "Evaluation date",
            _mapping_value(selected, "evaluation_date", "evaluated_at"),
        ),
    ]
    training_time = _mapping_value(
        selected, "training_time", "training_seconds", "fit_time"
    )
    if training_time is not None:
        details.append(("Training duration", _format_duration(training_time)))
    inference_time = _mapping_value(selected, "inference_time", "inference_time_ms")
    if inference_time is not None:
        details.append(("Measured inference time", str(inference_time)))
    rows = [
        (label, value)
        for label, value in details
        if value not in (None, "", "Unavailable")
    ]
    with glass_card("model-insights-overview"):
        left, right = st.columns((1.1, 1), gap="large")
        with left:
            st.markdown(
                '<p class="model-insights-kicker">Selected production model</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<h2 class="model-insights-production-name">{escape(model_name)}</h2>',
                unsafe_allow_html=True,
            )
            rank = _mapping_value(selected, "rank", "selection_rank")
            rank_text = (
                f"Saved ranking position: {escape(str(rank))}."
                if rank is not None
                else "Saved ranking position was not recorded."
            )
            st.markdown(
                f'<p class="model-insights-production-meta">{rank_text}</p>',
                unsafe_allow_html=True,
            )
            mapping = _class_rows(_mapping_value(selected, "class_mapping", "classes"))
            if not mapping:
                mapping = _class_rows(
                    _mapping_value(overview, "class_mapping", "classes")
                )
            if mapping:
                chips = "".join(
                    f'<span class="model-insights-class-chip">{escape(raw)} · {escape(label)}</span>'
                    for raw, label in mapping
                )
                st.markdown(
                    f'<div class="model-insights-class-map">{chips}</div>',
                    unsafe_allow_html=True,
                )
        with right:
            if rows:
                markup = "".join(
                    '<div class="model-insights-detail">'
                    f'<span class="model-insights-detail__label">{escape(label)}</span>'
                    f'<span class="model-insights-detail__value">{escape(str(value))}</span>'
                    "</div>"
                    for label, value in rows
                )
                st.markdown(
                    f'<div class="model-insights-detail-grid">{markup}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p class="model-insights-empty">No additional production metadata was recorded in the discovered artifacts.</p>',
                    unsafe_allow_html=True,
                )


def _render_selection_explanation(overview: Mapping[str, Any]) -> None:
    """Describe selection using the persisted rule rather than inferred rationale."""
    section_title(
        "Model selection",
        eyebrow="Recorded ranking",
        description="Why this artifact was selected, using only the repository's saved ranking and trainer logic.",
    )
    selection = _as_mapping(
        _mapping_value(overview, "selection", "selection_explanation")
    )
    explanation = _mapping_value(selection, "explanation", "text", "reason")
    if explanation is None:
        explanation = _mapping_value(
            overview, "selection_explanation", "selection_reason"
        )
    rule = _mapping_value(selection, "rule", "selection_rule", "criterion")
    with glass_card("model-insights-selection"):
        if explanation:
            st.markdown(
                f'<p class="model-insights-selection-copy">{escape(str(explanation))}</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="model-insights-empty">The saved artifacts identify this as the selected model, but the exact selection rule was not recorded.</p>',
                unsafe_allow_html=True,
            )
        if rule:
            st.markdown(
                f'<p class="model-insights-selection-rule"><strong>Recorded rule:</strong> {escape(str(rule))}</p>',
                unsafe_allow_html=True,
            )


def _render_performance_kpis(overview: Mapping[str, Any]) -> None:
    """Show the selected model's saved validation metrics without calling them confidence."""
    section_title(
        "Validation performance",
        eyebrow="Saved comparison metrics",
        description="Metrics from the persisted comparison experiment. They measure historical held-out validation performance, not certainty for an individual prediction.",
    )
    selected = _selected_model(overview)
    metrics = _as_mapping(_mapping_value(overview, "metrics", "selected_metrics"))
    if not metrics:
        metrics = _as_mapping(_mapping_value(selected, "metrics", "validation_metrics"))
    context = _as_mapping(_mapping_value(overview, "metric_context", "evaluation"))
    split = _mapping_value(context, "split_label", "split") or "Validation split"
    averaging = _mapping_value(context, "averaging", "average_method")
    metric_items = (
        ("Accuracy", "accuracy", "Share of correctly classified validation matches."),
        (
            "Precision",
            "precision",
            "Weighted precision when that averaging method is recorded.",
        ),
        ("Recall", "recall", "Weighted recall when that averaging method is recorded."),
        ("F1 score", "f1", "Weighted F1 when that averaging method is recorded."),
        (
            "Log loss",
            "log_loss",
            "Lower is better; it evaluates assigned outcome probabilities.",
        ),
    )
    available = [
        (label, key, description, _mapping_value(metrics, key, key.replace("_", " ")))
        for label, key, description in metric_items
    ]
    available = [item for item in available if item[3] is not None]
    if not available:
        st.info("No valid saved validation metrics were found for the selected model.")
        return
    with st.container(key="model-insights-kpis", border=False):
        columns = st.columns(min(len(available), 5), gap="small")
        for column, (label, key, definition, value) in zip(columns, available):
            with column:
                if key == "log_loss":
                    display = _format_number(value, decimals=3)
                    caption = f"{split} · lower is better"
                else:
                    display = _format_percent(value)
                    caption = str(split)
                metric_card(
                    label, display, caption=caption, delta=definition, trend="neutral"
                )
    context_bits = [str(split)]
    if averaging:
        context_bits.append(str(averaging))
    st.markdown(
        f'<p class="model-insights-definition">Metric context: {escape(" · ".join(context_bits))}. Accuracy is a classification score, not model confidence.</p>',
        unsafe_allow_html=True,
    )


def _comparison_display_frame(frame: pd.DataFrame, selected_name: str) -> pd.DataFrame:
    """Create a compact display/export view without modifying source metrics."""
    output = frame.copy()
    model_column = _column(output, "model", "model_name", "estimator")
    rank_column = _column(output, "rank")
    if rank_column:
        output = output.sort_values(rank_column, kind="stable", na_position="last")
    elif model_column:
        output = output.sort_values(
            model_column,
            key=lambda values: values.astype(str).str.casefold(),
            kind="stable",
        )
    if model_column:
        output.insert(
            len(output.columns),
            "Selected",
            output[model_column]
            .astype(str)
            .str.casefold()
            .eq(selected_name.casefold())
            .map({True: "Yes", False: "No"}),
        )
    return output.reset_index(drop=True)


def _render_comparison_table(overview: Mapping[str, Any]) -> pd.DataFrame:
    """Render the complete saved model comparison and its real CSV export."""
    section_title("Model comparison")
    frame = _comparison_frame(overview)
    if frame.empty:
        st.info("No valid saved model-comparison table was found.")
        return frame
    selected_name = _selected_model_name(overview)
    display = _comparison_display_frame(frame, selected_name)
    context = _as_mapping(_mapping_value(overview, "metric_context", "evaluation"))
    source = (
        _mapping_value(context, "source", "source_file", "metric_source")
        or "saved comparison artifact"
    )
    st.markdown(
        f'<p class="model-insights-comparison-caption">Evaluation source: {escape(str(source))}. Classification metrics are comparison metrics from the same saved validation experiment; log loss is lower-is-better.</p>',
        unsafe_allow_html=True,
    )
    column_config: dict[str, Any] = {}
    for raw_name in display.columns:
        normalized = re.sub(r"[^a-z0-9]+", "", str(raw_name).casefold())
        if normalized in {"accuracy", "precision", "recall", "f1", "f1score"}:
            column_config[str(raw_name)] = st.column_config.NumberColumn(
                str(raw_name), format="%.2f%%"
            )
        elif normalized in {"logloss", "trainingtime", "fittime"}:
            column_config[str(raw_name)] = st.column_config.NumberColumn(
                str(raw_name), format="%.3f"
            )
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config=column_config,
        key="model-insights-comparison-table",
    )
    # Download button removed per cleanup instructions; the comparison table
    # remains visible. Exports may be provided elsewhere if required.
    return display


def _render_comparison_charts(overview: Mapping[str, Any], frame: pd.DataFrame) -> None:
    # Performance trade-offs chart removed per cleanup request. Leave a no-op
    # function so other call sites remain import-compatible. The comparison
    # table above still renders; restore charting here only when needed.
    return


def _render_feature_importance(overview: Mapping[str, Any]) -> pd.DataFrame:
    """Feature importance display intentionally removed.

    The page previously showed named feature importance and allowed CSV
    export. Per the cleanup request these visible controls were removed.
    The service still provides validated artifacts; restore display using
    the service helpers instead of reintroducing placeholders.
    """
    return pd.DataFrame()


def _render_feature_groups(overview: Mapping[str, Any]) -> None:
    """Transparent grouping controls removed.

    This page no longer displays feature-group aggregation UI or the
    corresponding chart.
    """
    return


def _render_explainability_availability(overview: Mapping[str, Any]) -> None:
    """Explainability availability section removed.

    Responsible empty-state messaging for explainability artifacts is not
    shown in the simplified Model Insights route.
    """
    return


def _diagnostics_requested() -> bool:
    """On-demand held-out diagnostics have been removed from this page."""
    return False


def _load_diagnostics_if_requested() -> Mapping[str, Any] | None:
    """Held-out diagnostics loading is disabled for this simplified page."""
    return None


def _confusion_payload(
    diagnostics: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[str], str | None]:
    """Normalize an already validated held-out confusion payload for display."""
    payload = _mapping_value(diagnostics, "confusion_matrix", "confusion")
    source: str | None = None
    labels: list[str] = []
    if isinstance(payload, Mapping):
        source_value = _mapping_value(payload, "source", "source_label")
        source = str(source_value) if source_value else None
        labels = [
            str(item)
            for item in _mapping_value(payload, "labels", "class_labels", default=[])
            or []
        ]
        payload = _mapping_value(payload, "matrix", "values", "data")
    frame = _coerce_frame(payload)
    if (
        frame.empty
        and isinstance(payload, Sequence)
        and not isinstance(payload, (str, bytes))
    ):
        try:
            frame = pd.DataFrame(payload)
        except (TypeError, ValueError):
            frame = pd.DataFrame()
    return frame, labels, source


def _render_confusion_matrix(
    overview: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> pd.DataFrame:
    """Render raw or row-normalized held-out counts without hiding raw supports."""
    section_title(
        "Confusion matrix",
        eyebrow="Held-out test set",
        description="Rows are actual outcomes; columns are predicted outcomes. Counts remain available even when the display uses row-normalized percentages.",
    )
    matrix, labels, source = _confusion_payload(diagnostics)
    if matrix.empty:
        st.info(
            "No saved held-out confusion matrix was found, and this page will not calculate one from training data."
        )
        return pd.DataFrame()
    try:
        values = matrix.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    except (TypeError, ValueError):
        st.warning(
            "The held-out confusion matrix contains invalid values and is not displayed."
        )
        return pd.DataFrame()
    if (
        values.ndim != 2
        or values.shape[0] != values.shape[1]
        or (values < 0).any()
        or pd.isna(values).any()
    ):
        st.warning(
            "The held-out confusion matrix has an invalid shape or contains invalid counts."
        )
        return pd.DataFrame()
    verified_labels = _class_labels(overview, diagnostics)
    labels = labels or verified_labels
    if len(labels) != values.shape[0]:
        st.warning(
            "The held-out confusion matrix dimensions do not match the verified class mapping."
        )
        return pd.DataFrame()
    raw = pd.DataFrame(values.astype(int), index=labels, columns=labels)
    mode = st.radio(
        "Confusion-matrix display",
        ("Counts", "Row-normalized percentages"),
        horizontal=True,
        key="model_insights_confusion_mode",
        help="Row normalization divides each actual-outcome row by its support; raw counts remain downloadable below.",
    )
    row_sums = values.sum(axis=1, keepdims=True)
    if mode == "Row-normalized percentages":
        display = values / row_sums.clip(min=1)
        text = [[f"{item:.1%}" for item in row] for row in display]
        hover = (
            "Actual: %{y}<br>Predicted: %{x}<br>Row-normalized: %{z:.1%}<extra></extra>"
        )
        colorbar_title = "Row %"
    else:
        display = values
        text = [[f"{int(item):,}" for item in row] for row in display]
        hover = "Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:.0f}<extra></extra>"
        colorbar_title = "Count"
    figure = go.Figure(
        go.Heatmap(
            z=display,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            hovertemplate=hover,
            colorscale=[[0, "#241813"], [0.5, "#B94E13"], [1, "#F0B967"]],
            colorbar={"title": colorbar_title},
        )
    )
    figure.update_layout(
        **_plotly_layout(
            height=420,
            title="Actual outcome × predicted outcome",
            xaxis={"title": "Predicted outcome"},
            yaxis={"title": "Actual outcome", "autorange": "reversed"},
        )
    )
    _show_chart(figure, key="confusion-matrix")
    total = int(values.sum())
    source_text = f" Source: {source}." if source else ""
    st.markdown(
        f'<p class="model-insights-note">Held-out evaluation rows represented: {total:,}.{escape(source_text)}</p>',
        unsafe_allow_html=True,
    )
    _download_frame(
        raw.reset_index(names="Actual outcome"),
        label="Download confusion matrix CSV",
        filename="held_out_test_confusion_matrix.csv",
        key="model-insights-download-confusion",
    )
    return raw


def _render_class_performance(
    overview: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> pd.DataFrame:
    """Render true held-out classification-report metrics when service supplied them."""
    section_title(
        "Per-class performance",
        eyebrow="Held-out test report",
        description="Precision, recall, F1, and support are reported separately by verified outcome class when the test report is available.",
    )
    report = _first_frame(
        diagnostics, "class_report", "classification_report", "per_class_metrics"
    )
    if report.empty:
        st.info(
            "No held-out classification report is available from the current artifacts."
        )
        return report
    label_column = _column(report, "class", "label", "outcome", "class_label")
    precision_column = _column(report, "precision")
    recall_column = _column(report, "recall")
    f1_column = _column(report, "f1", "f1_score")
    support_column = _column(report, "support", "count")
    if not label_column or not all(
        (precision_column, recall_column, f1_column, support_column)
    ):
        st.warning(
            "The held-out classification report is incomplete and is not displayed as class-level performance."
        )
        return pd.DataFrame()
    display = report.copy()
    display[precision_column] = pd.to_numeric(
        display[precision_column], errors="coerce"
    )
    display[recall_column] = pd.to_numeric(display[recall_column], errors="coerce")
    display[f1_column] = pd.to_numeric(display[f1_column], errors="coerce")
    display[support_column] = pd.to_numeric(display[support_column], errors="coerce")
    display = display.dropna(
        subset=[
            label_column,
            precision_column,
            recall_column,
            f1_column,
            support_column,
        ]
    )
    if display.empty:
        st.warning("The held-out classification report has no valid class rows.")
        return display
    configured = {
        precision_column: st.column_config.NumberColumn("Precision", format="%.2f%%"),
        recall_column: st.column_config.NumberColumn("Recall", format="%.2f%%"),
        f1_column: st.column_config.NumberColumn("F1 score", format="%.2f%%"),
        support_column: st.column_config.NumberColumn("Support", format="%d"),
    }
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config=configured,
        key="model-insights-class-report",
    )
    hardest = display.loc[display[recall_column].idxmin(), label_column]
    st.markdown(
        f'<p class="model-insights-note">On this recorded held-out test set, {escape(str(hardest))} has the lowest recall among the displayed classes. This is a description of this evaluation set, not a claim about the class in all contexts.</p>',
        unsafe_allow_html=True,
    )
    _download_frame(
        display,
        label="Download per-class metrics CSV",
        filename="held_out_test_per_class_metrics.csv",
        key="model-insights-download-class-report",
    )
    return display


def _error_summary_rows(payload: Any) -> list[tuple[str, str]]:
    """Convert a compact service error summary into safe metric-card label/value pairs."""
    data = _as_mapping(payload)
    aliases = (
        (
            "Incorrect predictions",
            "misclassification_count",
            "error_count",
            "incorrect_count",
        ),
        ("Error rate", "error_rate", "misclassification_rate"),
        (
            "High-confidence errors",
            "high_confidence_incorrect_count",
            "high_confidence_error_count",
        ),
        ("Low-margin predictions", "low_margin_count", "low_margin_predictions"),
    )
    rows: list[tuple[str, str]] = []
    for label, *names in aliases:
        value = _mapping_value(data, *names)
        if value is None:
            continue
        text = _format_percent(value) if "rate" in names[0] else _format_number(value)
        rows.append((label, text))
    return rows


def _render_error_analysis(diagnostics: Mapping[str, Any]) -> None:
    """Render held-out errors only when a real prediction-derived summary exists."""
    section_title(
        "Error analysis",
        eyebrow="Held-out test predictions",
        description="Descriptions below are calculated from test-set predictions only. Confidence is the largest predicted class probability; margin is the difference between the largest and second-largest probabilities.",
    )
    analysis = _as_mapping(_mapping_value(diagnostics, "error_analysis", "errors"))
    if not analysis:
        st.info(
            "No held-out row-level prediction summary is available for detailed error analysis."
        )
        return
    rows = _error_summary_rows(analysis)
    if rows:
        columns = st.columns(min(4, len(rows)), gap="small")
        for column, (label, value) in zip(columns, rows):
            with column:
                metric_card(label, value, caption="Held-out test set", trend="neutral")
    definitions = [
        _mapping_value(analysis, "confidence_definition"),
        _mapping_value(analysis, "margin_definition"),
    ]
    definitions = [str(item) for item in definitions if item]
    if definitions:
        st.markdown(
            f'<p class="model-insights-note">{escape(" ".join(definitions))}</p>',
            unsafe_allow_html=True,
        )
    common = _as_mapping(
        _mapping_value(analysis, "most_common_confusion", "common_confusion")
    )
    if common:
        true_label = _mapping_value(common, "true_label", "actual", "actual_label")
        predicted_label = _mapping_value(
            common, "predicted_label", "prediction", "predicted"
        )
        count = _mapping_value(common, "count", "matches")
        if true_label is not None and predicted_label is not None and count is not None:
            st.markdown(
                f'<p class="model-insights-selection-rule">Most common recorded error: actual <strong>{escape(str(true_label))}</strong> predicted as <strong>{escape(str(predicted_label))}</strong> in {escape(_format_number(count))} held-out rows.</p>',
                unsafe_allow_html=True,
            )
    by_class = _first_frame(
        analysis, "by_true_class", "misclassifications_by_class", "error_by_class"
    )
    if not by_class.empty:
        st.dataframe(
            by_class,
            hide_index=True,
            width="stretch",
            key="model-insights-error-by-class",
        )
    grouped_sources = (
        ("Error rate by competition", "by_competition"),
        ("Error rate by neutral-venue flag", "by_neutral_venue"),
    )
    for heading, key in grouped_sources:
        grouped = _first_frame(analysis, key)
        if grouped.empty:
            continue
        st.markdown(
            f'<p class="model-insights-note"><strong>{escape(heading)}.</strong> '
            "Only groups with at least 25 held-out rows are included; sample sizes remain visible.</p>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            grouped,
            hide_index=True,
            width="stretch",
            key=f"model-insights-{_safe_key(key)}",
        )


def _render_calibration(diagnostics: Mapping[str, Any]) -> pd.DataFrame:
    """Render one-vs-rest held-out calibration curves only from verified test probabilities."""
    section_title(
        "Probability calibration",
        eyebrow="Held-out test probabilities",
        description="One-vs-rest calibration compares assigned probability with observed frequency. The diagonal is a reference, not proof that a model is calibrated.",
    )
    payload = _mapping_value(diagnostics, "calibration", "calibration_analysis")
    calibration = _as_mapping(payload)
    bins = (
        _first_frame(calibration, "bins", "curve", "calibration_bins")
        if calibration
        else _coerce_frame(payload)
    )
    if bins.empty:
        st.info(
            "Calibration analysis requires saved or safely derived held-out probabilities, which are not available in the current loaded diagnostics."
        )
        return bins
    class_column = _column(bins, "class", "label", "outcome", "class_label")
    predicted_column = _column(
        bins,
        "mean_predicted_probability",
        "predicted_probability",
        "mean_prediction",
        "confidence",
    )
    observed_column = _column(
        bins, "observed_frequency", "observed_rate", "fraction_positive"
    )
    count_column = _column(bins, "count", "bin_count", "support")
    if not class_column or not predicted_column or not observed_column:
        st.warning(
            "The held-out calibration artifact is incomplete and cannot be charted safely."
        )
        return pd.DataFrame()
    display = bins.copy()
    for column in (predicted_column, observed_column):
        display[column] = pd.to_numeric(display[column], errors="coerce")
    display = display.dropna(subset=[class_column, predicted_column, observed_column])
    if display.empty:
        st.warning("The held-out calibration artifact has no valid probability bins.")
        return display
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"color": "#9B887B", "dash": "dash"},
            name="Perfect calibration",
            hoverinfo="skip",
        )
    )
    for label, group in display.groupby(class_column, sort=True):
        color = _CLASS_COLORS.get(str(label), "#E3A13B")
        custom = group[count_column] if count_column else None
        hover = (
            "Class: "
            + escape(str(label))
            + "<br>Mean predicted probability: %{x:.2%}<br>Observed frequency: %{y:.2%}"
        )
        if custom is not None:
            hover += "<br>Rows in bin: %{customdata:.0f}"
        hover += "<extra></extra>"
        figure.add_trace(
            go.Scatter(
                x=group[predicted_column],
                y=group[observed_column],
                mode="lines+markers",
                name=str(label),
                marker={"size": 8, "color": color},
                line={"color": color},
                customdata=custom,
                hovertemplate=hover,
            )
        )
    figure.update_layout(
        **_plotly_layout(
            height=420,
            title="One-vs-rest calibration by outcome",
            xaxis={
                "title": "Mean predicted probability",
                "range": [0, 1],
                "tickformat": ".0%",
                "gridcolor": "rgba(255, 218, 184, 0.14)",
            },
            yaxis={
                "title": "Observed frequency",
                "range": [0, 1],
                "tickformat": ".0%",
                "gridcolor": "rgba(255, 218, 184, 0.14)",
            },
        )
    )
    _show_chart(figure, key="calibration")
    brier = _first_frame(calibration, "brier_scores", "brier")
    if not brier.empty:
        st.caption(
            "One-vs-rest Brier scores are reported by class; lower values are better."
        )
        st.dataframe(
            brier,
            hide_index=True,
            width="stretch",
            key="model-insights-brier-scores",
        )
    _download_frame(
        display,
        label="Download calibration-bin CSV",
        filename="held_out_test_calibration_bins.csv",
        key="model-insights-download-calibration",
    )
    return display


def _render_class_distribution(diagnostics: Mapping[str, Any]) -> None:
    """Render a real held-out target distribution only when test counts were supplied."""
    section_title(
        "Class distribution",
        eyebrow="Held-out test split",
        description="The outcome balance in the same held-out test split used for the optional diagnostics. Class imbalance can affect how summary metrics should be read.",
    )
    payload = _mapping_value(
        diagnostics,
        "class_distribution",
        "test_class_distribution",
        "target_distribution",
    )
    frame = _coerce_frame(payload)
    if frame.empty and isinstance(payload, Mapping):
        try:
            frame = pd.DataFrame(
                {"Class": list(payload.keys()), "Count": list(payload.values())}
            )
        except (TypeError, ValueError):
            frame = pd.DataFrame()
    label_column = _column(frame, "class", "label", "outcome", "class_label")
    count_column = _column(frame, "count", "support", "rows")
    if frame.empty or not label_column or not count_column:
        st.info("No held-out class-distribution artifact is available.")
        return
    display = frame.copy()
    display[count_column] = pd.to_numeric(display[count_column], errors="coerce")
    display = display.dropna(subset=[label_column, count_column])
    if display.empty:
        st.info("No valid held-out class counts are available.")
        return
    total = float(display[count_column].sum())
    figure = go.Figure(
        go.Bar(
            x=display[label_column].astype(str),
            y=display[count_column],
            marker_color=[
                _CLASS_COLORS.get(str(label), "#9B887B")
                for label in display[label_column]
            ],
            customdata=(display[count_column] / total) if total else None,
            hovertemplate="%{x}<br>Rows: %{y:.0f}<br>Share: %{customdata:.1%}<extra></extra>",
        )
    )
    figure.update_layout(
        **_plotly_layout(
            height=330,
            title="Held-out target-class counts",
            xaxis={"title": None},
            yaxis={"title": "Rows", "gridcolor": "rgba(255, 218, 184, 0.14)"},
            showlegend=False,
        )
    )
    _show_chart(figure, key="class-distribution")


def _render_limitations(overview: Mapping[str, Any]) -> None:
    """Responsible-use limitations section removed.

    The page no longer renders the prior limitations panel.
    """
    return


def _status_items(overview: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Normalize artifact availability from dict, records, or boolean fields."""
    status = _mapping_value(
        overview, "artifact_status", "artifact_availability", "availability"
    )
    if isinstance(status, Mapping):
        items: list[tuple[str, str]] = []
        for label, value in status.items():
            if isinstance(value, Mapping):
                reason = _mapping_value(value, "reason", "detail", "message")
                available = _mapping_value(value, "available", "is_available")
                explicit_status = _mapping_value(value, "status", "label")
                value = explicit_status if explicit_status is not None else available
                if reason and value in (None, False, "Unavailable", "unavailable"):
                    value = f"Unavailable — {reason}"
            if isinstance(value, bool):
                text = "Available" if value else "Unavailable"
            else:
                text = str(value)
            items.append((str(label), text))
        return items
    if isinstance(status, Sequence) and not isinstance(status, (str, bytes)):
        items = []
        for row in status:
            if isinstance(row, Mapping):
                label = _mapping_value(row, "artifact", "name", "label")
                value = _mapping_value(row, "status", "available", "value")
                if label is not None and value is not None:
                    items.append(
                        (
                            str(label),
                            "Available"
                            if value is True
                            else "Unavailable"
                            if value is False
                            else str(value),
                        )
                    )
        return items
    return []


def _render_transparency_and_status(
    overview: Mapping[str, Any], diagnostics: Mapping[str, Any] | None
) -> None:
    """Artifact provenance and diagnostic status have been removed from this page."""
    return


def _render_exports(
    comparison: pd.DataFrame,
    importance: pd.DataFrame,
    diagnostics: Mapping[str, Any] | None,
) -> None:
    """Export controls removed from this simplified page."""
    return


def render_model_insights() -> None:
    """Render a simplified Model Insights route.

    Per cleanup instructions, advanced on-demand diagnostics, feature
    importance charts, grouped feature summaries, responsible-use notes,
    artifact provenance, and exports are not shown in this UI.
    """
    apply_theme()
    _initialize_model_insights_state()
    _render_model_insights_styles()
    try:
        overview = get_model_insights_overview()
    except ModelInsightsError as error:
        st.error(f"Model Insights is unavailable: {error}")
        return
    except (OSError, ValueError, TypeError):
        st.error("Model Insights could not read the saved model artifacts safely.")
        return
    if not isinstance(overview, Mapping):
        st.error("Model Insights received an invalid artifact summary.")
        return
    with animated_container("model-insights-page", animation="fade_up"):
        _render_header(overview)
        _render_production_overview(overview)
        _render_selection_explanation(overview)
        _render_performance_kpis(overview)
        comparison = _render_comparison_table(overview)
        _render_comparison_charts(overview, comparison)
        _render_feature_importance(overview)


# Maintain backwards compatibility for the existing view import path.
render_model_insights_page = render_model_insights
