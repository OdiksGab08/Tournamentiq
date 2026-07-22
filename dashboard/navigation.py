"""Define native Streamlit routes and safe internal navigation helpers.

Purpose:
    Keep the dashboard's page map and cross-page navigation behavior in one
    deterministic module.
Responsibility:
    Register valid view modules, group them for the top navigation bar, and
    resolve intentional route changes without storing route state manually.
Inputs:
    Route declarations, local view paths, and the native Streamlit navigation
    APIs available in the current runtime.
Outputs:
    ``st.Page`` registrations, grouped route mappings, and safe page switches.
Collaboration:
    ``app.py`` consumes the grouped registrations; page components use
    ``navigate_to`` rather than importing paths or managing navigation state.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping

import streamlit as st


DASHBOARD_ROOT: Final = Path(__file__).resolve().parent
VIEWS_ROOT: Final = DASHBOARD_ROOT / "views"

HOME: Final = "Home"
MATCH_PREDICTOR: Final = "Match Predictor"
TEAM_COMPARISON: Final = "Team Comparison"
TOURNAMENT_SIMULATION: Final = "Tournament Simulation"
MONTE_CARLO: Final = "Monte Carlo"
STATISTICS: Final = "Statistics"


class NavigationError(ValueError):
    """Raised when a route cannot be registered or navigated to safely."""


@dataclass(frozen=True, slots=True)
class RouteSpec:
    """The immutable registration contract for one dashboard page."""

    name: str
    source: Path
    title: str
    icon: str
    group: str
    url_path: str
    default: bool = False


ROUTES: Final[tuple[RouteSpec, ...]] = (
    RouteSpec(
        name=HOME,
        source=VIEWS_ROOT / "1_Home.py",
        title=HOME,
        icon=":material/home:",
        group="",
        url_path="",
        default=True,
    ),
    RouteSpec(
        name=MATCH_PREDICTOR,
        source=VIEWS_ROOT / "2_Match_Predictor.py",
        title=MATCH_PREDICTOR,
        icon=":material/sports_soccer:",
        group="Predictions",
        url_path="match-predictor",
    ),
    RouteSpec(
        name=TEAM_COMPARISON,
        source=VIEWS_ROOT / "3_Team_Comparison.py",
        title=TEAM_COMPARISON,
        icon=":material/compare_arrows:",
        group="Predictions",
        url_path="team-comparison",
    ),
    RouteSpec(
        name=TOURNAMENT_SIMULATION,
        source=VIEWS_ROOT / "4_Tournament_Simulation.py",
        title=TOURNAMENT_SIMULATION,
        icon=":material/emoji_events:",
        group="Simulations",
        url_path="tournament-simulation",
    ),
    RouteSpec(
        name=MONTE_CARLO,
        source=VIEWS_ROOT / "5_Monte_Carlo.py",
        title=MONTE_CARLO,
        icon=":material/casino:",
        group="Simulations",
        url_path="monte-carlo",
    ),
    RouteSpec(
        name=STATISTICS,
        source=VIEWS_ROOT / "6_Statistics.py",
        title=STATISTICS,
        icon=":material/monitoring:",
        group="Analytics",
        url_path="statistics",
    ),
)
ROUTES_BY_NAME: Final[Mapping[str, RouteSpec]] = {route.name: route for route in ROUTES}


def ensure_native_navigation_supported() -> None:
    """Fail clearly if this Streamlit runtime lacks the required native APIs."""
    missing = [
        name
        for name in ("navigation", "Page", "switch_page")
        if not callable(getattr(st, name, None))
    ]
    if missing:
        raise NavigationError(
            "This dashboard requires Streamlit's native top navigation API. "
            f"Unavailable API: {', '.join(missing)}."
        )


def resolve_route(route_name: str) -> RouteSpec:
    """Return a registered route or raise a concise invalid-route error."""
    normalized = route_name.strip() if isinstance(route_name, str) else ""
    route = ROUTES_BY_NAME.get(normalized)
    if route is None:
        available = ", ".join(route.name for route in ROUTES)
        raise NavigationError(
            f"Unknown navigation target {normalized!r}. Available routes: {available}."
        )
    return route


def route_groups() -> OrderedDict[str, tuple[RouteSpec, ...]]:
    """Expose the deterministic grouped route structure without Streamlit state."""
    groups: OrderedDict[str, list[RouteSpec]] = OrderedDict(
        (("", []), ("Predictions", []), ("Simulations", []), ("Analytics", []))
    )
    for route in ROUTES:
        groups[route.group].append(route)
    return OrderedDict((name, tuple(routes)) for name, routes in groups.items())


def _make_page(route: RouteSpec) -> Any:
    """Create one page with a text-only fallback for unsupported icon syntax."""
    if not route.source.exists():
        raise NavigationError(f"The registered {route.title} page module is missing.")
    page_kwargs = {
        "title": route.title,
        "url_path": route.url_path,
        "default": route.default,
    }
    try:
        return st.Page(route.source, icon=route.icon, **page_kwargs)
    except (TypeError, ValueError):
        return st.Page(route.source, **page_kwargs)


def build_page_groups() -> dict[str, list[object]]:
    """Build the grouped native top-navigation structure from one route table."""
    ensure_native_navigation_supported()
    return {
        group: [_make_page(route) for route in routes]
        for group, routes in route_groups().items()
    }


def navigate_to(route_name: str) -> None:
    """Switch to a registered page without storing route state in session state."""
    try:
        route = resolve_route(route_name)
    except NavigationError as error:
        st.error(str(error))
        return
    st.switch_page(route.source)


__all__ = [
    "HOME",
    "MATCH_PREDICTOR",
    "MONTE_CARLO",
    "NavigationError",
    "ROUTES",
    "ROUTES_BY_NAME",
    "STATISTICS",
    "TEAM_COMPARISON",
    "TOURNAMENT_SIMULATION",
    "build_page_groups",
    "ensure_native_navigation_supported",
    "navigate_to",
    "resolve_route",
    "route_groups",
]
