"""Focused contracts for the native Streamlit top-navigation migration."""

from pathlib import Path

import pytest

import dashboard.navigation as navigation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_ROOT = PROJECT_ROOT / "dashboard"


def test_registered_routes_are_unique_and_home_is_the_only_default():
    """Verify registered navigation routes are unique with one default home page."""
    assert [route.name for route in navigation.ROUTES] == [
        "Home",
        "Match Predictor",
        "Team Comparison",
        "Tournament Simulation",
        "Monte Carlo",
        "Statistics",
    ]
    assert len({route.title for route in navigation.ROUTES}) == len(navigation.ROUTES)
    assert [route.name for route in navigation.ROUTES if route.default] == ["Home"]
    assert all(route.source.exists() for route in navigation.ROUTES)


def test_route_groups_match_the_native_top_navigation_information_architecture():
    """Verify routes retain the intended native top-navigation group structure."""
    groups = navigation.route_groups()

    assert list(groups) == ["", "Predictions", "Simulations", "Analytics"]
    assert [route.name for route in groups[""]] == ["Home"]
    assert [route.name for route in groups["Predictions"]] == [
        "Match Predictor",
        "Team Comparison",
    ]
    assert [route.name for route in groups["Simulations"]] == [
        "Tournament Simulation",
        "Monte Carlo",
    ]
    assert [route.name for route in groups["Analytics"]] == [
        "Statistics",
    ]


def test_page_registration_uses_each_registered_source_once(monkeypatch):
    """Verify Streamlit page registration uses every configured source exactly once."""
    registered: list[tuple[Path, dict[str, object]]] = []

    def fake_page(source: Path, **kwargs: object) -> dict[str, object]:
        registered.append((source, kwargs))
        return {"source": source, **kwargs}

    monkeypatch.setattr(navigation.st, "Page", fake_page)
    groups = navigation.build_page_groups()

    assert sum(len(pages) for pages in groups.values()) == len(navigation.ROUTES)
    assert [source for source, _ in registered] == [
        route.source for route in navigation.ROUTES
    ]
    assert registered[0][1]["default"] is True
    assert registered[0][1]["url_path"] == ""


def test_route_lookup_and_internal_navigation_use_registered_page_sources(monkeypatch):
    """Verify internal navigation resolves and switches to registered page sources."""
    selected: list[Path] = []
    monkeypatch.setattr(navigation.st, "switch_page", selected.append)

    navigation.navigate_to("Tournament Simulation")

    assert selected == [navigation.resolve_route("Tournament Simulation").source]
    with pytest.raises(navigation.NavigationError, match="Unknown navigation target"):
        navigation.resolve_route("Unknown route")


def test_invalid_internal_navigation_is_reported_without_switching(monkeypatch):
    """Verify unknown navigation targets show an error without changing pages."""
    messages: list[str] = []
    monkeypatch.setattr(navigation.st, "error", messages.append)
    monkeypatch.setattr(
        navigation.st, "switch_page", lambda _: pytest.fail("unexpected switch")
    )

    navigation.navigate_to("Unknown route")

    assert len(messages) == 1
    assert "Unknown navigation target" in messages[0]


def test_legacy_sidebar_and_session_router_are_absent_from_source_files():
    """Verify retired sidebar and session-state routing code is absent from sources."""
    app_source = (DASHBOARD_ROOT / "app.py").read_text(encoding="utf-8")
    assert app_source.count("st.set_page_config(") == 1
    assert "st.session_state.page" not in app_source
    assert "if page ==" not in app_source
    assert not (DASHBOARD_ROOT / "components" / "sidebar.py").exists()

    source_files = [
        path for path in DASHBOARD_ROOT.rglob("*.py") if "__pycache__" not in path.parts
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in source_files)
    assert "st.session_state.page" not in combined
    assert "st.sidebar" not in combined
    assert "render_sidebar" not in combined


def test_registered_view_wrappers_do_not_reconfigure_or_render_navigation():
    """Verify individual page wrappers leave shared configuration to the app shell."""
    for route in navigation.ROUTES:
        source = route.source.read_text(encoding="utf-8")
        assert "st.set_page_config" not in source
        assert "initialize_page" not in source
        assert "st.navigation" not in source
        assert "st.sidebar" not in source
