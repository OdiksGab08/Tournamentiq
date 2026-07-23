"""Render one real knockout-tournament simulation in the dashboard.

Purpose:
    Expose the existing tournament engine through a clear, stateful Streamlit
    page without modifying bracket or predictor behavior.
Responsibility:
    Collect allowed controls, show preflight/progress/results, and preserve
    session-local simulation context for the current dashboard user.
Inputs:
    User seed choices, Streamlit state, and normalized records from
    ``services.tournament_simulation_service``.
Outputs:
    Tournament bracket, round summaries, and the simulated champion.
Collaboration:
    Called by the Tournament Simulation view and delegates all engine access to
    the tournament-simulation service.
"""

from __future__ import annotations

import traceback
from html import escape
import re
from typing import Any, Mapping, Sequence

import streamlit as st

from components.team_selector import render_team_flag
from services.tournament_simulation_service import (
    TournamentSimulationError,
    get_tournament_overview,
    is_tournament_result_current,
    parse_simulation_seed,
    run_tournament_simulation,
    tournament_signature,
    validate_tournament_preflight,
)
from ui import (
    animated_container,
    apply_theme,
    glass_card,
    gradient_button,
    metric_card,
    page_header,
    section_title,
)

_DRAW_RESOLUTION_LABELS = {
    "draw_probability_split_evenly_then_sampled": (
        "The raw draw probability is split evenly into both teams’ knockout "
        "advancement probabilities, then ProbabilitySimulator samples the advancing team."
    ),
}


def _initialize_tournament_state() -> None:
    """Initialize page-specific state without storing models or source tables."""
    defaults: dict[str, Any] = {
        "tournament_simulation_result": None,
        "tournament_simulation_error": None,
        "tournament_simulation_signature": None,
        "tournament_seed": "",
        "tournament_simulation_running": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _safe_key(value: str) -> str:
    """Create an HTML/CSS-safe Streamlit key from dynamic backend labels."""
    return re.sub(r"[^a-z0-9-]+", "-", value.casefold()).strip("-") or "item"


def _format_probability(value: Any, *, decimals: int = 1) -> str:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    return f"{probability:.{decimals}%}"


def _resolution_text(value: Any) -> str | None:
    key = str(value or "").strip()
    return _DRAW_RESOLUTION_LABELS.get(key, key or None)


def _render_tournament_styles() -> None:
    """Inject only Tournament Simulation-specific visual and responsive CSS."""
    st.markdown(
        """
        <style>
            @keyframes tournament-gradient-flow {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            @keyframes tournament-trophy-float {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-4px); }
            }

            .tournament-setup-caption,
            .tournament-setup-note,
            .tournament-bracket-note,
            .tournament-summary-note,
            .tournament-champion-meta {
                margin: 0;
                color: var(--ui-color-text-secondary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.9rem;
                line-height: 1.55;
            }

            .tournament-setup-caption { margin-bottom: 0.9rem; }
            .tournament-setup-note { margin: 0.85rem 0 1.1rem; font-size: 0.82rem; }

            [class*="st-key-tournament-field-team-"] {
                min-height: 6.5rem;
                padding: 0.8rem;
                border: 1px solid var(--ui-color-border-subtle);
                border-radius: var(--ui-radius-md);
                background: var(--ui-glass-background);
            }

            .tournament-field-team__name {
                margin: 0.5rem 0 0;
                overflow: hidden;
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.84rem;
                font-weight: 750;
                text-overflow: ellipsis;
                white-space: nowrap;
            }

            [class*="st-key-tournament-field-team-"] [data-testid="stImage"] {
                width: fit-content;
                padding: 0.24rem;
                border: 1px solid var(--ui-color-border-subtle);
                border-radius: var(--ui-radius-sm);
                background: rgba(25, 17, 13, 0.5);
            }

            .tournament-bracket-note { margin-bottom: 0.9rem; }

            [class*="st-key-tournament-round-"] {
                min-width: 0;
            }

            .tournament-round-heading {
                margin: 0 0 0.8rem;
                color: var(--ui-color-accent);
                font-family: var(--ui-type-font-mono);
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.12em;
                text-transform: uppercase;
            }

            [class*="st-key-tournament-match-"] {
                position: relative;
                min-width: 0;
                margin-bottom: 0.85rem;
                padding: 0.95rem;
                overflow: hidden;
                border: 1px solid var(--ui-color-border-subtle);
                border-radius: var(--ui-radius-md);
                background: var(--ui-glass-background);
                box-shadow: inset 0 1px 0 rgba(255, 248, 242, 0.045), var(--ui-shadow-sm);
                backdrop-filter: blur(10px) saturate(110%);
                -webkit-backdrop-filter: blur(10px) saturate(110%);
                transition: transform 160ms ease, border-color 160ms ease;
            }

            [class*="st-key-tournament-match-"]::before,
            [class*="st-key-ui-glass-tournament-champion"]::before {
                content: "";
                position: absolute;
                top: 0;
                right: 0;
                left: 0;
                height: 3px;
                background: linear-gradient(90deg, var(--ui-color-primary), var(--ui-color-accent), var(--ui-color-warning), var(--ui-color-primary));
                background-size: 240% 100%;
                animation: tournament-gradient-flow 4.5s linear infinite;
            }

            [class*="st-key-tournament-match-"]:hover {
                transform: translateY(-2px);
                border-color: rgba(214, 151, 71, 0.48);
            }

            .tournament-match-kicker,
            .tournament-match-label,
            .tournament-team-role,
            .tournament-probability-label {
                margin: 0;
                color: var(--ui-color-text-muted);
                font-family: var(--ui-type-font-mono);
                font-size: 0.64rem;
                font-weight: 800;
                letter-spacing: 0.1em;
                text-transform: uppercase;
            }

            .tournament-team-row {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                min-width: 0;
                padding: 0.2rem 0;
            }

            .tournament-team-row__name {
                min-width: 0;
                overflow: hidden;
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.84rem;
                font-weight: 750;
                text-overflow: ellipsis;
                white-space: nowrap;
            }

            .tournament-team-row__state {
                margin-left: auto;
                color: var(--ui-color-accent);
                font-family: var(--ui-type-font-sans);
                font-size: 0.66rem;
                font-weight: 700;
                white-space: nowrap;
            }

            .tournament-match-winner {
                margin: 0.7rem 0 0;
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.77rem;
                line-height: 1.45;
            }

            .tournament-match-winner strong { color: var(--ui-color-accent); }

            .tournament-probability-bar {
                display: flex;
                width: 100%;
                min-height: 0.58rem;
                margin-top: 0.72rem;
                overflow: hidden;
                border: 1px solid var(--ui-color-border-subtle);
                border-radius: 999px;
                background: var(--ui-color-surface-muted);
            }

            .tournament-probability-bar__home { background: var(--ui-color-primary); }
            .tournament-probability-bar__draw { background: var(--ui-color-accent); }
            .tournament-probability-bar__away { background: var(--ui-color-success); }

            .tournament-probability-legend {
                display: flex;
                flex-wrap: wrap;
                gap: 0.4rem 0.65rem;
                margin-top: 0.5rem;
                color: var(--ui-color-text-secondary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.67rem;
            }

            .tournament-match-meta {
                margin: 0.55rem 0 0;
                color: var(--ui-color-text-muted);
                font-family: var(--ui-type-font-sans);
                font-size: 0.69rem;
                line-height: 1.42;
            }

            .tournament-round-summary-title {
                margin: 0 0 0.7rem;
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: 1.03rem;
                font-weight: 800;
            }

            .tournament-round-advancers {
                margin: 0.85rem 0 0;
                color: var(--ui-color-text-secondary);
                font-family: var(--ui-type-font-sans);
                font-size: 0.82rem;
                line-height: 1.5;
            }

            [class*="st-key-ui-glass-tournament-champion"] {
                position: relative;
                overflow: hidden;
                border-color: rgba(227, 161, 59, 0.38);
                background: linear-gradient(135deg, rgba(227, 161, 59, 0.14), rgba(36, 24, 19, 0.82));
            }

            .tournament-trophy {
                display: inline-grid;
                width: 3.25rem;
                height: 3.25rem;
                place-items: center;
                border: 1px solid rgba(240, 185, 103, 0.44);
                border-radius: var(--ui-radius-md);
                background: rgba(227, 161, 59, 0.12);
                font-size: 1.75rem;
                animation: tournament-trophy-float 2.8s ease-in-out infinite;
            }

            .tournament-champion-name {
                margin: 0.45rem 0 0;
                color: var(--ui-color-text-primary);
                font-family: var(--ui-type-font-sans);
                font-size: clamp(1.7rem, 4vw, 3.15rem);
                font-weight: 850;
                letter-spacing: -0.055em;
                line-height: 1;
            }

            .tournament-champion-meta { margin-top: 0.7rem; }

            @media (max-width: 900px) {
                .st-key-tournament-bracket-grid [data-testid="stHorizontalBlock"] {
                    flex-direction: column;
                }

                .st-key-tournament-bracket-grid [data-testid="stColumn"] {
                    width: 100% !important;
                    flex: 1 1 100% !important;
                }
            }

            @media (max-width: 760px) {
                .st-key-tournament-team-grid [data-testid="stHorizontalBlock"],
                .st-key-tournament-round-summary-metrics [data-testid="stHorizontalBlock"],
                .st-key-tournament-champion-grid [data-testid="stHorizontalBlock"] {
                    flex-direction: column;
                }

                .st-key-tournament-team-grid [data-testid="stColumn"],
                .st-key-tournament-round-summary-metrics [data-testid="stColumn"],
                .st-key-tournament-champion-grid [data-testid="stColumn"] {
                    width: 100% !important;
                    flex: 1 1 100% !important;
                }

            }

            @media (prefers-reduced-motion: reduce) {
                [class*="st-key-tournament-match-"]::before,
                [class*="st-key-ui-glass-tournament-champion"]::before,
                .tournament-trophy { animation: none; }

                [class*="st-key-tournament-match-"] { transition: none; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    """Render the page hierarchy before the tournament controls."""
    page_header(
        "Tournament Simulation",
        eyebrow="FIFA World Cup 2026 knockout",
        subtitle=(
            "Run one complete configured tournament using the trained match model "
            "and inspect the sampled knockout path round by round."
        ),
    )


def _render_active_field(teams: Sequence[str]) -> None:
    """Show the configured backend field without presenting non-functional controls."""
    for start in range(0, len(teams), 4):
        with st.container(key=f"tournament-team-grid-{start}", border=False):
            columns = st.columns(4)
            for column, team in zip(columns, teams[start : start + 4]):
                with column:
                    with st.container(
                        key=f"tournament-field-team-{_safe_key(str(team))}",
                        border=False,
                    ):
                        render_team_flag(str(team), width=30)
                        st.markdown(
                            f'<p class="tournament-field-team__name">{escape(str(team))}</p>',
                            unsafe_allow_html=True,
                        )


def _render_setup(overview: Mapping[str, Any]) -> tuple[int | None, bool]:
    """Render only real configuration controls supported by TournamentEngine."""
    configuration = overview["configuration"]
    teams = list(overview["teams"])
    section_title(
        "Tournament setup",
        eyebrow="Active backend configuration",
        description="Review the configured tournament field, set an optional seed, and run the existing simulation.",
        compact=True,
    )
    with glass_card("tournament-setup"):
        st.markdown(
            f'<p class="tournament-setup-caption"><strong>{escape(str(configuration["format"]))}</strong> · {len(teams)} configured teams · neutral World Cup context</p>',
            unsafe_allow_html=True,
        )
        _render_active_field(teams)
        st.markdown(
            """
            <p class="tournament-setup-note">
                Optional seed uses the existing ProbabilitySimulator seed support. Leave it blank for a fresh stochastic knockout run.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.text_input(
            "Optional simulation seed",
            key="tournament_seed",
            placeholder="For example: 2026",
            help="A whole-number seed makes the existing probability sampling reproducible.",
        )
        try:
            seed = parse_simulation_seed(st.session_state.tournament_seed)
        except TournamentSimulationError as error:
            st.error(str(error))
            return None, False
        clicked = gradient_button(
            "Run Tournament Simulation",
            key="tournament-run-button",
            width="content",
            disabled=st.session_state.tournament_simulation_running,
        )
    return seed, clicked


def _run_tournament(seed: int | None, signature: str) -> None:
    """Run the real engine with clear, non-fabricated status updates."""
    st.session_state.tournament_simulation_running = True
    st.session_state.tournament_simulation_error = None
    try:
        with st.status(
            "Preparing real tournament simulation…", expanded=False
        ) as status:
            status.update(
                label="Preparing team snapshots and validating the bracket…",
                state="running",
            )
            validate_tournament_preflight()
            status.update(
                label="Loading the cached trained match predictor…", state="running"
            )
            status.update(
                label="Simulating tournament rounds and building the bracket…",
                state="running",
            )
            result = run_tournament_simulation(seed)
            result["signature"] = signature
            st.session_state.tournament_simulation_result = result
            st.session_state.tournament_simulation_signature = signature
            status.update(label="Tournament simulation complete", state="complete")
    except TournamentSimulationError as error:
        st.session_state.tournament_simulation_result = None
        st.session_state.tournament_simulation_signature = signature
        st.session_state.tournament_simulation_error = str(error)
    except Exception as error:
        st.error(f"{type(error).__name__}: {error}")
        st.code(traceback.format_exc())
        raise
    finally:
        st.session_state.tournament_simulation_running = False


def _team_state(team: str, winner: str) -> str:
    return "Advances" if team.casefold() == winner.casefold() else "Eliminated"


def _render_match_team(team: str, winner: str, *, side: str, match_id: str) -> None:
    """Render one compact, accessible fixture side with a local flag fallback."""
    with st.container(key=f"tournament-match-{match_id}-{side}-flag", border=False):
        render_team_flag(team, width=24)
    st.markdown(
        f"""
        <div class="tournament-team-row">
            <span class="tournament-team-row__name">{escape(team)}</span>
            <span class="tournament-team-row__state">{escape(_team_state(team, winner))}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_probability_bar(match: Mapping[str, Any]) -> None:
    """Render the real three-outcome pre-resolution probability distribution."""
    values = (
        match.get("home_probability"),
        match.get("draw_probability"),
        match.get("away_probability"),
    )
    if any(value is None for value in values):
        st.markdown(
            '<p class="tournament-match-meta">The engine did not expose a three-outcome probability distribution for this fixture.</p>',
            unsafe_allow_html=True,
        )
        return
    home, draw, away = (float(value) for value in values)
    st.markdown(
        f"""
        <p class="tournament-probability-label">Pre-resolution model probabilities</p>
        <div class="tournament-probability-bar" aria-label="Pre-resolution home, draw, and away probabilities">
            <span class="tournament-probability-bar__home" style="width: {home * 100:.2f}%"></span>
            <span class="tournament-probability-bar__draw" style="width: {draw * 100:.2f}%"></span>
            <span class="tournament-probability-bar__away" style="width: {away * 100:.2f}%"></span>
        </div>
        <div class="tournament-probability-legend">
            <span>Home <strong>{_format_probability(home)}</strong></span>
            <span>Draw <strong>{_format_probability(draw)}</strong></span>
            <span>Away <strong>{_format_probability(away)}</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_match_card(match: Mapping[str, Any]) -> None:
    """Render an actual engine fixture, preserving winner and probability semantics."""
    match_id = str(match["match_id"])
    with st.container(key=f"tournament-match-{match_id}", border=False):
        st.markdown(
            f'<p class="tournament-match-kicker">{escape(str(match["round"]))} · Fixture {escape(match_id.rsplit("-", 1)[-1])}</p>',
            unsafe_allow_html=True,
        )
        _render_match_team(
            str(match["home_team"]),
            str(match["winner"]),
            side="home",
            match_id=match_id,
        )
        _render_match_team(
            str(match["away_team"]),
            str(match["winner"]),
            side="away",
            match_id=match_id,
        )
        st.markdown(
            f'<p class="tournament-match-winner">Actual advancing team: <strong>{escape(str(match["winner"]))}</strong></p>',
            unsafe_allow_html=True,
        )
        _render_probability_bar(match)

        meta_parts: list[str] = []
        if (
            match.get("home_advancement_probability") is not None
            and match.get("away_advancement_probability") is not None
        ):
            meta_parts.append(
                "Draw-adjusted advancement: "
                f"{_format_probability(match['home_advancement_probability'])} · "
                f"{_format_probability(match['away_advancement_probability'])}"
            )
        if match.get("score"):
            meta_parts.append(f"Score: {match['score']}")
        resolution = _resolution_text(match.get("resolution"))
        if resolution:
            meta_parts.append(f"Resolution: {resolution}")
        if meta_parts:
            st.markdown(
                f'<p class="tournament-match-meta">{escape(" · ".join(meta_parts))}</p>',
                unsafe_allow_html=True,
            )


def _render_bracket(result: Mapping[str, Any]) -> None:
    section_title(
        "Live tournament bracket",
        eyebrow="One completed engine run",
        description="Fixture order and advancing teams come directly from the configured TournamentEngine flow.",
    )
    st.markdown(
        '<p class="tournament-bracket-note">Probability distributions describe the model before knockout resolution; the advancing team is the actual sampled engine outcome.</p>',
        unsafe_allow_html=True,
    )
    rounds = list(result["rounds"])
    with st.container(key="tournament-bracket-grid", border=False):
        columns = st.columns(len(rounds))
        for column, round_data in zip(columns, rounds):
            with column:
                with st.container(
                    key=f"tournament-round-{_safe_key(str(round_data['name']))}",
                    border=False,
                ):
                    st.markdown(
                        f'<h3 class="tournament-round-heading">{escape(str(round_data["name"]))}</h3>',
                        unsafe_allow_html=True,
                    )
                    for match in round_data["matches"]:
                        _render_match_card(match)


def _summary_value(value: Any, *, percentage: bool = False) -> str:
    if value is None:
        return "Unavailable"
    return _format_probability(value) if percentage else str(value)


def _render_round_summaries(result: Mapping[str, Any]) -> None:
    section_title(
        "Round summaries",
        eyebrow="Derived from this completed run",
        description="An upset is counted only when the sampled advancing team had a strictly lower draw-adjusted advancement probability than its opponent.",
    )
    for summary in result.get("round_summaries", []):
        round_name = str(summary.get("round") or "Round")
        with glass_card(f"tournament-round-summary-{_safe_key(round_name)}"):
            st.markdown(
                f'<h3 class="tournament-round-summary-title">{escape(round_name)}</h3>',
                unsafe_allow_html=True,
            )
            with st.container(
                key=f"tournament-round-summary-metrics-{_safe_key(round_name)}",
                border=False,
            ):
                match_column, favorite_column, upset_column = st.columns(3)
                with match_column:
                    metric_card(
                        "Matches",
                        _summary_value(summary.get("match_count")),
                        caption="Completed fixtures",
                    )
                with favorite_column:
                    metric_card(
                        "Average favorite probability",
                        _summary_value(
                            summary.get("average_favorite_probability"), percentage=True
                        ),
                        caption="Draw-adjusted advancement",
                    )
                with upset_column:
                    metric_card(
                        "Upsets",
                        _summary_value(summary.get("upsets")),
                        caption="Probability-defined only",
                    )
            advancers = ", ".join(
                str(team) for team in summary.get("advancing_teams", [])
            )
            st.markdown(
                f'<p class="tournament-round-advancers"><strong>Advancing teams:</strong> {escape(advancers or "Unavailable")}</p>',
                unsafe_allow_html=True,
            )
            highest = summary.get("highest_confidence")
            closest = summary.get("closest_match")
            details: list[str] = []
            if highest:
                details.append(
                    "Highest confidence: "
                    f"{highest['matchup']} ({highest['favorite_team']} "
                    f"{_format_probability(highest['favorite_probability'])})"
                )
            if closest:
                details.append(f"Closest match: {closest['matchup']}")
            if details:
                st.markdown(
                    f'<p class="tournament-summary-note">{escape(" · ".join(details))}</p>',
                    unsafe_allow_html=True,
                )


def _render_champion(result: Mapping[str, Any]) -> None:
    section_title("Champion reveal", eyebrow="Actual final winner")
    champion = str(result["champion"])
    runner_up = result.get("runner_up")
    with glass_card("tournament-champion"):
        with st.container(key="tournament-champion-grid", border=False):
            flag_column, champion_column = st.columns((0.6, 2.5))
            with flag_column:
                st.markdown(
                    '<span class="tournament-trophy" aria-hidden="true">🏆</span>',
                    unsafe_allow_html=True,
                )
                render_team_flag(champion, width=72)
            with champion_column:
                st.markdown(
                    '<p class="tournament-match-kicker">Simulated champion</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<h2 class="tournament-champion-name">{escape(champion)}</h2>',
                    unsafe_allow_html=True,
                )
                if runner_up:
                    st.markdown(
                        f'<p class="tournament-champion-meta">Runner-up: {escape(str(runner_up))}</p>',
                        unsafe_allow_html=True,
                    )


def render_tournament_simulation_page() -> None:
    """Render Tournament Simulation only from app.py's existing route branch."""
    _initialize_tournament_state()
    apply_theme()
    _render_tournament_styles()
    try:
        overview = get_tournament_overview()
    except TournamentSimulationError as error:
        page_header("Tournament Simulation", eyebrow="FIFA World Cup 2026 knockout")
        st.error(str(error))
        return

    _render_header()
    for warning in overview.get("warnings", []):
        st.warning(str(warning))
    seed, clicked = _render_setup(overview)
    if seed is None and str(st.session_state.tournament_seed).strip():
        return

    try:
        signature = tournament_signature(seed)
    except TournamentSimulationError as error:
        st.error(str(error))
        return

    if st.session_state.tournament_simulation_signature != signature:
        st.session_state.tournament_simulation_error = None

    if clicked:
        _run_tournament(seed, signature)

    if (
        st.session_state.tournament_simulation_error
        and st.session_state.tournament_simulation_signature == signature
    ):
        st.error(st.session_state.tournament_simulation_error)

    saved_result = st.session_state.tournament_simulation_result
    if saved_result and not is_tournament_result_current(saved_result, signature):
        st.info(
            "Tournament setup changed. Run Tournament Simulation to build a current bracket."
        )
        saved_result = None

    if not saved_result:
        return

    with animated_container("tournament-results", animation="fade_up"):
        _render_bracket(saved_result)
        _render_round_summaries(saved_result)
        _render_champion(saved_result)
