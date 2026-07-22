"""Aggregate repeated real tournament paths into Monte Carlo outcome estimates.

Purpose:
    Estimate champion and stage probabilities by repeatedly invoking the same
    fixture engine and trained-model sampling rules used for one tournament.
Responsibility:
    Validate run configuration, aggregate genuine tournament outcomes, expose a
    legacy champion DataFrame, and provide structured dashboard diagnostics.
Inputs:
    Simulation count, optional seed or cached predictor, and the configured
    :class:`TournamentEngine` field.
Outputs:
    Champion-frequency DataFrames or detailed count, stage, and convergence
    payloads derived from repeated tournament executions.
Interactions:
    Dashboard Monte Carlo services call this class while preserving its legacy
    ``run`` API for existing Home-page and offline consumers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd

from src.simulator.tournament_engine import TournamentEngine


ProgressCallback = Callable[[int, int], None]


class MonteCarloSimulator:
    """Run repeated configured tournaments and aggregate empirical outcomes.

    Args:
        simulations: Positive number of tournament executions to aggregate.
        predictor: Optional cached predictor passed to a created engine.
        seed: Optional random seed for deterministic sampling paths.
        engine: Optional preconfigured tournament engine to reuse.

    Notes:
        One engine is retained for the run so fixture prediction caching and the
        saved model are reused across simulated tournaments.
    """

    def __init__(
        self,
        simulations: int = 10000,
        predictor: Any | None = None,
        seed: int | None = None,
        engine: TournamentEngine | None = None,
    ) -> None:
        """Create a validated simulator around one reusable tournament engine.

        Args:
            simulations: Positive number of tournaments to execute.
            predictor: Optional trained predictor shared with a created engine.
            seed: Optional seed passed to the created engine's sampler.
            engine: Optional engine that replaces automatic engine creation.

        Returns:
            None.

        Raises:
            ValueError: If ``simulations`` is not a positive integer.

        Notes:
            Optional predictor and seed inputs preserve established callers while
            allowing dashboard services to avoid repetitive model loading.
        """
        if isinstance(simulations, bool):
            raise ValueError("simulations must be a positive integer")
        try:
            self.simulations = int(simulations)
        except (TypeError, ValueError) as error:
            raise ValueError("simulations must be a positive integer") from error
        if self.simulations <= 0:
            raise ValueError("simulations must be a positive integer")

        self.seed = seed
        self.engine = (
            engine
            if engine is not None
            else TournamentEngine(predictor=predictor, seed=seed)
        )

    def run(self) -> pd.DataFrame:
        """Return the legacy champion-frequency table for repeated tournaments.

        Args:
            None.

        Returns:
            A descending DataFrame with ``Team``, ``Titles``, and percentage
            ``Probability`` columns.

        Notes:
            This contract is intentionally preserved for existing callers that
            consume a compact champion-only result instead of detailed stages.
        """
        champions: Counter[str] = Counter()

        print()
        print("=" * 70)
        print("RUNNING MONTE CARLO SIMULATION")
        print("=" * 70)

        for index in range(self.simulations):
            champion = self.engine.simulate()
            champions[champion] += 1

            if (index + 1) % 100 == 0:
                print(f"{index + 1}/{self.simulations} completed")

        result = pd.DataFrame(champions.items(), columns=["Team", "Titles"])
        result["Probability"] = result["Titles"] / self.simulations * 100
        return result.sort_values("Probability", ascending=False)

    def run_detailed(
        self,
        *,
        checkpoints: Iterable[int] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Aggregate champion, stage, and convergence counts from detailed runs.

        Args:
            checkpoints: Optional positive cumulative run counts at which to
                capture champion-count snapshots.
            progress_callback: Optional callback receiving completed and total
                simulation counts after each tournament.

        Returns:
            A structured payload of title counts, stage counts, convergence
            snapshots, configuration metadata, and sampling details.

        Notes:
            Checkpoints are snapshots from one cumulative run, not independent
            reruns; the legacy :meth:`run` output remains unchanged.
        """
        checkpoint_set = self._validate_checkpoints(checkpoints)
        configuration = self.engine.tournament_configuration()
        teams = [str(team) for team in configuration["teams"]]
        stage_names = [str(round_name) for round_name in configuration["rounds"]]
        opening_fixtures = [
            (str(home), str(away))
            for home, away in configuration["quarter_final_fixtures"]
        ]
        champion_counts: Counter[str] = Counter({team: 0 for team in teams})
        stage_counts: dict[str, Counter[str]] = {
            stage: Counter({team: 0 for team in teams}) for stage in stage_names
        }
        convergence: list[dict[str, Any]] = []

        for index in range(self.simulations):
            tournament = self.engine.simulate_detailed()
            self._validate_detailed_tournament(
                tournament, teams, stage_names, opening_fixtures
            )

            # Count round participation, not only wins, because advancement
            # probabilities describe reaching each stage in the real bracket.
            for round_data in tournament["rounds"]:
                stage = str(round_data["name"])
                for match in round_data["matches"]:
                    stage_counts[stage][str(match["home_team"])] += 1
                    stage_counts[stage][str(match["away_team"])] += 1

            champion_counts[str(tournament["champion"])] += 1
            completed = index + 1
            if completed in checkpoint_set:
                convergence.append(
                    {
                        "completed_simulations": completed,
                        "champion_counts": {
                            team: int(champion_counts[team]) for team in teams
                        },
                    }
                )
            if progress_callback is not None:
                progress_callback(completed, self.simulations)

        return {
            "simulation_count": self.simulations,
            "seed": self.seed,
            "tournament_name": tournament["tournament_name"],
            "format": tournament["format"],
            "teams": teams,
            "champion_counts": {team: int(champion_counts[team]) for team in teams},
            "stage_counts": {
                stage: {team: int(counts[team]) for team in teams}
                for stage, counts in stage_counts.items()
            },
            "convergence": convergence or None,
            "draw_resolution": tournament.get("draw_resolution"),
            "scores_generated": tournament.get("scores_generated", False),
        }

    def _validate_checkpoints(self, checkpoints: Iterable[int] | None) -> set[int]:
        """Validate requested cumulative checkpoints against this run's size."""
        if checkpoints is None:
            return set()
        validated: set[int] = set()
        for checkpoint in checkpoints:
            if isinstance(checkpoint, bool):
                raise ValueError("checkpoints must contain positive integer run counts")
            try:
                count = int(checkpoint)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "checkpoints must contain positive integer run counts"
                ) from error
            if count <= 0 or count > self.simulations:
                raise ValueError(
                    "checkpoints must be between 1 and the simulation count"
                )
            validated.add(count)
        return validated

    @staticmethod
    def _validate_detailed_tournament(
        tournament: dict[str, Any],
        teams: list[str],
        stage_names: list[str],
        opening_fixtures: list[tuple[str, str]],
    ) -> None:
        """Reject incomplete fixtures or invalid advancement before aggregation."""
        if not isinstance(tournament, dict):
            raise ValueError(
                "tournament engine returned an unsupported detailed result"
            )
        returned_teams = tournament.get("teams")
        if not isinstance(returned_teams, list) or returned_teams != teams:
            raise ValueError("tournament engine returned a different team field")
        rounds = tournament.get("rounds")
        if (
            not isinstance(rounds, list)
            or len(rounds) != len(stage_names)
            or any(
                not isinstance(round_data, dict) or round_data.get("name") != stage_name
                for round_data, stage_name in zip(rounds, stage_names, strict=True)
            )
        ):
            raise ValueError("tournament engine returned malformed round data")

        expected_fixtures = opening_fixtures
        final_winner: str | None = None
        for round_data, stage_name in zip(rounds, stage_names, strict=True):
            matches = round_data.get("matches")
            if not isinstance(matches, list) or len(matches) != len(expected_fixtures):
                raise ValueError(
                    f"tournament engine returned incomplete {stage_name} fixtures"
                )

            winners: list[str] = []
            for match, expected_fixture in zip(matches, expected_fixtures, strict=True):
                if not isinstance(match, dict):
                    raise ValueError(
                        f"tournament engine returned malformed {stage_name} match data"
                    )
                home = match.get("home_team")
                away = match.get("away_team")
                winner = match.get("winner")
                if (home, away) != expected_fixture:
                    raise ValueError(
                        f"tournament engine returned invalid {stage_name} fixture order"
                    )
                if home not in teams or away not in teams or home == away:
                    raise ValueError(
                        f"tournament engine returned invalid {stage_name} teams"
                    )
                if winner not in (home, away):
                    raise ValueError(
                        f"tournament engine returned an invalid {stage_name} winner"
                    )
                winners.append(str(winner))

            if len(
                {team for fixture in expected_fixtures for team in fixture}
            ) != 2 * len(expected_fixtures):
                raise ValueError(
                    f"tournament engine returned duplicate {stage_name} participants"
                )
            if len(winners) == 1:
                final_winner = winners[0]
                expected_fixtures = []
            else:
                expected_fixtures = [
                    (winners[index], winners[index + 1])
                    for index in range(0, len(winners), 2)
                ]

        if tournament.get("champion") != final_winner:
            raise ValueError(
                "tournament engine champion does not match the final winner"
            )
