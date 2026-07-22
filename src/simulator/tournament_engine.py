"""Run the configured 2026-style knockout tournament with the trained model.

Purpose:
    Execute the supported eight-team quarter-final, semi-final, and final flow
    using production match probabilities and binary knockout sampling.
Responsibility:
    Preserve legacy champion-only simulation while exposing the identical real
    fixture progression as structured detail for dashboard consumers.
Inputs:
    Configured fixtures, a trained ``Predictor`` or compatible replacement, and
    an optional probability sampler/seed.
Outputs:
    Champion strings, round winner lists, or structured brackets containing
    actual sampled match records.
Interactions:
    ``MonteCarloSimulator`` repeatedly invokes this engine; dashboard services
    use detailed results without duplicating fixture or sampling logic.
"""

from __future__ import annotations

from typing import Any, Sequence

from src.simulator.predictor import Predictor
from src.simulator.probability_simulator import ProbabilitySimulator


DEFAULT_QUARTER_FINALS: tuple[tuple[str, str], ...] = (
    ("France", "Morocco"),
    ("Spain", "Belgium"),
    ("Norway", "England"),
    ("Argentina", "Switzerland"),
)


class TournamentEngine:
    """Simulate the configured eight-team knockout field with one predictor.

    Args:
        predictor: Optional trained predictor reused for all fixtures.
        seed: Optional random seed used by a created probability sampler.
        simulator: Optional binary probability sampler.

    Notes:
        Fixture predictions are cached per engine because Monte Carlo runs reuse
        immutable model inputs for the same recurring team pairs.
    """

    TOURNAMENT_NAME = "2026 FIFA World Cup Knockout"
    FORMAT_NAME = "8-team knockout"
    DRAW_RESOLUTION = "draw_probability_split_evenly_then_sampled"

    def __init__(
        self,
        predictor: Any | None = None,
        seed: int | None = None,
        simulator: ProbabilitySimulator | None = None,
    ) -> None:
        """Create an engine, optionally reusing caller-managed runtime objects.

        Args:
            predictor: Optional predictor replacing automatic artifact loading.
            seed: Optional seed used when creating a sampler.
            simulator: Optional sampler replacing automatic sampler creation.

        Returns:
            None.

        Notes:
            Reusing a cached predictor keeps dashboard and Monte Carlo runs from
            loading the saved model for each engine or fixture.
        """
        self.predictor = predictor if predictor is not None else Predictor()
        self.seed = seed
        self.simulator = (
            simulator if simulator is not None else ProbabilitySimulator(seed=seed)
        )
        # A Monte Carlo run reuses one engine against immutable snapshots and a
        # fixed trained model. Cache fixture probabilities within that engine so
        # repeated tournament paths do not rebuild identical feature rows.
        self._match_prediction_cache: dict[tuple[str, str], dict[str, Any]] = {}

    @classmethod
    def tournament_configuration(cls) -> dict[str, Any]:
        """Return the fixed backend-supported field and knockout metadata.

        Args:
            None.

        Returns:
            Tournament name, format, rounds, opening fixtures, teams, and
            sampling metadata used by service preflight validation.

        Notes:
            This is the source of truth for supported teams and prevents UI
            layers from claiming a broader fixture configuration than the engine.
        """
        return {
            "tournament_name": cls.TOURNAMENT_NAME,
            "format": cls.FORMAT_NAME,
            "rounds": ("Quarter-finals", "Semi-finals", "Final"),
            "quarter_final_fixtures": list(DEFAULT_QUARTER_FINALS),
            "teams": [team for fixture in DEFAULT_QUARTER_FINALS for team in fixture],
            "team_count": len(DEFAULT_QUARTER_FINALS) * 2,
            "draw_resolution": cls.DRAW_RESOLUTION,
            "scores_generated": False,
            "neutral_venue": True,
        }

    def _simulate_match(self, home: str, away: str) -> dict[str, Any]:
        """Use the existing predictor and probability simulator for one fixture."""
        cache_key = (home.casefold(), away.casefold())
        if cache_key not in self._match_prediction_cache:
            self._match_prediction_cache[cache_key] = dict(
                self.predictor.predict(home, away)
            )
        prediction = self._match_prediction_cache[cache_key]
        home_probability = float(prediction["home_probability"])
        away_probability = float(prediction["away_probability"])
        winner = self.simulator.choose(
            home,
            away,
            home_probability,
            away_probability,
        )
        return {
            "home_team": home,
            "away_team": away,
            "home_probability": home_probability,
            "away_probability": away_probability,
            "home_win_probability": prediction.get("home_win_probability"),
            "draw_probability": prediction.get("draw_probability"),
            "away_win_probability": prediction.get("away_win_probability"),
            "winner": winner,
            "resolution": self.DRAW_RESOLUTION,
        }

    def play_round(self, fixtures: Sequence[tuple[str, str]]) -> list[str]:
        """Play one knockout round and return winners in fixture order.

        Args:
            fixtures: Ordered home/away pairs for the round.

        Returns:
            A list of winning team names in the same order as ``fixtures``.

        Notes:
            Console output is retained for legacy offline runs; structured UI
            consumers use :meth:`play_round_detailed` instead.
        """
        winners: list[str] = []

        print()
        print("=" * 70)
        print("ROUND RESULTS")
        print("=" * 70)

        for home, away in fixtures:
            result = self._simulate_match(home, away)
            winner = str(result["winner"])
            winners.append(winner)

            print()
            print(f"{home} vs {away}")
            print(f"Home Win Probability : {result['home_probability']:.2%}")
            print(f"Away Win Probability : {result['away_probability']:.2%}")
            print()
            print(f"Winner : {winner}")

        return winners

    def play_round_detailed(
        self, fixtures: Sequence[tuple[str, str]], round_name: str
    ) -> list[dict[str, Any]]:
        """Play one round and return real sampled match records in fixture order.

        Args:
            fixtures: Ordered home/away pairs for the round.
            round_name: Human-readable round label added to every record.

        Returns:
            Structured match dictionaries with probabilities, winner, fixture
            order, and draw-resolution metadata.

        Notes:
            Records originate from the same predictor and sampler used by the
            legacy winner-only method, not a parallel simulation path.
        """
        matches: list[dict[str, Any]] = []
        for match_number, (home, away) in enumerate(fixtures, start=1):
            record = self._simulate_match(home, away)
            record["round_name"] = round_name
            record["match_number"] = match_number
            matches.append(record)
        return matches

    def quarter_finals(self) -> list[str]:
        """Run the configured opening quarter-final fixtures.

        Args:
            None.

        Returns:
            Winners of :data:`DEFAULT_QUARTER_FINALS` in fixture order.

        Notes:
            Fixed ordering defines which winners meet in the semi-finals.
        """
        print()
        print("=" * 70)
        print("QUARTER FINALS")
        print("=" * 70)
        return self.play_round(DEFAULT_QUARTER_FINALS)

    def semi_finals(self, winners: Sequence[str]) -> list[str]:
        """Pair quarter-final winners and return semi-final winners.

        Args:
            winners: Four quarter-final winners in configured fixture order.

        Returns:
            Two final-bound team names in semi-final fixture order.

        Notes:
            Positional pairing deliberately preserves the fixed bracket path.
        """
        fixtures = ((winners[0], winners[1]), (winners[2], winners[3]))
        print()
        print("=" * 70)
        print("SEMI FINALS")
        print("=" * 70)
        return self.play_round(fixtures)

    def final(self, finalists: Sequence[str]) -> str:
        """Play the final fixture and return the sampled champion.

        Args:
            finalists: Two final-bound teams in bracket order.

        Returns:
            The champion team name.

        Notes:
            The scalar return value preserves the original tournament API used
            by legacy Monte Carlo aggregation.
        """
        fixtures = ((finalists[0], finalists[1]),)
        print()
        print("=" * 70)
        print("FINAL")
        print("=" * 70)
        champion = self.play_round(fixtures)[0]

        print()
        print("=" * 70)
        print("2026 FIFA WORLD CUP CHAMPION")
        print("=" * 70)
        print()
        print(champion)
        return champion

    def simulate(self) -> str:
        """Run the legacy full knockout flow and return its champion.

        Args:
            None.

        Returns:
            The sampled tournament champion.

        Notes:
            This compatibility method intentionally delegates through the same
            rounds used by detailed dashboard simulations.
        """
        quarter_final_winners = self.quarter_finals()
        semi_final_winners = self.semi_finals(quarter_final_winners)
        return self.final(semi_final_winners)

    def simulate_detailed(self) -> dict[str, Any]:
        """Run the full bracket and return a structured record of real outcomes.

        Args:
            None.

        Returns:
            Tournament metadata, ordered round records, champion, runner-up,
            semi-finalists, and sampling metadata.

        Notes:
            No scoreline, extra-time, or penalty fields are fabricated because
            the underlying predictor and sampler produce only probabilities and
            sampled advancing teams.
        """
        quarter_final_matches = self.play_round_detailed(
            DEFAULT_QUARTER_FINALS, "Quarter-finals"
        )
        quarter_final_winners = [
            str(match["winner"]) for match in quarter_final_matches
        ]

        semi_final_fixtures = (
            (quarter_final_winners[0], quarter_final_winners[1]),
            (quarter_final_winners[2], quarter_final_winners[3]),
        )
        semi_final_matches = self.play_round_detailed(
            semi_final_fixtures, "Semi-finals"
        )
        semi_final_winners = [str(match["winner"]) for match in semi_final_matches]

        final_matches = self.play_round_detailed(
            ((semi_final_winners[0], semi_final_winners[1]),), "Final"
        )
        final_match = final_matches[0]
        champion = str(final_match["winner"])
        runner_up = (
            str(final_match["away_team"])
            if champion == final_match["home_team"]
            else str(final_match["home_team"])
        )

        return {
            "tournament_name": self.TOURNAMENT_NAME,
            "format": self.FORMAT_NAME,
            "teams": [team for fixture in DEFAULT_QUARTER_FINALS for team in fixture],
            "rounds": [
                {"name": "Quarter-finals", "matches": quarter_final_matches},
                {"name": "Semi-finals", "matches": semi_final_matches},
                {"name": "Final", "matches": final_matches},
            ],
            "champion": champion,
            "runner_up": runner_up,
            "semifinalists": quarter_final_winners,
            "draw_resolution": self.DRAW_RESOLUTION,
            "scores_generated": False,
            "seed": self.seed,
        }
