"""Sample knockout winners from model-derived team probabilities.

Purpose:
    Convert a binary knockout probability distribution into repeatable sampled
    match winners for tournament and Monte Carlo simulations.
Responsibility:
    Optionally seed Python's random generator, normalize provided probabilities,
    and draw one team without creating scoreline data.
Inputs:
    Two team names and their draw-adjusted knockout probabilities.
Outputs:
    The selected winner's team name.
Interactions:
    ``TournamentEngine`` uses this sampler after the production ``Predictor``
    evaluates each fixture.
"""

# Use Python's pseudo-random generator to sample a winner from model probabilities.
import random


class ProbabilitySimulator:
    """Sample binary knockout outcomes from model probabilities.

    Args:
        seed: Optional random seed for reproducible tournament paths.

    Notes:
        The module-level random generator preserves the established simulation
        behavior used by legacy Monte Carlo callers.
    """

    def __init__(self, seed=None):
        """Initialize the optional reproducibility seed.

        Args:
            seed: Optional integer passed to Python's random generator.

        Returns:
            None.

        Notes:
            A ``None`` seed intentionally leaves generator state unchanged for
            non-deterministic simulation runs.
        """

        # Seed only when requested so users can reproduce a simulation path.
        if seed is not None:
            random.seed(seed)

    def choose(
        self,
        home_team,
        away_team,
        home_probability,
        away_probability,
    ):
        """Randomly select one knockout winner from the supplied probabilities.

        Args:
            home_team: Team represented by ``home_probability``.
            away_team: Team represented by ``away_probability``.
            home_probability: Draw-adjusted probability of a home-team advance.
            away_probability: Draw-adjusted probability of an away-team advance.

        Returns:
            The selected winning team name.

        Notes:
            Inputs are normalized before sampling so callers can supply weights;
            a non-positive total falls back to an unbiased random choice instead
            of fabricating a deterministic winner.
        """

        # Normalize the two weights before comparing them with one random draw.
        total = home_probability + away_probability

        # Preserve a valid simulation path for malformed zero-weight inputs
        # without pretending either team has evidence-backed superiority.
        if total <= 0:
            return random.choice(
                [
                    home_team,
                    away_team,
                ]
            )

        home_probability /= total
        away_probability /= total

        # Draw a uniform value in [0, 1) to select one advancing team.
        random_number = random.random()

        if random_number < home_probability:
            return home_team

        return away_team
