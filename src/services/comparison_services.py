"""Provide the retained backend abstraction for team-snapshot comparisons.

Purpose:
    Present two team snapshots in one comparison-oriented payload.
Responsibility:
    Delegate snapshot retrieval to the simulator layer and preserve the legacy
    ``team1``/``team2`` response shape.
Inputs:
    Two canonical national-team names.
Outputs:
    A mapping containing one snapshot for each requested team.
Interactions:
    The production dashboard uses ``dashboard.services.team_comparison_service``
    for richer comparison behavior; this module preserves the backend boundary.
"""

from src.simulator.live_snapshot import LiveSnapshot


class ComparisonService:
    """Expose the retained two-team snapshot comparison interface.

    Args:
        None.

    Notes:
        Snapshot construction is delegated to the simulator provider so callers
        receive the same underlying historical state used by compatible flows.
    """

    def __init__(self):

        self.snapshot = LiveSnapshot()

    def compare(self, team1: str, team2: str) -> dict[str, object]:
        """Return both requested team snapshots in the legacy payload shape.

        Args:
            team1: First canonical team name.
            team2: Second canonical team name.

        Returns:
            A dictionary containing ``team1`` and ``team2`` snapshot values.

        Notes:
            The service intentionally does not calculate a winner or synthetic
            comparison score; presentation layers interpret the raw snapshots.
        """

        a = self.snapshot.get_snapshot(team1)

        b = self.snapshot.get_snapshot(team2)

        return {
            "team1": a,
            "team2": b,
        }
