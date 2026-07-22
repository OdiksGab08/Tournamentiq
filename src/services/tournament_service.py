"""Preserve the legacy tournament service module path.

Purpose:
    Keep compatibility for callers importing the historical backend service.
Responsibility:
    Act as a documented placeholder; it does not execute tournament simulations.
Inputs:
    None.
Outputs:
    No runtime output.
Interactions:
    Active tournament execution lives in ``src.simulator.tournament_engine`` and
    ``dashboard.services.tournament_simulation_service``.
"""
