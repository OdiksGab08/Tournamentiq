"""Preserve the legacy Monte Carlo service module path.

Purpose:
    Keep compatibility for callers importing the historical backend service.
Responsibility:
    Act as a documented placeholder; it does not execute simulations.
Inputs:
    None.
Outputs:
    No runtime output.
Interactions:
    Active Monte Carlo execution lives in ``src.simulator.monte_carlo`` and
    ``dashboard.services.monte_carlo_service``.
"""
