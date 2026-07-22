"""Preserve the legacy simulation-pipeline module path.

Purpose:
    Retain compatible imports without duplicating tournament or Monte Carlo
    orchestration.
Responsibility:
    Serve as a documented placeholder; it performs no simulation work.
Inputs:
    None.
Outputs:
    No runtime output.
Interactions:
    Active simulations are implemented by ``src.simulator`` and invoked by the
    dashboard tournament and Monte Carlo service adapters.
"""
