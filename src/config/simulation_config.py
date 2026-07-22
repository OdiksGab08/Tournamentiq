"""Declare retained defaults for compatible tournament-simulation workflows.

Purpose:
    Keep source-controlled simulation settings available at their historical
    import path.
Responsibility:
    Define constants only; no simulations are executed by this module.
Inputs:
    None. Values are maintained directly in source.
Outputs:
    Simulation configuration constants for offline or compatible callers.
Interactions:
    Active dashboard simulations use their validated service configuration and
    runtime engine contracts; these values remain available for legacy imports.
"""

SIMULATIONS = 10000

USE_MONTE_CARLO = True

PENALTY_SHOOTOUT = True

EXTRA_TIME = True

HOME_ADVANTAGE = False
