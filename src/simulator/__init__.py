"""Run trained-model inference and tournament simulations for TournamentIQ.

Purpose:
    Group runtime components that turn persisted model artifacts and historical
    snapshots into match, tournament, and Monte Carlo predictions.
Responsibility:
    Provide package organization; individual modules own data snapshots,
    feature assembly, probability sampling, and tournament progression.
Inputs:
    Saved models, preprocessors, processed historical data, and configured
    tournament fields.
Outputs:
    Structured match probabilities and simulated tournament outcomes.
Interactions:
    Dashboard services cache and invoke these modules without duplicating model
    or simulation logic.
"""
