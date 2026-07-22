"""Build leakage-safe historical features for model training and live inference.

Purpose:
    Group feature-engineering modules used to derive team, form, head-to-head,
    tournament, and competition signals.
Responsibility:
    Provide package-level organization only; individual modules own calculations
    and artifact persistence.
Inputs:
    Chronologically ordered historical match tables.
Outputs:
    Feature DataFrames consumed by the training pipeline and simulator snapshot
    builders.
Interactions:
    ``feature_pipeline`` combines package outputs into the final training
    dataset used by ``src.models`` and the runtime predictor.
"""
