"""Retain legacy preprocessing import paths for compatible workflows.

Purpose:
    Preserve package organization for older preprocessing entry points.
Responsibility:
    Provide namespace compatibility only; active transformation stages live in
    ``src.data`` and ``src.models.preprocess``.
Inputs:
    None.
Outputs:
    A stable ``src.preprocessing`` import path.
Interactions:
    New workflows should use the active data-cleaning, standardization, and
    model-preprocessor modules rather than duplicate transformations here.
"""
