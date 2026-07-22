"""Retain historical pipeline import paths without duplicating active workflows.

Purpose:
    Preserve the package namespace used by earlier orchestration entry points.
Responsibility:
    Provide compatibility organization only; no pipeline is executed here.
Inputs:
    None.
Outputs:
    Stable ``src.pipelines`` imports for compatible callers.
Interactions:
    Active data, training, prediction, and simulation orchestration resides in
    ``src.features``, ``src.models``, ``src.simulator``, and dashboard services.
"""
