"""Retain the legacy prediction namespace for compatible imports.

Purpose:
    Preserve package structure for historical prediction entry points.
Responsibility:
    Provide namespace compatibility only; no inference logic is implemented
    directly in this package.
Inputs:
    None.
Outputs:
    A stable ``src.prediction`` import path.
Interactions:
    Current production inference lives in ``src.simulator.predictor`` and its
    dashboard service adapters.
"""
