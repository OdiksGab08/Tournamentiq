"""Provide the preserved backend namespace for TournamentIQ.

Purpose:
    Group data, feature, model, simulation, and compatibility modules under a
    stable import root.
Responsibility:
    Define package structure only; runtime behavior belongs to child modules.
Inputs:
    None.
Outputs:
    A stable ``src`` package namespace for application and script imports.
Interactions:
    Dashboard services and offline workflows import active model and simulator
    implementations from this namespace.
"""
