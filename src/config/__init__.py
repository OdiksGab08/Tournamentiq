"""Expose configuration namespaces retained for project-wide training settings.

Purpose:
    Group source-controlled configuration values used by historical training and
    simulation workflows.
Responsibility:
    Provide package organization only; individual configuration modules define
    constants without performing I/O or model work.
Inputs:
    None. Values are maintained in Python modules.
Outputs:
    Importable configuration namespaces.
Interactions:
    Active estimator settings are consumed by ``src.models``; simulator defaults
    remain available to compatible offline workflows.
"""
