"""Expose dashboard-facing adapters around retained project capabilities.

Purpose:
    Mark the service package that translates existing data, model, and
    simulation capabilities into UI-safe dashboard contracts.
Responsibility:
    Provide a stable package boundary; individual modules own all loading,
    validation, caching, and result normalization behavior.
Inputs:
    Imported service modules and their existing local project artifacts.
Outputs:
    No direct runtime output; importable service namespaces for components.
Collaboration:
    Dashboard components call these adapters instead of reaching directly into
    the ML pipeline, simulator, or raw data layout.
"""
