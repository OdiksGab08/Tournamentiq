"""Retain service-layer namespaces for compatible backend integrations.

Purpose:
    Group thin backend service abstractions around simulator capabilities.
Responsibility:
    Provide package organization only; active dashboard services live under
    ``dashboard.services`` and call the simulator directly.
Inputs:
    Canonical team names, model artifacts, and simulator outputs supplied by
    callers of individual services.
Outputs:
    Service-level prediction or comparison payloads where implementations exist.
Interactions:
    This package remains available for legacy imports while the production UI
    uses the dashboard-specific adapters.
"""
