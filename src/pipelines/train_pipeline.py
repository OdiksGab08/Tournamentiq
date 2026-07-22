"""Preserve the legacy training-pipeline module path.

Purpose:
    Keep compatible imports stable while the active training workflow is
    implemented in ``src.models.prepare_data``, ``train_validation_split``, and
    ``trainer``.
Responsibility:
    Act as a namespace-compatible placeholder; it performs no training work.
Inputs:
    None.
Outputs:
    No runtime output.
Interactions:
    Callers requiring model training should use the active ``src.models``
    modules rather than duplicating orchestration here.
"""
