"""Provide ingestion adapters for raw football-data sources.

Purpose:
    Group source-specific loaders and download helpers used to acquire raw
    historical inputs.
Responsibility:
    Supply package organization only; individual adapters own source paths and
    network behavior.
Inputs:
    Local raw-data files or external source endpoints.
Outputs:
    Raw pandas DataFrames or persisted source exports.
Interactions:
    Data-cleaning modules consume retained raw outputs before standardization and
    feature engineering.
"""
