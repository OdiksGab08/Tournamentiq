"""Define the shared training hyperparameters for the model factory.

Purpose:
    Keep reproducibility and estimator-size defaults in one importable module.
Responsibility:
    Expose constants only; no model, data, or filesystem work occurs here.
Inputs:
    None. Values are source-controlled configuration defaults.
Outputs:
    Constants consumed by :func:`src.models.models.get_models`.
Interactions:
    The trainer creates candidate estimators from these values and saves their
    fitted artifacts for the dashboard predictor.
"""

RANDOM_STATE = 42

N_ESTIMATORS = 300

MAX_DEPTH = None

MAX_ITER = 5000

N_JOBS = -1
