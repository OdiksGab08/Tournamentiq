"""Retain shared model-development configuration values for compatible workflows.

Purpose:
    Centralize historical training, validation, calibration, and ensemble
    defaults in an importable configuration module.
Responsibility:
    Define constants only; this module does not load data, fit models, or write
    artifacts.
Inputs:
    None. Values are source-controlled configuration settings.
Outputs:
    Constants for model-development code that imports this configuration path.
Interactions:
    The active model factory currently uses ``src.models.model_config``; this
    broader configuration module is preserved for compatible training scripts.
"""

# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

RANDOM_STATE = 42

# --------------------------------------------------
# Data Split
# --------------------------------------------------

TRAIN_RATIO = 0.80

VALIDATION_RATIO = 0.10

TEST_RATIO = 0.10

# --------------------------------------------------
# Cross Validation
# --------------------------------------------------

CV_FOLDS = 5

SCORING = "f1_weighted"

# --------------------------------------------------
# Logistic Regression
# --------------------------------------------------

LOGISTIC_MAX_ITER = 5000

# --------------------------------------------------
# Tree Models
# --------------------------------------------------

N_ESTIMATORS = 300

MAX_DEPTH = None

MIN_SAMPLES_SPLIT = 2

MIN_SAMPLES_LEAF = 1

# --------------------------------------------------
# XGBoost
# --------------------------------------------------

XGB_EVAL_METRIC = "mlogloss"

XGB_TREE_METHOD = "hist"

# --------------------------------------------------
# Probability Calibration
# --------------------------------------------------

CALIBRATION_METHOD = "isotonic"

# --------------------------------------------------
# Ensemble
# --------------------------------------------------

TOP_MODELS = 3

VOTING = "soft"
