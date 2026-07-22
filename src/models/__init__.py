"""Train, evaluate, rank, and package TournamentIQ classification models.

Purpose:
    Organize the model-development stages that turn engineered historical
    features into saved inference artifacts.
Responsibility:
    Provide package structure; individual modules own preparation, splitting,
    preprocessing, training, evaluation, and comparison.
Inputs:
    Processed feature datasets produced by ``src.features``.
Outputs:
    Saved preprocessors, trained estimators, and model-ranking artifacts under
    ``models/``.
Interactions:
    The dashboard and simulator load ``best_model.pkl`` and ``preprocessor.pkl``
    through the runtime predictor.
"""
