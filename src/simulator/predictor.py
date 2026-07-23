"""Load production artifacts and generate match-outcome probabilities.

Purpose:
    Provide the lowest-level runtime adapter between live feature construction,
    the persisted preprocessor, and the selected trained classifier.
Responsibility:
    Load production artifacts once, transform one matchup feature row, and
    expose both raw three-outcome and draw-adjusted knockout probabilities.
Inputs:
    ``models/best_model.pkl``, ``models/preprocessor.pkl``, and two canonical
    team names supplied at prediction time.
Outputs:
    A dictionary of home, draw, away, and knockout-oriented probabilities.
Interactions:
    Dashboard match, tournament, and Monte Carlo services reuse this class via
    cached adapters rather than reimplementing model inference.
"""

# Load the serialized scikit-learn estimator and preprocessing pipeline.
import joblib

# Reuse deployment-safe path resolution and artifact logging for local and Cloud runs.
from src.config.deployment import find_project_root, log_artifact
# Build the exact feature columns expected by the saved preprocessor.
from src.simulator.feature_builder import FeatureBuilder

# Resolve the repository root and production-model directory independently of CWD.
ROOT = find_project_root(__file__)

MODEL_DIR = ROOT / "models"


class Predictor:
    """Run production match inference using saved model artifacts.

    Args:
        None.

    Notes:
        Initialization deliberately loads artifacts once; dashboard services
        cache the resulting instance across Streamlit reruns.
    """

    def __init__(self):

        # Log model loading so deployment logs reveal which Git-LFS artifact is used.
        print("Loading model...")

        self.model = joblib.load(
            log_artifact(MODEL_DIR / "best_model.pkl", label="production model")
        )

        # Load the matching transformer after the estimator so raw features use training schema.
        print("Loading preprocessor...")

        self.preprocessor = joblib.load(
            log_artifact(MODEL_DIR / "preprocessor.pkl", label="preprocessor")
        )

        # Create one reusable feature builder for all predictions made by this predictor.
        self.builder = FeatureBuilder()

        print("Predictor Ready.")

    def predict(
        self,
        home_team,
        away_team,
    ):
        """Predict a neutral World Cup matchup using the saved classifier.

        Args:
            home_team: Team assigned to home-oriented model features.
            away_team: Team assigned to away-oriented model features.

        Returns:
            A dictionary containing raw home/draw/away probabilities and
            draw-adjusted ``home_probability``/``away_probability`` values for
            knockout simulation.

        Notes:
            Raw three-class probabilities remain available for the Match
            Predictor UI, while knockout callers split draws evenly because the
            probability simulator needs a binary winner distribution.
        """

        # Build the one-row raw feature table for the selected teams.
        features = self.builder.build(
            home_team,
            away_team,
        )

        # Apply the saved training transformations before calling the classifier.
        X = self.preprocessor.transform(features)

        # Request the model's three-class home/draw/away probability vector.
        probabilities = self.model.predict_proba(X)[0]

        home_probability = float(probabilities[0])
        draw_probability = float(probabilities[1])
        away_probability = float(probabilities[2])

        # Knockout callers require a binary winner distribution; retaining the
        # raw draw value separately preserves the classifier's true output.
        home_knockout = home_probability + draw_probability / 2
        away_knockout = away_probability + draw_probability / 2

        return {
            "home": home_team,
            "away": away_team,
            # Preserve the raw three-outcome model output for callers that need
            # to display a match probability distribution. Existing knockout
            # consumers continue to use the draw-adjusted probabilities below.
            "home_win_probability": home_probability,
            "draw_probability": draw_probability,
            "away_win_probability": away_probability,
            "home_probability": home_knockout,
            "away_probability": away_knockout,
        }
