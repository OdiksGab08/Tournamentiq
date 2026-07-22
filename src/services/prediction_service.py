"""Provide the retained service abstraction for match-prediction consumers.

Purpose:
    Delegate match inference to the simulator predictor and annotate returned
    probabilities with a simple confidence label.
Responsibility:
    Construct the predictor once per service instance, forward matchup inputs,
    and add display-oriented confidence metadata to its result.
Inputs:
    Home/away team names plus retained tournament and neutral-venue arguments.
Outputs:
    Predictor output augmented with ``confidence`` and ``confidence_level``.
Interactions:
    Production dashboard inference uses ``dashboard.services.match_prediction_service``;
    this class remains available for compatible backend integrations.
"""

from src.simulator.predictor import Predictor


class PredictionService:
    """Expose the retained predictor-backed service interface.

    Args:
        None.

    Notes:
        The underlying simulator predictor owns feature construction and saved
        model loading; this service only adds consumer-facing metadata.
    """

    def __init__(self):

        self.predictor = Predictor()

    def predict_match(
        self,
        home_team,
        away_team,
        tournament="FIFA World Cup",
        neutral=True,
    ):
        """Predict a matchup and classify the highest returned probability.

        Args:
            home_team: Team passed as the home-side model input.
            away_team: Team passed as the away-side model input.
            tournament: Retained tournament context argument for compatible
                callers.
            neutral: Retained neutral-venue argument for compatible callers.

        Returns:
            Predictor result enriched with numeric confidence and a qualitative
            confidence-level label.

        Notes:
            Confidence is a presentation aid derived from the maximum returned
            outcome probability; it does not alter the model probabilities.
        """

        # ``Predictor`` now owns a fixed neutral World Cup feature context.
        # Keep the legacy parameters in this adapter's public signature so
        # existing integrations do not break, but do not forward arguments the
        # production predictor no longer accepts.
        del tournament, neutral
        result = self.predictor.predict(home_team, away_team)

        confidence = max(
            result["home_probability"],
            result["draw_probability"],
            result["away_probability"],
        )

        if confidence >= 0.80:
            level = "Very High"
        elif confidence >= 0.65:
            level = "High"
        elif confidence >= 0.50:
            level = "Medium"
        else:
            level = "Low"

        result["confidence"] = confidence
        result["confidence_level"] = level

        return result
