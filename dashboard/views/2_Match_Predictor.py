"""Register and render the Match Predictor view for native navigation.

Purpose:
    Bridge the Match Predictor route declared in ``navigation`` to the shared
    real-model prediction workspace.
Responsibility:
    Import and execute the page renderer once when Streamlit selects this view;
    no inference or widget logic is duplicated here.
Inputs:
    Native Streamlit route execution initiated by ``app.py``.
Outputs:
    The Match Predictor interface and canonical results from
    ``components.match_predictor``.
Collaboration:
    Serves only as the route wrapper; the component and service adapter own
    prediction state, validation, and trained-model interaction.
"""

from components.match_predictor import render_match_predictor_page


render_match_predictor_page()
