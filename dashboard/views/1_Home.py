"""Register and render the TournamentIQ Home view for native navigation.

Purpose:
    Bridge the Home route declared in ``navigation`` to its presentation
    renderer without adding page-local application logic.
Responsibility:
    Import and execute the Home component exactly once when Streamlit selects
    this registered view.
Inputs:
    Native Streamlit route execution initiated by ``app.py``.
Outputs:
    The fully rendered Home page produced by ``components.home``.
Collaboration:
    Exists only as a route wrapper; state, inference, and metrics remain owned
    by the shared Home component and its services.
"""

# Import the shared Home renderer; this thin route wrapper contains no page logic.
from components.home import render_home_page

# Render the Home page selected by native Streamlit navigation.
render_home_page()
