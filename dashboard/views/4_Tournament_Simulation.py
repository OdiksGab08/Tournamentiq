"""Register and render the Tournament Simulation view for native navigation.

Purpose:
    Connect the declared tournament-simulation route to the existing engine's
    Streamlit presentation component.
Responsibility:
    Execute the renderer once without adding bracket, predictor, or simulation
    behavior to the thin route wrapper.
Inputs:
    Native Streamlit route execution initiated by ``app.py``.
Outputs:
    The rendered tournament-simulation page from
    ``components.tournament_simulation``.
Collaboration:
    Delegates all visible work to the component, which in turn uses the
    validated simulation service.
"""

from components.tournament_simulation import render_tournament_simulation_page


render_tournament_simulation_page()
