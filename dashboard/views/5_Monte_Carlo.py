"""Register and render the Monte Carlo analysis view for native navigation.

Purpose:
    Connect the declared Monte Carlo route to the component that presents real
    repeated-tournament analysis.
Responsibility:
    Execute the route renderer once and avoid duplicating simulator controls,
    session state, or result handling in this wrapper.
Inputs:
    Native Streamlit route execution initiated by ``app.py``.
Outputs:
    The Monte Carlo interface from ``components.monte_carlo_analysis``.
Collaboration:
    Route selection remains in ``navigation``; simulation work stays in the
    component and its dedicated service adapter.
"""

from components.monte_carlo_analysis import render_monte_carlo_analysis_page


render_monte_carlo_analysis_page()
