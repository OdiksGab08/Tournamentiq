"""Register and render the Team Comparison view for native navigation.

Purpose:
    Connect the declared Team Comparison route to its data-grounded component.
Responsibility:
    Execute the page renderer once without introducing route, service, or
    comparison logic in the wrapper module.
Inputs:
    Native Streamlit route execution initiated by ``app.py``.
Outputs:
    The rendered Team Comparison page from ``components.team_comparison``.
Collaboration:
    Navigation owns route selection; the component and service own user state
    and all comparison calculations.
"""

# Import the shared comparison renderer so calculations stay outside the route wrapper.
from components.team_comparison import render_team_comparison_page

# Render the Team Comparison page chosen in navigation.
render_team_comparison_page()
