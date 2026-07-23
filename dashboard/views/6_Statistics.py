"""Register and render the historical Statistics view for native navigation.

Purpose:
    Connect the declared Statistics route to the validated historical-data
    presentation component.
Responsibility:
    Execute the renderer once without recreating filter, data, or KPI logic in
    the route wrapper.
Inputs:
    Native Streamlit route execution initiated by ``app.py``.
Outputs:
    The rendered historical-statistics page from
    ``components.statistics_dashboard``.
Collaboration:
    Navigation chooses this wrapper; the component and statistics service own
    all filtering, validation, and metric production.
"""

# Import the historical-statistics renderer for the Analytics route.
from components.statistics_dashboard import render_statistics_dashboard

# Render the selected Statistics page.
render_statistics_dashboard()
