"""Define canonical tournament-name substitutions for historical matches.

Purpose:
    Reconcile alternative competition labels from imported football datasets.
Responsibility:
    Expose a data-only mapping used by the tournament-standardization step.
Inputs:
    Raw or cleaned tournament labels from international match records.
Outputs:
    :data:`TOURNAMENT_MAPPING` for deterministic label replacement.
Interactions:
    ``standardize_tournaments.py`` consumes this mapping before integrity
    validation and competition-strength feature generation.
"""

TOURNAMENT_MAPPING = {
    # ----------------------------
    # FIFA
    # ----------------------------
    "World Cup": "FIFA World Cup",
    "Fifa World Cup": "FIFA World Cup",
    "World Championship": "FIFA World Cup",
    "World Cup Qualification": "FIFA World Cup Qualification",
    "Fifa World Cup Qualification": "FIFA World Cup Qualification",
    # ----------------------------
    # UEFA
    # ----------------------------
    "European Championship": "UEFA European Championship",
    "Euro": "UEFA European Championship",
    "Uefa Euro": "UEFA European Championship",
    "Uefa Nations League": "UEFA Nations League",
    # ----------------------------
    # South America
    # ----------------------------
    "Copa América": "Copa America",
    "Copa America": "Copa America",
    # ----------------------------
    # Africa
    # ----------------------------
    "African Cup Of Nations": "Africa Cup of Nations",
    "Africa Cup Of Nations": "Africa Cup of Nations",
    "Afcon": "Africa Cup of Nations",
    # ----------------------------
    # Asia
    # ----------------------------
    "Asian Cup": "AFC Asian Cup",
    "Afc Asian Cup": "AFC Asian Cup",
    # ----------------------------
    # North America
    # ----------------------------
    "Gold Cup": "CONCACAF Gold Cup",
    "Concacaf Gold Cup": "CONCACAF Gold Cup",
    # ----------------------------
    # Oceania
    # ----------------------------
    "Ofc Nations Cup": "OFC Nations Cup",
}
