"""Declare the fixed opening field for the supported knockout simulation.

Purpose:
    Store the ordered quarter-final fixtures used by the legacy simulator.
Responsibility:
    Provide configuration data only; this module neither predicts nor samples
    match outcomes.
Inputs:
    None. Fixtures are source-controlled tournament configuration.
Outputs:
    :data:`QUARTER_FINALS`, an ordered list of team-pair tuples.
Interactions:
    Tournament-engine configuration mirrors this field when constructing the
    detailed knockout flow consumed by dashboard simulation services.
"""

QUARTER_FINALS = [
    ("France", "Morocco"),
    ("Spain", "Belgium"),
    ("Norway", "England"),
    ("Argentina", "Switzerland"),
]
