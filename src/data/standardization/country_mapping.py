"""Define canonical country-name substitutions for historical match records.

Purpose:
    Resolve common source-specific aliases for national teams and host countries.
Responsibility:
    Provide data-only mappings; this module performs no I/O or transformation.
Inputs:
    Country labels emitted by the raw international-results source.
Outputs:
    :data:`COUNTRY_MAPPING`, consumed by the country-standardization stage.
Interactions:
    ``standardize_countries.py`` applies these replacements before feature
    generation and live team lookup use canonical team names.
"""

COUNTRY_MAPPING = {
    # USA
    "Usa": "United States",
    "United States Of America": "United States",
    # South Korea
    "Korea Republic": "South Korea",
    "Republic Of Korea": "South Korea",
    # North Korea
    "Dpr Korea": "North Korea",
    "Korea Dpr": "North Korea",
    # Iran
    "Ir Iran": "Iran",
    # Ivory Coast
    "Côte D'ivoire": "Ivory Coast",
    "Cote D'ivoire": "Ivory Coast",
    # Czechia
    "Czech Republic": "Czechia",
    # Russia
    "Russian Federation": "Russia",
    # China
    "China Pr": "China",
    # UAE
    "Uae": "United Arab Emirates",
    # DR Congo
    "Dr Congo": "Congo DR",
    "Congo Dr": "Congo DR",
    # Republic of the Congo
    "Congo": "Congo Republic",
    # Cape Verde
    "Cape Verde Islands": "Cape Verde",
    # Eswatini
    "Swaziland": "Eswatini",
    # North Macedonia
    "Macedonia": "North Macedonia",
    # Myanmar
    "Burma": "Myanmar",
    # Timor-Leste
    "East Timor": "Timor-Leste",
    # Vietnam
    "Vietnam Republic": "Vietnam",
    # Curaçao
    "Curacao": "Curaçao",
}
