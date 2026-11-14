SUPPORTED_VARIABLES = {
    "MERRA2_T2M": {
        "label": "MERRA-2 temperature at 2m",
        "theme": "temperature",
        "unit": "K",
        "place": "Milan",
        "description": "Monthly near-surface air temperature from the MERRA-2 reanalysis product.",
    },
    "GPM_PRECIP": {
        "label": "GPM precipitation",
        "theme": "precipitation",
        "unit": "mm/month",
        "place": "Po Valley",
        "description": "Monthly precipitation from the GPM satellite-era product.",
    },
    "GLDAS_SSM": {
        "label": "GLDAS soil moisture",
        "theme": "soil moisture",
        "unit": "kg/m^2",
        "place": "Po Valley",
        "description": "Monthly root-zone soil moisture from GLDAS.",
    },
}

DEFAULT_VARIABLES = {
    "temperature": ["MERRA2_T2M"],
    "precipitation": ["GPM_PRECIP"],
    "soil moisture": ["GLDAS_SSM"],
}


def get_variable_meta(variable_name: str):
    return SUPPORTED_VARIABLES.get(variable_name)


def supported_theme(theme_name: str):
    return theme_name.lower() in {"temperature", "precipitation", "soil moisture"}
