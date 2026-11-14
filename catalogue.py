SUPPORTED_VARIABLES = {
    "MERRA2_T2M": {
        "label": "MERRA-2 temperature at 2 m",
        "theme": "temperature",
        "unit": "°C anomaly",
        "place": "Milan",
        "description": "Monthly near-surface air temperature. Demo values are synthetic, not NASA observations.",
    },
    "GPM_PRECIP": {
        "label": "GPM precipitation",
        "theme": "precipitation",
        "unit": "mm/month",
        "place": "Po Valley",
        "description": "Monthly precipitation. Demo values are synthetic, not NASA observations.",
    },
    "GLDAS_SSM": {
        "label": "GLDAS soil moisture",
        "theme": "soil moisture",
        "unit": "kg/m²",
        "place": "Po Valley",
        "description": "Monthly soil moisture. Demo values are synthetic, not NASA observations.",
    },
}
DEFAULT_VARIABLES = {
    "temperature": ["MERRA2_T2M"],
    "precipitation": ["GPM_PRECIP"],
    "soil moisture": ["GLDAS_SSM"],
}


def get_variable_meta(variable_name: str):
    return SUPPORTED_VARIABLES.get(variable_name)


def supported_theme(theme_name: str) -> bool:
    return theme_name.lower() in DEFAULT_VARIABLES
