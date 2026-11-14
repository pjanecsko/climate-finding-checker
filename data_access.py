from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from contracts import Plan

CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache"
FIRST_YEAR, LAST_YEAR = 1981, 2025


def _cache_path(variable: str, place: str) -> Path:
    slug = "_".join(place.lower().split())
    return CACHE_DIR / f"{slug}_{variable.lower()}.csv"


def _synthetic_monthly_series(variable_name: str, start: str = "1981-01-01", end: str = "2025-12-01", place: str = "Milan", lat: float = 45.46, lon: float = 9.19) -> pd.Series:
    """Deterministic demo data. This is deliberately not represented as an observation."""
    dates = pd.date_range(start=pd.Timestamp(start).to_period("M").to_timestamp(), end=pd.Timestamp(end).to_period("M").to_timestamp(), freq="MS")
    if dates.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], name="date"), name=variable_name)
    month = dates.month.to_numpy()
    year = dates.year.to_numpy()
    configs = {
        "MERRA2_T2M": (15.0, 8.0, 0.035, 0.8, 0.20),
        "GPM_PRECIP": (65.0, 28.0, -0.15, 14.0, 3.5),
        "GLDAS_SSM": (180.0, 30.0, -0.08, 9.0, 4.0),
    }
    if variable_name not in configs:
        raise ValueError(f"Unsupported variable: {variable_name}")
    base, amplitude, annual_trend, cycle_amp, noise_amp = configs[variable_name]
    elapsed_years = year - FIRST_YEAR
    # Small deterministic cell effect so neighbouring-cell challenges are not identical.
    cell_factor = ((round(lat * 100) * 31 + round(lon * 100) * 17) % 101 - 50) / 500.0
    seasonal = amplitude * np.sin((month - 1) * 2 * np.pi / 12 + 0.8)
    interannual = cycle_amp * np.cos(elapsed_years / 2.0 + cell_factor)
    noise = noise_amp * np.sin(elapsed_years * 1.7 + month * 0.91 + cell_factor)
    cell_shift = cell_factor * (0.25 if variable_name == "MERRA2_T2M" else 2.0)
    values = base + seasonal + annual_trend * elapsed_years + interannual + noise + cell_shift
    if variable_name == "GPM_PRECIP":
        values = np.clip(values, 5, None)
    elif variable_name == "GLDAS_SSM":
        values = np.clip(values, 50, None)
    series = pd.Series(values.astype(float), index=dates, name=variable_name)
    series.index.name = "date"
    return series


def load_full_series(variable_name: str, place: str = "Milan", lat: float = 45.46, lon: float = 9.19) -> pd.Series:
    """Load committed/local cache if present; otherwise generate deterministic demo data."""
    path = _cache_path(variable_name, place)
    if path.exists():
        try:
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
            if variable_name in frame.columns:
                series = frame[variable_name]
            elif len(frame.columns):
                series = frame.iloc[:, 0]
            else:
                raise ValueError("Empty cache")
            series.index = pd.DatetimeIndex(series.index).to_period("M").to_timestamp()
            return series.sort_index().rename(variable_name).astype(float)
        except (OSError, ValueError, IndexError, pd.errors.ParserError):
            pass
    return _synthetic_monthly_series(variable_name, f"{FIRST_YEAR}-01-01", f"{LAST_YEAR}-12-01", place, lat, lon)


def load_series_for_plan(plan: Plan, variable_name: str | None = None, *, full_record: bool = False) -> pd.Series:
    variable = variable_name or (plan.variables[0] if plan.variables else "MERRA2_T2M")
    full = load_full_series(variable, plan.place, plan.lat, plan.lon)
    if full_record:
        return full
    start = pd.Timestamp(plan.start).to_period("M").to_timestamp()
    end = pd.Timestamp(plan.end).to_period("M").to_timestamp()
    requested_index = pd.date_range(start, end, freq="MS")
    # Reindex first so absent months remain explicitly missing rather than disappearing.
    return full.reindex(requested_index).rename(variable)


def load_plan_metadata(plan: Plan) -> dict:
    return {
        "dataset": "deterministic synthetic demo (not NASA data)",
        "version": "demo-v1",
        "place": plan.place,
        "requested_coordinates": {"lat": plan.lat, "lon": plan.lon},
        "resolved_cell_coordinates": {"lat": plan.lat, "lon": plan.lon},
        "retrieval_time": pd.Timestamp.now(tz="UTC").isoformat(),
        "units": "variable-specific demo units",
    }
