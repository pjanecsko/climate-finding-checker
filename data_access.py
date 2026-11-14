from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from contracts import Plan


CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _synthetic_monthly_series(variable_name: str, start: str, end: str, place: str = "Milan") -> pd.Series:
    dates = pd.date_range(start=start, end=end, freq="MS")
    month = dates.month.to_numpy()
    year = dates.year.to_numpy()
    base = {
        "MERRA2_T2M": 287.0,
        "GPM_PRECIP": 65.0,
        "GLDAS_SSM": 180.0,
    }.get(variable_name, 0.0)

    # Deterministic monthly signal: warming, precipitation shifts, soil drying tendency.
    seasonal = np.sin((month - 1) / 12.0 * 2 * np.pi + 0.8) * {
        "MERRA2_T2M": 8.0,
        "GPM_PRECIP": 28.0,
        "GLDAS_SSM": 30.0,
    }.get(variable_name, 1.0)
    trend = (year - year.min()) * {
        "MERRA2_T2M": 0.02,
        "GPM_PRECIP": -0.2,
        "GLDAS_SSM": -0.1,
    }.get(variable_name, 0.0)
    interannual = np.cos((year - year.min()) / 2.0) * {
        "MERRA2_T2M": 0.8,
        "GPM_PRECIP": 14.0,
        "GLDAS_SSM": 9.0,
    }.get(variable_name, 0.0)
    noise = np.sin((year - year.min()) * 1.7 + month) * {
        "MERRA2_T2M": 0.2,
        "GPM_PRECIP": 3.5,
        "GLDAS_SSM": 4.0,
    }.get(variable_name, 0.0)

    values = base + seasonal + trend + interannual + noise
    if variable_name == "MERRA2_T2M":
        values = values + np.linspace(0, 3, len(values))
    elif variable_name == "GPM_PRECIP":
        values = np.clip(values, 5, None)
    elif variable_name == "GLDAS_SSM":
        values = np.clip(values, 50, None)

    series = pd.Series(values, index=dates, name=variable_name)
    series.index.name = "date"
    return series


def _cache_path(variable_name: str, place: str) -> Path:
    slug = place.lower().replace(" ", "_")
    return CACHE_DIR / f"{slug}_{variable_name.lower()}.csv"


def save_demo_cache(variable_name: str, start: str, end: str, place: str = "Milan") -> pd.Series:
    series = _synthetic_monthly_series(variable_name, start, end, place)
    path = _cache_path(variable_name, place)
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(path)
    return series


def load_cached_series(variable_name: str, place: str = "Milan") -> pd.Series | None:
    path = _cache_path(variable_name, place)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            return None
        series = df.iloc[:, 0].rename(variable_name)
        return series.sort_index()
    except Exception:
        return None


def load_series_for_plan(plan: Plan, variable_name: str | None = None) -> pd.Series:
    target = variable_name or (plan.variables[0] if plan.variables else "MERRA2_T2M")
    cached = load_cached_series(target, plan.place)
    if cached is not None and not cached.empty:
        return cached.loc[(cached.index >= plan.start) & (cached.index <= plan.end)].sort_index()

    # Fall back to synthetic database when NASA data is unavailable.
    series = _synthetic_monthly_series(target, plan.start, plan.end, plan.place)
    save_demo_cache(target, plan.start, plan.end, plan.place)
    return series


def load_plan_metadata(plan: Plan) -> dict:
    return {
        "dataset": "synthetic-demo-cache",
        "version": "demo-v1",
        "place": plan.place,
        "lat": plan.lat,
        "lon": plan.lon,
        "retrieval_time": pd.Timestamp.utcnow().isoformat(),
        "units": "synthetic",
    }


def write_metadata(metadata: dict) -> None:
    path = CACHE_DIR / "meta.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
