from __future__ import annotations

import numpy as np
import pandas as pd


def climatology_for(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    month_key = series.index.month
    return series.groupby(month_key).mean().sort_index()


def anomaly_series(series: pd.Series) -> pd.Series:
    clean = series.sort_index().dropna()
    if clean.empty:
        return pd.Series(dtype=float)
    climatology = climatology_for(clean)
    month_values = climatology.loc[clean.index.month].to_numpy()
    return clean - month_values


def monthly_coverage(series: pd.Series) -> dict:
    if series.empty:
        return {"months_present": 0.0, "full_years": 0, "months_per_year": {}}
    clean = series.sort_index().dropna()
    if clean.empty:
        return {"months_present": 0.0, "full_years": 0, "months_per_year": {}}
    start = clean.index.min().to_period("M").to_timestamp()
    end = clean.index.max().to_period("M").to_timestamp()
    full_months = pd.date_range(start=start, end=end, freq="MS")
    total = len(full_months)
    observed = clean.reindex(full_months).notna().sum()
    years = {}
    for year in range(start.year, end.year + 1):
        subset = clean[clean.index.year == year]
        years[str(year)] = int(subset.notna().sum())
    return {"months_present": float(observed / max(total, 1)), "full_years": len(years), "months_per_year": years}


def series_coverage_ok(series: pd.Series) -> tuple[bool, dict]:
    cov = monthly_coverage(series)
    months_present = cov["months_present"]
    full_years = cov["full_years"]
    month_ok = True
    for month in range(1, 13):
        subset = series[series.index.month == month]
        if subset.empty:
            month_ok = False
            break
        if subset.notna().mean() < 0.70:
            month_ok = False
            break
    ok = full_years >= 10 and months_present >= 0.80 and month_ok
    return ok, cov


def seasonal_sen_slope(series: pd.Series) -> float:
    clean = series.dropna().sort_index()
    if len(clean) < 2:
        return 0.0
    x = np.arange(len(clean), dtype=float) / 12.0
    y = clean.to_numpy(dtype=float)
    slopes = []
    for i in range(len(y) - 1):
        for j in range(i + 1, len(y)):
            denom = x[j] - x[i]
            if denom == 0:
                continue
            slopes.append((y[j] - y[i]) / denom)
    if not slopes:
        return 0.0
    return float(np.median(slopes)) * 10.0
