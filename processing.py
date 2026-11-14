from __future__ import annotations

import numpy as np
import pandas as pd


def climatology_for(series: pd.Series) -> pd.Series:
    clean = series.dropna().sort_index()
    if clean.empty:
        return pd.Series(dtype=float)
    return clean.groupby(clean.index.month).mean().sort_index()


def anomaly_series(series: pd.Series, climatology: pd.Series | None = None) -> pd.Series:
    ordered = series.sort_index().astype(float)
    if climatology is None:
        climatology = climatology_for(ordered)
    if ordered.empty or climatology.empty:
        return pd.Series(index=ordered.index, dtype=float, name=series.name)
    month_means = ordered.index.month.map(climatology)
    return pd.Series(ordered.to_numpy() - np.asarray(month_means, dtype=float), index=ordered.index, name=series.name)


def monthly_coverage(series: pd.Series) -> dict:
    if series.empty:
        return {"months_present": 0.0, "full_years": 0, "months_per_year": {}}
    start = series.index.min().to_period("M").to_timestamp()
    end = series.index.max().to_period("M").to_timestamp()
    expected = pd.date_range(start, end, freq="MS")
    present = series.reindex(expected).notna()
    total_years = end.year - start.year + 1
    by_year = {}
    for year in range(start.year, end.year + 1):
        year_index = expected[expected.year == year]
        by_year[str(year)] = int(series.reindex(year_index).notna().sum())
    return {
        "months_present": float(present.sum() / len(expected)) if len(expected) else 0.0,
        "full_years": total_years,
        "months_per_year": by_year,
    }


def series_coverage_ok(series: pd.Series, *, threshold: float = 0.80, month_threshold: float = 0.70, minimum_years: int = 10) -> tuple[bool, dict]:
    cov = monthly_coverage(series)
    if series.empty:
        return False, cov
    start = series.index.min().to_period("M").to_timestamp()
    end = series.index.max().to_period("M").to_timestamp()
    expected = pd.date_range(start, end, freq="MS")
    month_fractions = {}
    for month in range(1, 13):
        expected_months = expected[expected.month == month]
        present = series.reindex(expected_months).notna().sum()
        month_fractions[month] = float(present / len(expected_months)) if len(expected_months) else 0.0
    cov["calendar_month_coverage"] = month_fractions
    ok = (
        cov["full_years"] >= minimum_years
        and cov["months_present"] >= threshold
        and all(value >= month_threshold for value in month_fractions.values())
    )
    return ok, cov


def seasonal_sen_slope(series: pd.Series) -> float:
    """Median within-calendar-month pairwise slope, expressed per decade."""
    clean = series.dropna().sort_index()
    slopes = []
    for month in range(1, 13):
        subset = clean[clean.index.month == month]
        years = subset.index.year.to_numpy(dtype=float)
        values = subset.to_numpy(dtype=float)
        for i in range(len(values) - 1):
            for j in range(i + 1, len(values)):
                if years[j] != years[i]:
                    slopes.append((values[j] - values[i]) / (years[j] - years[i]))
    return float(np.median(slopes) * 10.0) if slopes else 0.0
