from __future__ import annotations

import numpy as np
import pandas as pd

from contracts import Plan
from data_access import load_series_for_plan, load_plan_metadata
from catalogue import get_variable_meta


def _month_index_for(series: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(series.index)


def monthly_coverage(series: pd.Series) -> dict:
    monthly = series.sort_index()
    if monthly.empty:
        return {"months_present": 0.0, "full_years": 0, "months_per_year": {} }

    dates = _month_index_for(monthly)
    # Keep a strict monthly index for the available period.
    full_months = pd.date_range(start=dates.min().strftime("%Y-%m-01"), end=dates.max().strftime("%Y-%m-01"), freq="MS")
    month_count = len(full_months)
    observed_months = monthly.reindex(full_months).notna().sum()
    years = sorted(set(d.year for d in dates))
    months_per_year = {}
    for year in years:
        subset = monthly[monthly.index.year == year]
        months_per_year[year] = int(subset.notna().sum())

    return {
        "months_present": float(observed_months / max(month_count, 1)),
        "full_years": len(years),
        "months_per_year": months_per_year,
    }


def climatology_for(series: pd.Series) -> pd.Series:
    months = series.index.month
    values = series.groupby(months).mean()
    return values.sort_index()


def anomaly_series(series: pd.Series) -> pd.Series:
    clim = climatology_for(series)
    anomalies = series.copy()
    anomalies = anomalies - clim[series.index.month].to_numpy()
    return anomalies


def series_coverage_ok(series: pd.Series) -> tuple[bool, dict]:
    cov = monthly_coverage(series)
    total_years = max(cov["full_years"], 1)
    years_with_full_months = 0
    for months in cov["months_per_year"].values():
        if months >= 12:
            years_with_full_months += 1

    years_ok = total_years >= 10
    month_ok = cov["months_present"] >= 0.80
    month_by_month_ok = True
    for month in range(1, 13):
        cells = series[series.index.month == month]
        if cells.empty:
            month_by_month_ok = False
            break
        present_fraction = cells.notna().mean()
        if present_fraction < 0.70:
            month_by_month_ok = False
            break

    ok = years_ok and month_ok and month_by_month_ok
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


def _newey_west_pvalue(x: np.ndarray, y: np.ndarray, beta: float, residuals: np.ndarray, lag: int | None = None) -> float:
    n = len(y)
    if lag is None:
        lag = max(1, min(12, n // 5))
    x_center = x - x.mean()
    sxx = float(np.sum(x_center ** 2))
    if sxx == 0:
        return 1.0
    base_var = float(np.sum(residuals ** 2) / max(n - 2, 1))
    variance = (base_var / sxx)
    for k in range(1, lag + 1):
        w = (1 - (k / (lag + 1)))
        cov = np.sum(residuals[:-k] * residuals[k:] * x_center[:-k] * x_center[k:]) / max(n - k, 1)
        variance += 2 * w * cov / sxx
    std_err = np.sqrt(max(variance, 1e-12))
    z = abs(beta) / std_err
    import scipy.stats as st
    return float(2 * (1 - st.norm.cdf(z)))


def trend_analysis(plan: Plan, variable_name: str | None = None, series: pd.Series | None = None) -> dict:
    target = variable_name or (plan.variables[0] if plan.variables else "MERRA2_T2M")
    if series is None:
        series = load_series_for_plan(plan, target)
    series = series.sort_index()

    # Use a fixed climatology, full-record for the selected series.
    anomalies = anomaly_series(series)
    clean = anomalies.dropna().sort_index()
    if clean.empty:
        raise ValueError("No valid monthly data available for trend calculation.")

    years = np.array([p.year + (p.month - 1) / 12.0 for p in clean.index])
    y = clean.to_numpy(dtype=float)
    x = years - years.min()
    beta0, beta1 = np.polyfit(x, y, 1)
    pred = beta0 + beta1 * x
    residuals = y - pred
    n = len(y)
    r1 = 0.0 if n < 2 else float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
    if np.isnan(r1):
        r1 = 0.0
    n_eff = max(1.0, n * (1 - r1) / (1 + r1))

    sxx = np.sum((x - x.mean()) ** 2)
    if sxx == 0:
        raise ValueError("Time axis is degenerate; cannot compute trend.")

    rss = np.sum(residuals ** 2)
    dof = max(n - 2, 1)
    sigma2 = rss / dof
    se_beta = np.sqrt(sigma2 / sxx)
    t_stat = beta1 / se_beta
    from scipy import stats

    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=dof)))
    t_crit = stats.t.ppf(0.975, df=dof)
    slope_per_decade = 10.0 * beta1
    ci_low = 10.0 * (beta1 - t_crit * se_beta)
    ci_high = 10.0 * (beta1 + t_crit * se_beta)

    if p_value < 0.05 and slope_per_decade > 0:
        verdict = "significant increase"
    elif p_value < 0.05 and slope_per_decade < 0:
        verdict = "significant decrease"
    else:
        verdict = "not enough evidence"

    cov_stats = series_coverage_ok(series)
    coverage = float(cov_stats[1]["months_present"])
    hic = {"selection": target, "slope_per_decade": slope_per_decade, "p": p_value}

    # Rough Weatherhead-style estimate, clearly labeled as rough.
    years_needed = 10.0
    if abs(slope_per_decade) > 1e-12:
        years_needed = max(10.0, abs(1.96 ** 2 / slope_per_decade))
    # Keep the output intentionally rough rather than pretending to solve autocorrelation.
    warnings = [
        "This uses a rough AR(1) effective-sample-size adjustment, not a full autocorrelation model.",
        "The years-needed estimate is a rough approximation that assumes the current rate and AR(1) noise.",
    ]

    sen = seasonal_sen_slope(anomalies)
    hac_p = _newey_west_pvalue(x, y, beta1, residuals)

    return {
        "slope_per_decade": float(slope_per_decade),
        "ci": (float(ci_low), float(ci_high)),
        "p": float(p_value),
        "verdict": verdict,
        "n": int(n),
        "n_eff": float(n_eff),
        "r1": float(r1),
        "coverage": float(coverage),
        "years_needed": float(years_needed),
        "sen_per_decade": float(sen),
        "hac_p": float(hac_p),
        "settings": {
            "series": target,
            "method": "OLS on monthly anomalies",
            "climatology": "full-record calendar-month mean",
            "years": plan.start,
            "end": plan.end,
        },
        "warnings": warnings,
    }


def validate_plan(plan: Plan) -> dict:
    if not plan.variables:
        return {"status": "incomplete", "reason": "No variable was selected.", "alternative": "Choose a supported climate variable."}

    for variable in plan.variables:
        if variable not in {"MERRA2_T2M", "GPM_PRECIP", "GLDAS_SSM"}:
            return {
                "status": "unsupported",
                "reason": f"Variable '{variable}' is not in the supported catalogue.",
                "alternative": "Use a supported variable such as MERRA2_T2M or GPM_PRECIP.",
            }

    if plan.kind not in {"trend", "two_place"}:
        return {"status": "unsupported", "reason": "This investigation type is not in the catalogue.", "alternative": "Choose a trend analysis."}

    if plan.start > plan.end:
        return {"status": "incomplete", "reason": "The date range is invalid.", "alternative": "Choose a start date earlier than the end date."}

    series = load_series_for_plan(plan, plan.variables[0])
    ok, cov = series_coverage_ok(series)
    if not ok:
        return {
            "status": "incomplete",
            "reason": "The selected date window fails the coverage rule.",
            "alternative": "Use a longer period or a more complete window.",
        }

    return {"status": "supported", "reason": "Coverage and variable are acceptable.", "alternative": ""}
