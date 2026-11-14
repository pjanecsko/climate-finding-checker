from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from contracts import TrendResult, Validation
from data_access import load_series_for_plan
from processing import anomaly_series, seasonal_sen_slope, series_coverage_ok


def validate_plan(plan):
    if not plan.variables:
        return Validation("incomplete", "No variable was selected.", "Choose a supported climate variable.")

    supported = {"MERRA2_T2M", "GPM_PRECIP", "GLDAS_SSM"}
    for variable in plan.variables:
        if variable not in supported:
            return Validation(
                "unsupported",
                f"Variable '{variable}' is not in the supported catalogue.",
                "Choose MERRA2_T2M, GPM_PRECIP or GLDAS_SSM.",
            )
    if plan.start > plan.end:
        return Validation("incomplete", "The date range is invalid.", "Choose a start date earlier than the end date.")

    series = load_series_for_plan(plan, plan.variables[0])
    ok, _ = series_coverage_ok(series)
    if not ok:
        return Validation("incomplete", "The selected date window fails the coverage rule.", "Try a longer period with more complete monthly coverage.")

    return Validation("supported", "Coverage and variable are acceptable.")


def _newey_west_pvalue(x: np.ndarray, y: np.ndarray, beta: float, residuals: np.ndarray, lag: int | None = None) -> float:
    n = len(y)
    if lag is None:
        lag = max(1, min(12, n // 5))
    x_center = x - x.mean()
    sxx = float(np.sum(x_center ** 2))
    if sxx == 0:
        return 1.0
    base_var = float(np.sum(residuals ** 2) / max(n - 2, 1))
    variance = base_var / sxx
    for k in range(1, lag + 1):
        w = 1 - (k / (lag + 1))
        cov = np.sum(residuals[:-k] * residuals[k:] * x_center[:-k] * x_center[k:]) / max(n - k, 1)
        variance += 2 * w * cov / sxx
    std_err = np.sqrt(max(variance, 1e-12))
    z = abs(beta) / std_err
    return float(2 * (1 - stats.norm.cdf(z)))


def trend_result_from_series(plan, variable_name: str, series: pd.Series) -> TrendResult:
    anomalies = anomaly_series(series)
    clean = anomalies.dropna().sort_index()
    if clean.empty:
        raise ValueError("No usable monthly anomalies are available.")

    years = np.array([(p.year + (p.month - 1) / 12.0) for p in clean.index], dtype=float)
    x = years - years.min()
    y = clean.to_numpy(dtype=float)
    beta0, beta1 = np.polyfit(x, y, 1)
    pred = beta0 + beta1 * x
    residuals = y - pred
    n = len(y)
    r1 = 0.0 if n < 2 else float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
    if np.isnan(r1):
        r1 = 0.0
    n_eff = max(1.0, n * (1 - r1) / (1 + r1))

    sxx = np.sum((x - x.mean()) ** 2)
    sigma2 = np.sum(residuals ** 2) / max(n - 2, 1)
    se_beta = np.sqrt(sigma2 / sxx)
    t_stat = beta1 / se_beta
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=max(n - 2, 1))))
    t_crit = stats.t.ppf(0.975, df=max(n - 2, 1))
    slope_per_decade = 10.0 * beta1
    ci_low = 10.0 * (beta1 - t_crit * se_beta)
    ci_high = 10.0 * (beta1 + t_crit * se_beta)
    coverage = series_coverage_ok(series)[1]["months_present"]
    if p_value < 0.05 and slope_per_decade > 0:
        verdict = "significant increase"
    elif p_value < 0.05 and slope_per_decade < 0:
        verdict = "significant decrease"
    else:
        verdict = "not enough evidence"

    years_needed = 10.0 if abs(slope_per_decade) < 1e-12 else max(10.0, abs(1.96 ** 2 / slope_per_decade))
    sen_slope = seasonal_sen_slope(anomalies)
    warnings = [
        "Rough AR(1) effective-sample-size adjustment used.",
        "The years-needed estimate is a rough approximation based on the current rate and AR(1) noise.",
    ]
    return TrendResult(
        slope_per_decade=float(slope_per_decade),
        ci=(float(ci_low), float(ci_high)),
        p=float(p_value),
        verdict=verdict,
        n=int(n),
        n_eff=float(n_eff),
        r1=float(r1),
        coverage=float(coverage),
        years_needed=float(years_needed),
        sen_per_decade=float(sen_slope),
        hac_p=float(_newey_west_pvalue(x, y, beta1, residuals)),
        settings={
            "variable": variable_name,
            "method": "OLS anomaly trend",
            "climatology": "fixed full-record calendar-month mean",
            "start": plan.start,
            "end": plan.end,
        },
        warnings=warnings,
    )


def trend_analysis(plan, variable_name: str | None = None, series: pd.Series | None = None):
    if series is None:
        series = load_series_for_plan(plan, variable_name or plan.variables[0])
    if variable_name is None:
        variable_name = plan.variables[0]
    return trend_result_from_series(plan, variable_name, series)
