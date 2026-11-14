from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from catalogue import SUPPORTED_VARIABLES
from contracts import Plan, TrendResult, Validation
from data_access import load_series_for_plan
from processing import anomaly_series, climatology_for, seasonal_sen_slope, series_coverage_ok


def validate_plan(plan: Plan, series: pd.Series | None = None) -> Validation:
    if not plan.variables:
        return Validation("incomplete", "No variable was selected.", "Choose a supported climate variable.")
    if plan.kind != "trend":
        return Validation("unsupported", "This investigation type is not in the demo catalogue.", "Choose a single-location trend analysis.")
    for variable in plan.variables:
        if variable not in SUPPORTED_VARIABLES:
            return Validation("unsupported", f"Variable '{variable}' is not in the catalogue.", "Choose one of the listed climate variables.")
    start, end = pd.Timestamp(plan.start), pd.Timestamp(plan.end)
    if start > end:
        return Validation("incomplete", "The date range is invalid.", "Choose a start date earlier than the end date.")
    if series is None:
        series = load_series_for_plan(plan)
    ok, coverage = series_coverage_ok(series)
    if not ok:
        return Validation(
            "incomplete",
            f"Coverage rules failed: {coverage['full_years']} calendar years and {coverage['months_present']:.1%} of months present; requirements are at least 10 years, 80% overall, and 70% for each calendar month.",
            "Try a longer date window or a series with more complete monthly coverage.",
        )
    return Validation("supported", "Coverage and variable meet the configured rules.")


def _hac_p_value(x: np.ndarray, y: np.ndarray, beta: np.ndarray, residuals: np.ndarray, max_lag: int = 12) -> float:
    """Newey-West (Bartlett weights), with a documented fixed 12-month lag."""
    n = len(y)
    design = np.column_stack((np.ones(n), x))
    lag = min(max_lag, max(n - 1, 0))
    scores = design * residuals[:, None]
    meat = scores.T @ scores
    for k in range(1, lag + 1):
        weight = 1.0 - k / (lag + 1.0)
        gamma = scores[k:].T @ scores[:-k]
        meat += weight * (gamma + gamma.T)
    bread = np.linalg.pinv(design.T @ design)
    covariance = bread @ meat @ bread
    se = float(np.sqrt(max(covariance[1, 1], 0.0)))
    if not np.isfinite(se) or se == 0:
        return 1.0
    z = abs(float(beta[1]) / se)
    return float(2.0 * stats.norm.sf(z))


def trend_analysis(
    plan: Plan,
    variable_name: str | None = None,
    series: pd.Series | None = None,
    *,
    reference_series: pd.Series | None = None,
    method: str = "OLS",
) -> TrendResult:
    variable = variable_name or (plan.variables[0] if plan.variables else "MERRA2_T2M")
    if series is None:
        series = load_series_for_plan(plan, variable)
    series = series.sort_index().astype(float)
    # Reference series pins the calendar-month climatology across date-window challenges.
    baseline = reference_series if reference_series is not None else series
    anomalies = anomaly_series(series, climatology_for(baseline))
    clean = anomalies.dropna()
    if len(clean) < 3:
        raise ValueError("At least three non-missing monthly values are needed to fit a trend.")

    decimal_year = clean.index.year.to_numpy(dtype=float) + (clean.index.month.to_numpy(dtype=float) - 1.0) / 12.0
    x = decimal_year - decimal_year.min()
    y = clean.to_numpy(dtype=float)
    design = np.column_stack((np.ones(len(x)), x))
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    residuals = y - fitted
    n = len(y)
    r1 = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1]) if n > 2 else 0.0
    if not np.isfinite(r1):
        r1 = 0.0
    r1 = float(np.clip(r1, -0.99, 0.99))
    n_eff = float(max(1.0, min(float(n), n * (1.0 - r1) / (1.0 + r1))))

    sxx = float(np.sum((x - x.mean()) ** 2))
    df = max(n - 2, 1)
    sigma2 = float(np.sum(residuals ** 2) / df)
    se_beta = float(np.sqrt(max(sigma2 / sxx, 0.0))) if sxx else 0.0
    # Inflate ordinary least-squares uncertainty using the rough AR(1) n_eff approximation.
    inflation = float(np.sqrt(max(n / n_eff, 1.0)))
    adjusted_se = se_beta * inflation
    adjusted_df = max(n_eff - 2.0, 1.0)
    t_stat = float(beta[1] / adjusted_se) if adjusted_se else 0.0
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=adjusted_df))
    t_critical = float(stats.t.ppf(0.975, df=adjusted_df))
    slope = float(beta[1] * 10.0)
    ci = (float((beta[1] - t_critical * adjusted_se) * 10.0), float((beta[1] + t_critical * adjusted_se) * 10.0))
    if p_value < 0.05 and slope > 0:
        verdict = "significant increase"
    elif p_value < 0.05 and slope < 0:
        verdict = "significant decrease"
    else:
        verdict = "not enough evidence"

    coverage = series_coverage_ok(series)[1]["months_present"]
    sen = seasonal_sen_slope(anomalies)
    hac_p = _hac_p_value(x, y, beta, residuals, max_lag=12)
    # Transparent rough display estimate, not a forecast or a power calculation.
    years_needed = float(max(10.0, 1.96**2 / abs(slope))) if abs(slope) > 1e-12 else float("inf")
    settings = {
        "variable": variable,
        "method": method,
        "climatology": "fixed full-record calendar-month mean",
        "start": plan.start,
        "end": plan.end,
        "hac_lag_months": 12,
    }
    warnings = [
        "AR(1) effective sample size is a rough adjustment, not a full solution to autocorrelation.",
        "Years-needed is a rough estimate; it assumes the current rate and AR(1) noise.",
        "HAC/Newey-West cross-check uses a fixed 12-month Bartlett lag.",
    ]
    return TrendResult(
        slope_per_decade=slope,
        ci=ci,
        p=p_value,
        verdict=verdict,
        n=n,
        n_eff=n_eff,
        r1=r1,
        coverage=float(coverage),
        years_needed=years_needed,
        sen_per_decade=float(sen),
        hac_p=hac_p,
        settings=settings,
        warnings=warnings,
    )
