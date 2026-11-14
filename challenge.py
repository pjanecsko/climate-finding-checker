from __future__ import annotations

from contracts import ChallengeRow, Plan, TrendResult
from analysis import trend_analysis
from data_access import load_full_series, load_series_for_plan
from processing import series_coverage_ok


def _copy_plan(plan: Plan, *, start: str | None = None, lat: float | None = None, lon: float | None = None) -> Plan:
    return Plan(
        place=plan.place,
        lat=plan.lat if lat is None else lat,
        lon=plan.lon if lon is None else lon,
        variables=list(plan.variables),
        start=start or plan.start,
        end=plan.end,
        kind=plan.kind,
        mode=plan.mode,
        season=plan.season,
        lat2=plan.lat2,
        lon2=plan.lon2,
    )


def build_challenge_rows(plan: Plan, baseline: TrendResult) -> list[ChallengeRow]:
    variable = plan.variables[0]
    full_record = load_full_series(variable, plan.place, plan.lat, plan.lon)
    rows: list[ChallengeRow] = []

    # Date checks use the same full-record climatology as the baseline.
    for start in ("1981-01-01", "1990-01-01", "2000-01-01", "2010-01-01"):
        if pd_timestamp(start) >= pd_timestamp(plan.end):
            continue
        variant = _copy_plan(plan, start=start)
        data = load_series_for_plan(variant, variable)
        result = trend_analysis(variant, variable, data, reference_series=full_record)
        rows.append(ChallengeRow("date", f"Date: {start}–{plan.end}", result, "Changes only the fitted date window; the full-record monthly climatology stays fixed."))

    # Nearby cell checks are deterministic synthetic analogues until the NASA provider is connected.
    for dlat, dlon, label in ((-0.25, 0.0, "north neighbour"), (0.25, 0.0, "south neighbour"), (0.0, 0.25, "east neighbour"), (0.0, -0.25, "west neighbour")):
        variant = _copy_plan(plan, lat=plan.lat + dlat, lon=plan.lon + dlon)
        data = load_series_for_plan(variant, variable)
        reference = load_full_series(variable, variant.place, variant.lat, variant.lon)
        result = trend_analysis(variant, variable, data, reference_series=reference)
        rows.append(ChallengeRow("region", f"Region: {label} ({variant.lat:.2f}, {variant.lon:.2f})", result, "Changes the selected grid-cell coordinates; demo cells are synthetic and are not NASA grid cells."))

    rows.append(ChallengeRow("method", "Method: OLS (baseline)", baseline, "Primary result: OLS anomaly trend; uncertainty uses the stated rough AR(1) effective-sample-size adjustment."))
    hac_result = trend_analysis(plan, variable, load_series_for_plan(plan, variable), reference_series=full_record, method="HAC cross-check")
    rows.append(ChallengeRow("method", "Method: HAC/Newey–West", hac_result, "Keeps the OLS slope; compare the fixed 12-month HAC p-value in the result settings and cross-check."))
    sen_result = trend_analysis(plan, variable, load_series_for_plan(plan, variable), reference_series=full_record, method="seasonal Sen cross-check")
    sen_result.settings["cross_check_sen_slope_per_decade"] = sen_result.sen_per_decade
    rows.append(ChallengeRow("method", "Method: seasonal Sen’s slope", sen_result, "Reports the median within-calendar-month pairwise slope as a robust slope cross-check; it does not replace the primary verdict."))

    data = load_series_for_plan(plan, variable)
    for label, threshold, month_threshold, note in (
        ("default threshold", 0.80, 0.70, "Requires >=80% of months, >=10 calendar years and >=70% for each calendar month."),
        ("stricter threshold", 0.90, 0.75, "Requires >=90% of months and >=75% for each calendar month."),
    ):
        _, details = series_coverage_ok(data, threshold=threshold, month_threshold=month_threshold)
        threshold_result = TrendResult(**{**baseline.__dict__, "settings": {**baseline.settings, "coverage_threshold": threshold, "month_threshold": month_threshold, "coverage_passes_variant": details["months_present"] >= threshold and all(v >= month_threshold for v in details["calendar_month_coverage"].values())}})
        rows.append(ChallengeRow("coverage", f"Coverage: {label}", threshold_result, note))
    return rows


def pd_timestamp(value: str):
    import pandas as pd
    return pd.Timestamp(value)
