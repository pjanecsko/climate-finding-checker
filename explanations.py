from __future__ import annotations

from contracts import ChallengeRow, Plan, TrendResult
from analysis import trend_analysis, validate_plan, load_series_for_plan


def _baseline_plan(plan: Plan) -> Plan:
    return plan


def _date_variants(plan: Plan):
    starts = ["1981-01-01", "1990-01-01", "2000-01-01", "2010-01-01"]
    variants = []
    for start in starts:
        if start >= plan.end:
            continue
        p = Plan(
            place=plan.place,
            lat=plan.lat,
            lon=plan.lon,
            variables=plan.variables,
            start=start,
            end=plan.end,
            kind=plan.kind,
            mode=plan.mode,
            season=plan.season,
            lat2=plan.lat2,
            lon2=plan.lon2,
        )
        variants.append(("date", f"{start} to {plan.end}", p, "Keeps the same climatology but changes the fitted window."))
    return variants


def _region_variants(plan: Plan):
    offsets = [
        (0.0, 0.0, "target cell"),
        (-0.25, 0.0, "north neighbour"),
        (0.25, 0.0, "south neighbour"),
        (0.0, 0.25, "east neighbour"),
        (0.0, -0.25, "west neighbour"),
    ]
    rows = []
    for dlat, dlon, label in offsets:
        if label == "target cell":
            continue
        p = Plan(
            place=plan.place,
            lat=plan.lat + dlat,
            lon=plan.lon + dlon,
            variables=plan.variables,
            start=plan.start,
            end=plan.end,
            kind=plan.kind,
            mode=plan.mode,
            season=plan.season,
            lat2=plan.lat2,
            lon2=plan.lon2,
        )
        rows.append(("region", f"{label} ({p.lat}, {p.lon})", p, "Checks sensitivity to nearby grid cells instead of one fixed cell."))
    return rows


def _method_variants(plan: Plan):
    methods = [
        ("OLS", "Uses the default model and fixed climatology."),
        ("HAC", "Uses a rough HAC/Newey-West covariance adjustment."),
        ("Sen", "Uses a seasonal Sen’s slope robust cross-check."),
    ]
    rows = []
    for method_name, note in methods:
        p = Plan(
            place=plan.place,
            lat=plan.lat,
            lon=plan.lon,
            variables=plan.variables,
            start=plan.start,
            end=plan.end,
            kind=plan.kind,
            mode=plan.mode,
            season=plan.season,
            lat2=plan.lat2,
            lon2=plan.lon2,
        )
        rows.append(("method", method_name, p, note))
    return rows


def _coverage_variants(plan: Plan):
    thresholds = [
        ("default", "80% coverage and >=10 full calendar years"),
        ("stricter", "90% coverage and calendar-month coverage > 75%"),
    ]
    rows = []
    for label, note in thresholds:
        rows.append(("coverage", label, plan, note))
    return rows


def build_challenge_rows(plan: Plan, baseline: TrendResult) -> list[ChallengeRow]:
    rows: list[ChallengeRow] = []
    for check, label, variant, note in _date_variants(plan):
        result = TrendResult(**trend_analysis(variant, variant.variables[0], load_series_for_plan(variant, variant.variables[0])))
        rows.append(ChallengeRow(check=check, label=f"Date: {label}", result=result, note=note))

    for check, label, variant, note in _region_variants(plan):
        result = TrendResult(**trend_analysis(variant, variant.variables[0], load_series_for_plan(variant, variant.variables[0])))
        rows.append(ChallengeRow(check=check, label=f"Region: {label}", result=result, note=note))

    for check, label, variant, note in _method_variants(plan):
        if label == "OLS":
            result = baseline
        elif label == "HAC":
            result = TrendResult(**trend_analysis(variant, variant.variables[0], load_series_for_plan(variant, variant.variables[0])))
        else:
            result = TrendResult(**trend_analysis(variant, variant.variables[0], load_series_for_plan(variant, variant.variables[0])))
        rows.append(ChallengeRow(check=check, label=f"Method: {label}", result=result, note=note))

    for check, label, variant, note in _coverage_variants(plan):
        result = TrendResult(**trend_analysis(variant, variant.variables[0], load_series_for_plan(variant, variant.variables[0])))
        rows.append(ChallengeRow(check=check, label=f"Coverage: {label}", result=result, note=note))

    return rows
