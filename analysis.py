from __future__ import annotations

from contracts import TrendResult


def result_summary(result: TrendResult) -> str:
    if result.verdict == "significant increase":
        direction = "significant increase"
    elif result.verdict == "significant decrease":
        direction = "significant decrease"
    else:
        direction = "no statistically clear trend"
    return (
        f"The fitted monthly-anomaly trend indicates a {direction} of about "
        f"{result.slope_per_decade:.2f} per decade, with a p-value of {result.p:.3f}. "
        "This is not a claim of certainty; it is a sensitivity result under one reasonable analysis setup."
    )


def challenge_note(change: str, result: TrendResult) -> str:
    return (
        f"{change}: trend slope {result.slope_per_decade:.2f} per decade, "
        f"p={result.p:.3f}, coverage={result.coverage:.2%}. "
        "The value changes because the fitted window, cell, method or coverage rule changed."
    )
