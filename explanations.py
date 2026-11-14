from contracts import TrendResult


def result_summary(result: TrendResult) -> str:
    direction = {
        "significant increase": "a statistically significant increase",
        "significant decrease": "a statistically significant decrease",
        "not enough evidence": "no statistically clear trend",
    }[result.verdict]
    return (
        f"The fitted monthly-anomaly trend suggests {direction}: {result.slope_per_decade:.3f} per decade "
        f"(95% interval {result.ci[0]:.3f} to {result.ci[1]:.3f}; p={result.p:.3g}). "
        "This describes the result under the selected analysis choices; it is not a claim that one analysis establishes the truth."
    )
