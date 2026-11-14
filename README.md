from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from catalogue import DEFAULT_VARIABLES, get_variable_meta
from contracts import Plan
from analysis import trend_analysis, validate_plan
from challenge import build_challenge_rows
from explanations import result_summary


st.set_page_config(layout="wide")

st.title("Climate finding checker")
st.caption("A sensitivity-first climate app: the result is not a claim of truth, just a map of how plausible choices change the finding.")

example_prompts = [
    "Milan warming",
    "Po Valley drought",
    "Soil moisture decline",
    "2022 one-off?",
]

for chip in example_prompts:
    if st.button(chip, key=f"chip_{chip}", use_container_width=False):
        st.session_state["example_prompt"] = chip

if "example_prompt" in st.session_state:
    st.info(f"Example selected: {st.session_state['example_prompt']}")

with st.form("question_form"):
    col1, col2 = st.columns(2)
    with col1:
        place = st.selectbox("Where?", ["Milan", "Bergamo", "Po Valley", "Other"], index=0)
        theme = st.selectbox("Theme?", ["temperature", "precipitation", "soil moisture"], index=0)
        variable = st.selectbox("Variable", DEFAULT_VARIABLES.get(theme, ["MERRA2_T2M"]))
    with col2:
        start = st.date_input("Start date", value=date(1981, 1, 1), min_value=date(1981, 1, 1), max_value=date(2025, 12, 1))
        end = st.date_input("End date", value=date(2025, 12, 1), min_value=date(1981, 1, 1), max_value=date(2025, 12, 1))
        st.markdown("Example chips stand in for a structured question parser; no free-text AI inference here.")

    submitted = st.form_submit_button("Run check")

if submitted:
    plan = Plan(
        place=place,
        lat=45.46,
        lon=9.19,
        variables=[variable],
        start=start.isoformat(),
        end=end.isoformat(),
        kind="trend",
        mode="main",
    )
    validation = validate_plan(plan)
    if validation.status != "supported":
        st.warning(f"Validation: {validation.reason}")
        st.info(f"Alternative: {validation.alternative}")
        st.stop()

    # Use a synthetic, deterministic monthly series until NASA data is available in cache.
    from data_access import load_series_for_plan

    series = load_series_for_plan(plan, variable)
    result = trend_analysis(plan, variable, series)

    st.subheader("Result card")
    cols = st.columns(5)
    with cols[0]:
        st.metric("Slope / decade", f"{result.slope_per_decade:.2f}")
    with cols[1]:
        st.metric("95% CI", f"{result.ci[0]:.2f} to {result.ci[1]:.2f}")
    with cols[2]:
        st.metric("p-value", f"{result.p:.3f}")
    with cols[3]:
        st.metric("Coverage", f"{result.coverage:.2%}")
    with cols[4]:
        st.metric("n_eff", f"{result.n_eff:.0f}")

    st.markdown(result_summary(result))
    for warning in result.warnings:
        st.warning(warning)

    # Plot raw data, anomalies, trend line and confidence band.
    month_index = pd.date_range(start=plan.start, end=plan.end, freq="MS")
    raw = pd.Series(index=month_index, data=series.reindex(month_index).to_numpy())
    raw = raw.dropna()
    anomalies = raw - raw.groupby(raw.index.month).transform("mean")
    fit_x = pd.Series(range(len(anomalies)), index=anomalies.index)
    x_numeric = [i for i in range(len(anomalies))]
    y_vals = anomalies.to_numpy()
    beta0, beta1 = __import__("numpy").polyfit(x_numeric, y_vals, 1)
    trend_line = beta0 + beta1 * np.asarray(x_numeric)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=raw.index, y=raw.values, mode="lines", name="raw monthly value", line={"color": "#7f8c8d", "width": 1}, opacity=0.7))
    fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies.values, mode="lines", name="monthly anomaly", line={"color": "#d35400", "width": 2}))
    fig.add_trace(go.Scatter(x=anomalies.index, y=trend_line, mode="lines", name="trend line", line={"color": "#2980b9", "width": 3}))
    fig.update_layout(title=f"{variable} for {plan.place}", xaxis_title="Month", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Challenge my finding")
    baseline = result
    rows = build_challenge_rows(plan, baseline)
    for row in rows:
        st.markdown(f"**{row.check.capitalize()} / {row.label}**")
        st.write(f"Slope: {row.result.slope_per_decade:.2f} / decade | p={row.result.p:.3f} | coverage={row.result.coverage:.2%} | verdict={row.result.verdict}")
        st.caption(row.note)
        st.markdown("---")

    st.caption("All challenge rows are predefined; the interface does not allow endless arbitrary date fishing to seek a preferred answer.")
