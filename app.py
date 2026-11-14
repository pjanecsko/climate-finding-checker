from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import trend_analysis, validate_plan
from catalogue import DEFAULT_VARIABLES, get_variable_meta
from challenge import build_challenge_rows
from contracts import Plan
from data_access import load_full_series, load_plan_metadata, load_series_for_plan
from explanations import result_summary
from processing import anomaly_series, climatology_for

st.set_page_config(page_title="Climate finding checker", page_icon="🌦️", layout="wide")
st.title("Climate finding checker")
st.caption("Explore whether a climate finding is sensitive to reasonable analytical choices. Results are not claims of certainty.")
st.warning("Demo mode: values currently come from deterministic synthetic data, not NASA observations. Do not interpret them as evidence about Milan or the Po Valley.")

with st.expander("Example questions"):
    st.markdown("**Is Milan warming?** · **Is the Po Valley getting drier?** · **Was 2022 a one-off?**")

with st.form("question_flow"):
    st.subheader("Ask a structured question")
    col1, col2 = st.columns(2)
    with col1:
        place = st.selectbox("Where?", ["Milan", "Po Valley", "Bergamo"])
        theme = st.selectbox("Theme?", list(DEFAULT_VARIABLES))
        variable = st.selectbox("Variable", DEFAULT_VARIABLES[theme], format_func=lambda key: get_variable_meta(key)["label"])
    with col2:
        start = st.date_input("Start date", value=date(1981, 1, 1), min_value=date(1981, 1, 1), max_value=date(2025, 12, 1)
        )
        end = st.date_input("End date", value=date(2025, 12, 1), min_value=date(1981, 1, 1), max_value=date(2025, 12, 1))
        st.caption("The supported analysis is a monthly trend for one selected variable and location.")
    submitted = st.form_submit_button("Check the finding", type="primary")

if submitted:
    coordinates = {"Milan": (45.4642, 9.19), "Po Valley": (45.0, 10.0), "Bergamo": (45.6983, 9.6773)}
    lat, lon = coordinates[place]
    plan = Plan(place=place, lat=lat, lon=lon, variables=[variable], start=start.isoformat(), end=end.isoformat())
    series = load_series_for_plan(plan, variable)
    validation = validate_plan(plan, series)
    if validation.status != "supported":
        st.error(f"{validation.status.title()}: {validation.reason}")
        st.info(validation.alternative)
        st.stop()

    full_record = load_full_series(variable, place, lat, lon)
    result = trend_analysis(plan, variable, series, reference_series=full_record)
    meta = get_variable_meta(variable)
    st.caption(f"{meta['description']} Dataset: {load_plan_metadata(plan)['dataset']}. Target cell at {lat:.4f}, {lon:.4f}.")
    st.subheader("Baseline result")
    metrics = st.columns(5)
    metrics[0].metric("Slope / decade", f"{result.slope_per_decade:.3f} {meta['unit']}")
    metrics[1].metric("95% interval", f"{result.ci[0]:.3f} to {result.ci[1]:.3f}")
    metrics[2].metric("Primary p-value", f"{result.p:.3g}")
    metrics[3].metric("Coverage", f"{result.coverage:.1%}")
    metrics[4].metric("Effective n (rough)", f"{result.n_eff:.0f} / {result.n}")
    st.markdown(result_summary(result))
    st.caption(f"Verdict: **{result.verdict}**. HAC/Newey–West 12-month-lag p={result.hac_p:.3g}; seasonal Sen slope={result.sen_per_decade:.3f} per decade. Mann–Kendall is not used for the verdict.")
    for warning in result.warnings:
        st.caption(f"Note: {warning}")

    anomalies = anomaly_series(series, climatology_for(full_record)).dropna()
    if not anomalies.empty:
        elapsed_years = anomalies.index.year.to_numpy(dtype=float) + (anomalies.index.month.to_numpy(dtype=float) - 1) / 12.0
        x = elapsed_years - elapsed_years.min()
        beta = np.polyfit(x, anomalies.to_numpy(), 1)
        fitted = beta[0] + beta[1] * x
        lower = fitted + (result.ci[0] / 10.0) * x
        upper = fitted + (result.ci[1] / 10.0) * x
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series, name="Raw monthly value", line={"color": "#9aa4ad", "width": 1}, opacity=0.55))
        fig.add_trace(go.Scatter(x=anomalies.index, y=upper, name="95% trend interval", line={"width": 0}, showlegend=True))
        fig.add_trace(go.Scatter(x=anomalies.index, y=lower, name="Interval band", fill="tonexty", fillcolor="rgba(41,128,185,0.16)", line={"width": 0}))
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies, name="Monthly anomaly", line={"color": "#d35400", "width": 1.5}))
        fig.add_trace(go.Scatter(x=anomalies.index, y=fitted, name="OLS trend", line={"color": "#1f77b4", "width": 3}))
        fig.update_layout(title="Raw values and monthly anomalies", xaxis_title="Month", yaxis_title=meta["unit"], legend_title="Series")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Challenge my finding")
    st.caption("A fixed set of date, nearby-cell, method and coverage checks. The baseline above stays visible; variants are not arbitrary user-selected searches.")
    for row in build_challenge_rows(plan, result):
        with st.container(border=True):
            st.markdown(f"**{row.label}**")
            st.write(row.note)
            st.write(f"Slope: {row.result.slope_per_decade:.3f}/decade · primary p={row.result.p:.3g} · coverage={row.result.coverage:.1%} · {row.result.verdict}")
            if row.check == "coverage":
                st.write("Passes this threshold: " + ("yes" if row.result.settings.get("coverage_passes_variant") else "no"))
