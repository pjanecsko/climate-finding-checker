# Climate finding checker

A small Streamlit prototype for exploring how a climate trend finding changes under reasonable analytical choices.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app starts without secrets or a live NASA request. **Current data are deterministic synthetic demo data, not NASA measurements.** The app labels this clearly; do not use demo outputs as evidence about the climate.

## Implemented

- Stable shared dataclasses in `contracts.py`
- Structured Streamlit question form with location, theme, variable and date range
- Monthly series with missing months retained as missing
- Fixed full-record calendar-month climatology for anomaly calculation and date challenges
- OLS monthly-anomaly trend, rough AR(1) effective-sample-size adjustment, HAC/Newey–West 12-month cross-check and seasonal Sen slope
- Coverage checks and fixed date, region, method and coverage challenge rows
- Synthetic provider fallback so the interface works offline

## Not yet implemented

NASA Giovanni retrieval, Earthdata Login token handling, NASA POWER fallback, real-data cache, land-cell snapping, Mann–Kendall cross-check, and deployment. No credentials should be committed. Connect a real data provider before presenting results as observational findings.
