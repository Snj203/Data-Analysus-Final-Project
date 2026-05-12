"""Forecasting page — historical revenue plus 30-day forecast from notebook 03."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_data, load_forecast, sidebar_filters, inject_css
from src.charts import line_forecast


st.set_page_config(page_title="Forecasting", page_icon="🔮", layout="wide")
inject_css()

st.title("Revenue Forecasting")
st.caption("ML-based projection of daily revenue for the next 30 days.")

df = load_data()
filtered = sidebar_filters(df)
forecast = load_forecast()

if filtered.empty:
    st.warning("No rows match the current filters.")
    st.stop()

st.markdown('<div class="section-header">Forecast horizon</div>',
            unsafe_allow_html=True)

horizon = st.slider("Days to display", min_value=7, max_value=30, value=30, step=1)

history = (filtered[~filtered["is_cancelled"]]
           .groupby(filtered["date"].dt.date)["revenue"].sum()
           .reset_index())
history["date"] = pd.to_datetime(history["date"])

if forecast is None:
    st.warning(
        "Forecast file `data/forecast.csv` not found. "
        "Run `notebooks/03_forecasting.ipynb` end-to-end to generate it."
    )
    st.plotly_chart(line_forecast(history, pd.DataFrame()), use_container_width=True)
else:
    forecast_view = forecast.head(horizon)
    st.plotly_chart(line_forecast(history, forecast_view), use_container_width=True)

    st.markdown('<div class="section-header">Forecast values</div>',
                unsafe_allow_html=True)
    st.dataframe(forecast_view.round(2), use_container_width=True, hide_index=True)

    if "model" in forecast.columns:
        st.caption(f"Model used: {forecast['model'].iloc[0]}")

with st.expander("📖 Methodology"):
    st.markdown(
        """
        **Pipeline (see `notebooks/03_forecasting.ipynb`):**

        1. Build daily revenue series from shipped (non-cancelled) orders.
        2. Stationarity check (Augmented Dickey-Fuller).
        3. Train/test split — last 30 days held out for evaluation.
        4. Three models compared:
           - Baseline: 7-day moving average forward-fill.
           - Linear Regression on `[day_index, day_of_week, month]` features.
           - Holt-Winters Exponential Smoothing (additive seasonality).
        5. Best model by MAPE is refit on the full history and projects 30 days.

        **Limitations.** Only ~3 months of data → seasonality is approximate; no
        holiday calendar; no exogenous variables (price, marketing spend).
        Treat as directional, not as a committed plan.
        """
    )
