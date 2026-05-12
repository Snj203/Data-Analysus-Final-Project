"""Geographic page — where orders are shipping to."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_data, sidebar_filters, inject_css
from src.charts import bar_top_states, bar_top_cities, bubble_map_india


st.set_page_config(page_title="Geographic", page_icon="🗺️", layout="wide")
inject_css()

st.title("Geographic Distribution")
st.caption("Where Amazon India orders are flowing — state and city concentration.")

df = load_data()
filtered = sidebar_filters(df)

if filtered.empty:
    st.warning("No rows match the current filters.")
    st.stop()

st.plotly_chart(bubble_map_india(filtered), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(bar_top_states(filtered, top=15), use_container_width=True)
with col2:
    st.plotly_chart(bar_top_cities(filtered, top=15), use_container_width=True)

st.markdown('<div class="section-header">State-level KPI table</div>',
            unsafe_allow_html=True)

state_kpis = (filtered.groupby("ship_state")
              .agg(orders=("order_id", "count"),
                   revenue=("revenue", "sum"),
                   cancellations=("is_cancelled", "sum"))
              .assign(cancel_rate_pct=lambda d: d["cancellations"] / d["orders"] * 100,
                      avg_order_value=lambda d: d["revenue"] / (d["orders"] - d["cancellations"]).replace(0, pd.NA))
              .sort_values("revenue", ascending=False)
              .reset_index()
              .round(2))

st.dataframe(state_kpis, use_container_width=True, hide_index=True)
st.caption(f"States in view: {state_kpis['ship_state'].nunique()}")
