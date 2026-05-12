"""Sales Analytics page — deeper dive into trends, mix, and outliers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_data, sidebar_filters, inject_css
from src.charts import (
    heatmap_dow_month,
    heatmap_correlation,
    scatter_amount_vs_qty,
    stacked_bar_category_fulfilment,
)


st.set_page_config(page_title="Sales Analytics", page_icon="📊", layout="wide")
inject_css()

st.title("Sales Analytics")
st.caption("Granular look at revenue patterns, product mix, and outliers.")

df = load_data()
filtered = sidebar_filters(df)

if filtered.empty:
    st.warning("No rows match the current filters.")
    st.stop()

st.markdown('<div class="section-header">Time-series revenue</div>',
            unsafe_allow_html=True)
granularity = st.selectbox("Aggregation granularity",
                           ["Daily", "Weekly", "Monthly"], index=0)

if granularity == "Daily":
    series = filtered.groupby(filtered["date"].dt.date)["revenue"].sum()
elif granularity == "Weekly":
    series = (filtered.assign(_w=filtered["date"].dt.to_period("W").apply(lambda p: p.start_time))
              .groupby("_w")["revenue"].sum())
else:
    series = (filtered.assign(_m=filtered["date"].dt.to_period("M").astype(str))
              .groupby("_m")["revenue"].sum())

series_df = series.reset_index()
series_df.columns = ["period", "revenue"]
fig = px.line(series_df, x="period", y="revenue",
              title=f"{granularity} revenue", markers=(granularity != "Daily"))
fig.update_layout(height=380, plot_bgcolor="white",
                  xaxis_title="", yaxis_title="Revenue (INR)")
st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="section-header">Patterns & mix</div>',
            unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(heatmap_dow_month(filtered), use_container_width=True)
with col2:
    st.plotly_chart(heatmap_correlation(filtered), use_container_width=True)

st.plotly_chart(stacked_bar_category_fulfilment(filtered, top=8),
                use_container_width=True)

st.markdown('<div class="section-header">Order amount vs quantity</div>',
            unsafe_allow_html=True)
st.plotly_chart(scatter_amount_vs_qty(filtered), use_container_width=True)

st.markdown('<div class="section-header">Top 20 orders by amount</div>',
            unsafe_allow_html=True)
top_orders = (filtered.dropna(subset=["amount"])
              .sort_values("amount", ascending=False)
              .head(20)
              [["order_id", "date", "category", "size", "qty",
                "amount", "ship_city", "ship_state", "status"]])
st.dataframe(top_orders, use_container_width=True, hide_index=True)

cancellation_by_cat = (filtered.groupby("category")
                       .agg(orders=("order_id", "count"),
                            cancellations=("is_cancelled", "sum"))
                       .assign(cancel_rate=lambda d: d["cancellations"] / d["orders"] * 100)
                       .sort_values("cancel_rate", ascending=False)
                       .reset_index())
st.markdown('<div class="section-header">Cancellation rate by category</div>',
            unsafe_allow_html=True)
st.dataframe(cancellation_by_cat.round(2), use_container_width=True, hide_index=True)
