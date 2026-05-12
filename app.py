"""Amazon Sales Intelligence Dashboard — Overview page (entrypoint).

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_data, sidebar_filters, inject_css
from src.kpis import compute_kpis, format_rupees, kpi_card_html
from src.charts import (
    line_revenue_with_ma,
    bar_revenue_by_category,
    donut_order_status,
)


st.set_page_config(
    page_title="Amazon Sales Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

st.title("Amazon Sales Intelligence Dashboard")
st.caption(
    "Business analytics on Amazon India sales (April–June 2022) — "
    "128k+ orders, 9 product categories, all-India geography."
)

df = load_data()
filtered = sidebar_filters(df)

if filtered.empty:
    st.warning("No rows match the current filters. Loosen the sidebar selection.")
    st.stop()

kpis = compute_kpis(filtered)

st.markdown('<div class="section-header">Key Performance Indicators</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.markdown(kpi_card_html("Total Revenue", format_rupees(kpis["total_revenue"])),
            unsafe_allow_html=True)
c2.markdown(kpi_card_html("Total Orders", f"{kpis['total_orders']:,}"),
            unsafe_allow_html=True)
c3.markdown(kpi_card_html("Cancellation Rate", f"{kpis['cancellation_rate']:.1f}%"),
            unsafe_allow_html=True)

c4, c5, c6 = st.columns(3)
c4.markdown(kpi_card_html("Avg Order Value", format_rupees(kpis["avg_order_value"])),
            unsafe_allow_html=True)
mom = kpis["mom_growth"]
if mom is None:
    mom_str, dir_ = "—", "flat"
else:
    mom_str = f"{mom:+.1f}%"
    dir_ = "up" if mom > 0 else "down" if mom < 0 else "flat"
c5.markdown(kpi_card_html("MoM Growth", mom_str, delta_dir=dir_),
            unsafe_allow_html=True)
c6.markdown(kpi_card_html("Best Category", kpis["best_category"]),
            unsafe_allow_html=True)

st.markdown('<div class="section-header">Revenue Trend</div>',
            unsafe_allow_html=True)
st.plotly_chart(line_revenue_with_ma(filtered, window=7), use_container_width=True)

col_a, col_b = st.columns([3, 2])
with col_a:
    st.plotly_chart(bar_revenue_by_category(filtered, top=10),
                    use_container_width=True)
with col_b:
    st.plotly_chart(donut_order_status(filtered), use_container_width=True)

with st.expander("📌 Snapshot summary"):
    top_state = filtered.groupby("ship_state")["revenue"].sum().idxmax() \
        if not filtered.empty else "—"
    top_sku = filtered.groupby("sku")["revenue"].sum().idxmax() \
        if "sku" in filtered.columns and not filtered.empty else "—"
    peak_day = filtered.groupby(filtered["date"].dt.date)["revenue"].sum().idxmax() \
        if not filtered.empty else "—"
    st.write(f"- **Top state by revenue:** {top_state}")
    st.write(f"- **Top SKU by revenue:** {top_sku}")
    st.write(f"- **Peak revenue day:** {peak_day}")
    st.write(f"- **Unique cities reached:** {kpis['unique_cities']:,}")
    st.write(f"- **Unique SKUs sold:** {kpis['unique_skus']:,}")

st.markdown(
    '<div class="footer">Data: Amazon Sale Report (Kaggle) · '
    'Built with Streamlit + Plotly · '
    'See <code>plan.txt</code> and <code>report/report.md</code> for methodology.</div>',
    unsafe_allow_html=True,
)
