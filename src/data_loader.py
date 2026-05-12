"""Data loading and filtering utilities for the Streamlit dashboard.

Everything that hits the disk or applies sidebar filters lives here so the
pages stay focused on layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_CSV = PROJECT_ROOT / "data" / "Amazon Sale Report Cleaned.csv"
RAW_CSV = PROJECT_ROOT / "data" / "Amazon Sale Report.csv"
FORECAST_CSV = PROJECT_ROOT / "data" / "forecast.csv"


@st.cache_data(show_spinner="Loading sales data…")
def load_data() -> pd.DataFrame:
    """Load the cleaned dataset. Falls back to the raw CSV with on-the-fly
    minimal cleaning so the dashboard still runs before the notebook is
    executed."""
    if CLEANED_CSV.exists():
        df = pd.read_csv(CLEANED_CSV, parse_dates=["date"])
    else:
        df = _quick_clean(pd.read_csv(RAW_CSV))
    return df


def _quick_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleaning so the app does not crash when the user hasn't run
    the cleaning notebook yet."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    if "sales_channel_" in df.columns:
        df = df.rename(columns={"sales_channel_": "sales_channel"})
    df["date"] = pd.to_datetime(df["date"], format="%m-%d-%y", errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0)
    df["is_cancelled"] = df["status"].str.contains("Cancel", case=False, na=False)
    df["revenue"] = df["amount"].where(~df["is_cancelled"], 0).fillna(0)
    df["order_month"] = df["date"].dt.to_period("M").astype(str)
    df["order_weekday"] = df["date"].dt.day_name()
    df["order_year"] = df["date"].dt.year
    df["has_promotion"] = df.get("promotion_ids", pd.Series([None] * len(df))).notna()
    for col in ("ship_city", "ship_state"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().str.strip()
    return df


@st.cache_data
def load_forecast() -> pd.DataFrame | None:
    if not FORECAST_CSV.exists():
        return None
    return pd.read_csv(FORECAST_CSV, parse_dates=["date"])


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render the global sidebar filters and return the filtered dataframe."""
    st.sidebar.markdown("### Filters")

    min_d, max_d = df["date"].min(), df["date"].max()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_d.date(), max_d.date()),
        min_value=min_d.date(),
        max_value=max_d.date(),
    )

    categories = sorted(df["category"].dropna().unique().tolist())
    sel_categories = st.sidebar.multiselect("Category", categories, default=categories)

    statuses = sorted(df["status"].dropna().unique().tolist())
    sel_statuses = st.sidebar.multiselect("Order Status", statuses, default=statuses)

    states = sorted(df["ship_state"].dropna().unique().tolist())
    sel_states = st.sidebar.multiselect("Ship State", states, default=states)

    fulfilments = sorted(df["fulfilment"].dropna().unique().tolist())
    sel_fulfilments = st.sidebar.multiselect("Fulfilment", fulfilments, default=fulfilments)

    city_query = st.sidebar.text_input("Search city (contains)", value="")

    include_outliers = st.sidebar.checkbox("Include amount outliers", value=True)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Source rows: {len(df):,}  \n"
        f"Date span: {min_d.date()} → {max_d.date()}"
    )

    return apply_filters(
        df,
        date_range=date_range,
        categories=sel_categories,
        statuses=sel_statuses,
        states=sel_states,
        fulfilments=sel_fulfilments,
        city_query=city_query,
        include_outliers=include_outliers,
    )


def apply_filters(
    df: pd.DataFrame,
    date_range: tuple,
    categories: Iterable[str],
    statuses: Iterable[str],
    states: Iterable[str],
    fulfilments: Iterable[str],
    city_query: str,
    include_outliers: bool,
) -> pd.DataFrame:
    out = df
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        out = out[(out["date"] >= start) & (out["date"] <= end)]
    if categories:
        out = out[out["category"].isin(list(categories))]
    if statuses:
        out = out[out["status"].isin(list(statuses))]
    if states:
        out = out[out["ship_state"].isin(list(states))]
    if fulfilments:
        out = out[out["fulfilment"].isin(list(fulfilments))]
    if city_query:
        out = out[out["ship_city"].astype(str).str.contains(city_query.strip().upper(), na=False)]
    if not include_outliers and "is_amount_outlier" in out.columns:
        out = out[~out["is_amount_outlier"]]
    return out


def inject_css() -> None:
    css_path = PROJECT_ROOT / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
