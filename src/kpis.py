"""KPI computation helpers — pure pandas, no Streamlit imports."""

from __future__ import annotations

import pandas as pd


def compute_kpis(df: pd.DataFrame) -> dict:
    """Return the headline KPIs for the dashboard.

    Profit is intentionally absent — the dataset has no cost-of-goods column,
    so we report Revenue (Amount on non-cancelled orders) instead and
    document the substitution in the report.
    """
    if df.empty:
        return {
            "total_revenue": 0.0,
            "total_orders": 0,
            "cancellation_rate": 0.0,
            "avg_order_value": 0.0,
            "mom_growth": None,
            "best_category": "—",
            "unique_cities": 0,
            "unique_skus": 0,
        }

    revenue = df["revenue"] if "revenue" in df.columns else df["amount"].fillna(0)
    total_revenue = float(revenue.sum())
    total_orders = int(df["order_id"].nunique()) if "order_id" in df.columns else int(len(df))
    cancellations = int(df["is_cancelled"].sum()) if "is_cancelled" in df.columns else 0
    cancellation_rate = (cancellations / len(df) * 100) if len(df) else 0.0

    shipped = df[~df["is_cancelled"]] if "is_cancelled" in df.columns else df
    avg_order_value = float(shipped["amount"].mean()) if not shipped.empty else 0.0

    monthly = (
        df.assign(_m=df["date"].dt.to_period("M"))
        .groupby("_m")["revenue" if "revenue" in df.columns else "amount"]
        .sum()
        .sort_index()
    )
    if len(monthly) >= 2 and monthly.iloc[-2] > 0:
        mom_growth = float((monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2] * 100)
    else:
        mom_growth = None

    by_cat = df.groupby("category")["revenue" if "revenue" in df.columns else "amount"].sum()
    best_category = by_cat.idxmax() if not by_cat.empty else "—"

    return {
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "cancellation_rate": cancellation_rate,
        "avg_order_value": avg_order_value,
        "mom_growth": mom_growth,
        "best_category": str(best_category),
        "unique_cities": int(df["ship_city"].nunique()) if "ship_city" in df.columns else 0,
        "unique_skus": int(df["sku"].nunique()) if "sku" in df.columns else 0,
    }


def format_rupees(amount: float) -> str:
    """Indian-style currency formatter — lakh/crore breaks for readability."""
    if amount is None:
        return "—"
    if abs(amount) >= 1e7:
        return f"₹{amount / 1e7:.2f} Cr"
    if abs(amount) >= 1e5:
        return f"₹{amount / 1e5:.2f} L"
    if abs(amount) >= 1e3:
        return f"₹{amount / 1e3:.1f} K"
    return f"₹{amount:.0f}"


def kpi_card_html(label: str, value: str, delta: str | None = None, delta_dir: str = "flat") -> str:
    delta_html = ""
    if delta:
        cls = {"up": "kpi-delta-up", "down": "kpi-delta-down", "flat": "kpi-delta-flat"}[delta_dir]
        delta_html = f'<div class="{cls}">{delta}</div>'
    return (
        '<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{delta_html}'
        '</div>'
    )


def monthly_revenue(df: pd.DataFrame) -> pd.Series:
    rev_col = "revenue" if "revenue" in df.columns else "amount"
    return (
        df.assign(_m=df["date"].dt.to_period("M"))
        .groupby("_m")[rev_col]
        .sum()
        .rename("revenue")
    )
