"""Plotly chart builders for the dashboard pages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PALETTE = px.colors.qualitative.Set2
PRIMARY = "#FF6B35"
SECONDARY = "#1F77B4"


def _empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text="No data for current filters",
                       xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=14, color="#9CA3AF"))
    fig.update_layout(title=title, height=380, plot_bgcolor="white")
    return fig


def line_revenue_with_ma(df: pd.DataFrame, window: int = 7) -> go.Figure:
    if df.empty:
        return _empty_fig("Daily revenue trend")
    daily = df.groupby(df["date"].dt.date)["revenue"].sum().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["ma"] = daily["revenue"].rolling(window=window, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["revenue"], name="Daily revenue",
                             line=dict(color=SECONDARY, width=1.2), opacity=0.55))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["ma"], name=f"{window}-day MA",
                             line=dict(color=PRIMARY, width=2.5)))
    fig.update_layout(title="Daily revenue trend with moving average",
                      xaxis_title="Date", yaxis_title="Revenue (INR)",
                      height=400, plot_bgcolor="white", hovermode="x unified")
    return fig


def bar_revenue_by_category(df: pd.DataFrame, top: int = 10) -> go.Figure:
    if df.empty:
        return _empty_fig("Revenue by category")
    by_cat = (df.groupby("category")["revenue"].sum()
              .sort_values(ascending=False).head(top).reset_index())
    fig = px.bar(by_cat, x="revenue", y="category", orientation="h",
                 color="revenue", color_continuous_scale="Oranges",
                 title=f"Top {top} categories by revenue")
    fig.update_layout(yaxis=dict(autorange="reversed"), height=420,
                      coloraxis_showscale=False, plot_bgcolor="white")
    fig.update_xaxes(title="Revenue (INR)")
    fig.update_yaxes(title="")
    return fig


def donut_order_status(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig("Order status distribution")
    counts = df["status"].value_counts().reset_index()
    counts.columns = ["status", "count"]
    fig = px.pie(counts, names="status", values="count", hole=0.55,
                 title="Order status distribution",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=420)
    return fig


def heatmap_dow_month(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig("Revenue heatmap — Weekday × Month")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = (df.assign(month=df["date"].dt.strftime("%Y-%m"))
               .pivot_table(index="order_weekday", columns="month",
                            values="revenue", aggfunc="sum", fill_value=0))
    pivot = pivot.reindex([d for d in order if d in pivot.index])
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Oranges",
                    title="Revenue heatmap — Weekday × Month")
    fig.update_layout(height=380, plot_bgcolor="white")
    fig.update_xaxes(title="Month")
    fig.update_yaxes(title="Weekday")
    return fig


def heatmap_correlation(df: pd.DataFrame) -> go.Figure:
    cols = [c for c in ["amount", "qty", "is_cancelled", "has_promotion", "b2b"]
            if c in df.columns]
    if df.empty or len(cols) < 2:
        return _empty_fig("Correlation heatmap")
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr(method="pearson")
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu",
                    zmin=-1, zmax=1, title="Pearson correlation (filtered data)")
    fig.update_layout(height=380)
    return fig


def scatter_amount_vs_qty(df: pd.DataFrame, sample: int = 8000) -> go.Figure:
    if df.empty:
        return _empty_fig("Amount vs Quantity")
    plot_df = df.dropna(subset=["amount", "qty"])
    if len(plot_df) > sample:
        plot_df = plot_df.sample(sample, random_state=42)
    fig = px.scatter(plot_df, x="qty", y="amount", color="category",
                     opacity=0.55, title="Amount vs Quantity (sampled)",
                     color_discrete_sequence=PALETTE)
    fig.update_layout(height=440, plot_bgcolor="white")
    fig.update_xaxes(title="Quantity")
    fig.update_yaxes(title="Amount (INR)")
    return fig


def bar_top_states(df: pd.DataFrame, top: int = 15) -> go.Figure:
    if df.empty:
        return _empty_fig("Top states by revenue")
    by_state = (df.groupby("ship_state")["revenue"].sum()
                .sort_values(ascending=False).head(top).reset_index())
    fig = px.bar(by_state, x="revenue", y="ship_state", orientation="h",
                 color="revenue", color_continuous_scale="Blues",
                 title=f"Top {top} states by revenue")
    fig.update_layout(yaxis=dict(autorange="reversed"), height=480,
                      coloraxis_showscale=False, plot_bgcolor="white")
    fig.update_xaxes(title="Revenue (INR)")
    fig.update_yaxes(title="")
    return fig


def bar_top_cities(df: pd.DataFrame, top: int = 15) -> go.Figure:
    if df.empty:
        return _empty_fig("Top cities by orders")
    by_city = df["ship_city"].value_counts().head(top).reset_index()
    by_city.columns = ["ship_city", "orders"]
    fig = px.bar(by_city, x="orders", y="ship_city", orientation="h",
                 color="orders", color_continuous_scale="Teal",
                 title=f"Top {top} cities by order count")
    fig.update_layout(yaxis=dict(autorange="reversed"), height=480,
                      coloraxis_showscale=False, plot_bgcolor="white")
    fig.update_xaxes(title="Order count")
    fig.update_yaxes(title="")
    return fig


def bubble_map_india(df: pd.DataFrame) -> go.Figure:
    """Bubble map of orders per state using state-capital coordinates."""
    if df.empty:
        return _empty_fig("Geographic distribution")
    capitals = _india_state_capitals()
    by_state = df.groupby("ship_state").agg(
        orders=("order_id", "count"),
        revenue=("revenue", "sum"),
    ).reset_index()
    by_state["state_key"] = by_state["ship_state"].str.upper().str.strip()
    capitals_df = pd.DataFrame(capitals, columns=["state_key", "lat", "lon"])
    merged = by_state.merge(capitals_df, on="state_key", how="inner")
    if merged.empty:
        return _empty_fig("Geographic distribution")
    fig = px.scatter_geo(merged, lat="lat", lon="lon",
                         size="revenue", hover_name="ship_state",
                         hover_data={"orders": True, "revenue": ":.0f",
                                     "lat": False, "lon": False},
                         color="revenue", color_continuous_scale="Plasma",
                         title="Orders concentration across India",
                         projection="natural earth")
    fig.update_geos(scope="asia", center=dict(lat=22, lon=80),
                    lataxis_range=[5, 38], lonaxis_range=[65, 100],
                    showcountries=True, countrycolor="#9CA3AF",
                    showland=True, landcolor="#F5F7FA")
    fig.update_layout(height=550)
    return fig


def _india_state_capitals() -> list[tuple[str, float, float]]:
    return [
        ("ANDHRA PRADESH", 16.5062, 80.6480),
        ("ARUNACHAL PRADESH", 27.0844, 93.6053),
        ("ASSAM", 26.1445, 91.7362),
        ("BIHAR", 25.5941, 85.1376),
        ("CHHATTISGARH", 21.2514, 81.6296),
        ("GOA", 15.4909, 73.8278),
        ("GUJARAT", 23.0225, 72.5714),
        ("HARYANA", 28.4595, 77.0266),
        ("HIMACHAL PRADESH", 31.1048, 77.1734),
        ("JHARKHAND", 23.3441, 85.3096),
        ("KARNATAKA", 12.9716, 77.5946),
        ("KERALA", 8.5241, 76.9366),
        ("MADHYA PRADESH", 23.2599, 77.4126),
        ("MAHARASHTRA", 19.0760, 72.8777),
        ("MANIPUR", 24.8170, 93.9368),
        ("MEGHALAYA", 25.5788, 91.8933),
        ("MIZORAM", 23.7271, 92.7176),
        ("NAGALAND", 25.6701, 94.1086),
        ("ODISHA", 20.2961, 85.8245),
        ("PUNJAB", 30.7333, 76.7794),
        ("RAJASTHAN", 26.9124, 75.7873),
        ("SIKKIM", 27.3389, 88.6065),
        ("TAMIL NADU", 13.0827, 80.2707),
        ("TELANGANA", 17.3850, 78.4867),
        ("TRIPURA", 23.8315, 91.2868),
        ("UTTAR PRADESH", 26.8467, 80.9462),
        ("UTTARAKHAND", 30.3165, 78.0322),
        ("WEST BENGAL", 22.5726, 88.3639),
        ("DELHI", 28.7041, 77.1025),
        ("JAMMU & KASHMIR", 34.0837, 74.7973),
        ("LADAKH", 34.1526, 77.5770),
        ("PUDUCHERRY", 11.9416, 79.8083),
        ("CHANDIGARH", 30.7333, 76.7794),
        ("ANDAMAN & NICOBAR", 11.7401, 92.6586),
        ("DADRA AND NAGAR HAVELI", 20.1809, 73.0169),
        ("LAKSHADWEEP", 10.5667, 72.6417),
    ]


def line_forecast(history: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not history.empty:
        fig.add_trace(go.Scatter(x=history["date"], y=history["revenue"],
                                 name="Historical", line=dict(color=SECONDARY, width=1.5)))
    if forecast is not None and not forecast.empty:
        fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["predicted_revenue"],
                                 name="Forecast", line=dict(color=PRIMARY, width=2.5, dash="solid")))
        if {"lower", "upper"}.issubset(forecast.columns):
            fig.add_trace(go.Scatter(
                x=list(forecast["date"]) + list(forecast["date"])[::-1],
                y=list(forecast["upper"]) + list(forecast["lower"])[::-1],
                fill="toself", fillcolor="rgba(255,107,53,0.18)",
                line=dict(color="rgba(0,0,0,0)"), name="Confidence band",
                hoverinfo="skip", showlegend=True,
            ))
    fig.update_layout(title="Revenue history with forecast",
                      xaxis_title="Date", yaxis_title="Revenue (INR)",
                      height=460, plot_bgcolor="white", hovermode="x unified")
    return fig


def stacked_bar_category_fulfilment(df: pd.DataFrame, top: int = 8) -> go.Figure:
    if df.empty:
        return _empty_fig("Revenue by category × fulfilment")
    top_cats = df.groupby("category")["revenue"].sum().nlargest(top).index
    sub = df[df["category"].isin(top_cats)]
    pivot = sub.pivot_table(index="category", columns="fulfilment",
                            values="revenue", aggfunc="sum", fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    fig = go.Figure()
    for i, col in enumerate(pivot.columns):
        fig.add_trace(go.Bar(name=col, x=pivot.index, y=pivot[col],
                             marker_color=PALETTE[i % len(PALETTE)]))
    fig.update_layout(barmode="stack", title="Revenue by category × fulfilment",
                      height=420, plot_bgcolor="white",
                      xaxis_title="", yaxis_title="Revenue (INR)")
    return fig
