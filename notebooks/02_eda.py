"""Exploratory Data Analysis & Statistics on the cleaned dataset.

Produces figures used by the final PDF report.

Run:
    python notebooks/02_eda.py
"""

# %% [markdown]
# # Notebook 02 — EDA & Statistics
#
# **Input**: `data/Amazon Sale Report Cleaned.csv` (output of notebook 01).
#
# This notebook produces the analytical evidence that backs every claim in the
# final report. Every chart is exported to `report/figures/` for inclusion in
# `report/report.md`.

# %%
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path.cwd()
    if ROOT.name == "notebooks":
        ROOT = ROOT.parent
CLEAN = ROOT / "data" / "Amazon Sale Report Cleaned.csv"
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CLEAN, parse_dates=["date"], low_memory=False)
print("Loaded:", df.shape)

# %% [markdown]
# ## 1. Descriptive statistics

# %%
print("\nNumeric describe:")
print(df[["amount", "qty", "revenue"]].describe().round(2))

print("\nCategory counts (top):")
print(df["category"].value_counts().head(10))

print("\nStatus counts:")
print(df["status"].value_counts())

# %% [markdown]
# ## 2. Distribution analysis

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["amount"].dropna(), bins=60, kde=True, ax=axes[0], color="#FF6B35")
axes[0].set_title("Amount distribution (raw)")
axes[0].set_xlabel("Order amount (INR)")

sns.histplot(np.log1p(df["amount"].dropna()), bins=60, kde=True,
             ax=axes[1], color="#1F77B4")
axes[1].set_title("Amount distribution (log-scaled)")
axes[1].set_xlabel("log(1 + amount)")
fig.suptitle("Order amount — raw vs log-scaled")
fig.tight_layout()
fig.savefig(FIG_DIR / "01_amount_distribution.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 5))
sns.boxplot(data=df, x="category", y="amount", ax=ax, showfliers=False)
ax.set_title("Order amount by category (outliers hidden)")
ax.set_xlabel("")
ax.set_ylabel("Amount (INR)")
plt.xticks(rotation=30, ha="right")
fig.tight_layout()
fig.savefig(FIG_DIR / "02_amount_by_category_box.png")
plt.close(fig)

# %% [markdown]
# ## 3. Time-trend analysis

# %%
daily = df.groupby(df["date"].dt.date)["revenue"].sum()
daily.index = pd.to_datetime(daily.index)

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(daily.index, daily.values, alpha=0.45, label="Daily revenue", color="#1F77B4")
ax.plot(daily.index, daily.rolling(7, min_periods=1).mean(),
        color="#FF6B35", linewidth=2.4, label="7-day MA")
ax.set_title("Daily revenue with 7-day moving average")
ax.set_xlabel("Date")
ax.set_ylabel("Revenue (INR)")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG_DIR / "03_daily_revenue_ma.png")
plt.close(fig)

monthly = df.groupby(df["date"].dt.to_period("M"))["revenue"].sum()
mom = monthly.pct_change() * 100
fig, ax1 = plt.subplots(figsize=(9, 4.5))
monthly.plot(kind="bar", ax=ax1, color="#1F77B4", alpha=0.8, label="Monthly revenue")
ax1.set_ylabel("Revenue (INR)", color="#1F77B4")
ax1.set_xlabel("Month")
ax1.set_title("Monthly revenue and month-over-month growth")
ax2 = ax1.twinx()
ax2.plot(range(len(mom)), mom.values, color="#FF6B35", marker="o",
         linewidth=2, label="MoM growth %")
ax2.set_ylabel("MoM growth (%)", color="#FF6B35")
ax2.axhline(0, color="#9CA3AF", linestyle="--", linewidth=0.8)
fig.tight_layout()
fig.savefig(FIG_DIR / "04_monthly_revenue_growth.png")
plt.close(fig)

# %% [markdown]
# ## 4. Categorical / comparative analysis

# %%
fig, ax = plt.subplots(figsize=(10, 5))
top_cat = df.groupby("category")["revenue"].sum().sort_values(ascending=True)
top_cat.plot(kind="barh", ax=ax, color="#FF6B35")
ax.set_title("Revenue by category")
ax.set_xlabel("Revenue (INR)")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(FIG_DIR / "05_revenue_by_category.png")
plt.close(fig)

cancel_by_cat = (df.groupby("category")
                 .agg(orders=("order_id", "count"),
                      cancelled=("is_cancelled", "sum"))
                 .assign(rate=lambda d: d["cancelled"] / d["orders"] * 100)
                 .sort_values("rate", ascending=True))
fig, ax = plt.subplots(figsize=(10, 5))
cancel_by_cat["rate"].plot(kind="barh", ax=ax, color="#DC2626")
ax.set_title("Cancellation rate by category")
ax.set_xlabel("Cancellation rate (%)")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(FIG_DIR / "06_cancellation_rate.png")
plt.close(fig)

top_states = (df.groupby("ship_state")["revenue"].sum()
              .sort_values(ascending=False).head(15)
              .sort_values(ascending=True))
fig, ax = plt.subplots(figsize=(10, 6))
top_states.plot(kind="barh", ax=ax, color="#1F77B4")
ax.set_title("Top 15 states by revenue")
ax.set_xlabel("Revenue (INR)")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(FIG_DIR / "07_top_states.png")
plt.close(fig)

# %% [markdown]
# ## 5. Correlation analysis

# %%
corr_cols = ["amount", "qty", "is_cancelled", "has_promotion", "b2b",
             "is_amount_outlier", "is_qty_outlier"]
corr_df = df[corr_cols].astype(float)
pearson = corr_df.corr(method="pearson")
spearman = corr_df.corr(method="spearman")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(pearson, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, ax=axes[0], square=True, cbar_kws={"shrink": 0.7})
axes[0].set_title("Pearson correlation")
sns.heatmap(spearman, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, ax=axes[1], square=True, cbar_kws={"shrink": 0.7})
axes[1].set_title("Spearman correlation")
fig.tight_layout()
fig.savefig(FIG_DIR / "08_correlations.png")
plt.close(fig)

# %% [markdown]
# ## 6. Statistical hypothesis tests

# %%
print("\n--- Chi-square: cancellation × category ---")
ct = pd.crosstab(df["category"], df["is_cancelled"])
chi2, p, dof, _ = stats.chi2_contingency(ct)
print(f"chi2 = {chi2:.2f}, dof = {dof}, p = {p:.2e}")
print("Decision @ α=0.05:", "REJECT H0 — cancellation depends on category"
      if p < 0.05 else "FAIL to reject H0")

print("\n--- One-way ANOVA: amount differs across categories? ---")
groups = [g["amount"].dropna().values for _, g in df.groupby("category")]
F, p_anova = stats.f_oneway(*groups)
print(f"F = {F:.2f}, p = {p_anova:.2e}")
print("Decision @ α=0.05:", "REJECT H0 — category affects amount"
      if p_anova < 0.05 else "FAIL to reject H0")

print("\n--- Mann-Whitney U: B2B vs B2C order amount ---")
b2b_amt = df.loc[df["b2b"], "amount"].dropna()
b2c_amt = df.loc[~df["b2b"], "amount"].dropna()
u, p_mw = stats.mannwhitneyu(b2b_amt, b2c_amt, alternative="two-sided")
print(f"U = {u:.0f}, p = {p_mw:.2e}")
print(f"Median amount B2B: {b2b_amt.median():.0f} | B2C: {b2c_amt.median():.0f}")

# %% [markdown]
# ## 7. Day-of-week × month heatmap

# %%
order = ["Monday", "Tuesday", "Wednesday", "Thursday",
         "Friday", "Saturday", "Sunday"]
pivot = (df.assign(month=df["date"].dt.strftime("%Y-%m"))
           .pivot_table(index="order_weekday", columns="month",
                        values="revenue", aggfunc="sum", fill_value=0)
           .reindex(order))
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.heatmap(pivot, cmap="Oranges", annot=False, cbar_kws={"label": "Revenue (INR)"}, ax=ax)
ax.set_title("Revenue heatmap — Weekday × Month")
ax.set_xlabel("Month")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(FIG_DIR / "09_heatmap_dow_month.png")
plt.close(fig)

# %% [markdown]
# ## 8. Summary insights

# %%
total_rev = df["revenue"].sum()
total_orders = df["order_id"].nunique()
cancel_rate = df["is_cancelled"].mean() * 100
top_category = df.groupby("category")["revenue"].sum().idxmax()
top_state = df.groupby("ship_state")["revenue"].sum().idxmax()

print("\n========== Headline metrics ==========")
print(f"Total revenue (shipped orders):     ₹{total_rev:,.0f}")
print(f"Unique orders:                       {total_orders:,}")
print(f"Cancellation rate:                   {cancel_rate:.2f}%")
print(f"Top category by revenue:             {top_category}")
print(f"Top state by revenue:                {top_state}")
print(f"Date range:                          "
      f"{df['date'].min().date()} → {df['date'].max().date()}")
print("======================================")

print(f"\nFigures saved to: {FIG_DIR}")
