"""Data cleaning pipeline for the Amazon Sale Report dataset.

This file is the source-of-truth for the cleaning logic. It is structured as
Jupytext-style cells (# %% headers) so it can be converted to a .ipynb
notebook with `python build_notebooks.py`.

Running this as a plain script produces the cleaned CSV the dashboard
depends on:

    python notebooks/01_cleaning.py
"""

# %% [markdown]
# # Notebook 01 — Data Cleaning
#
# **Dataset**: `data/Amazon Sale Report.csv` — 128,975 raw rows of Amazon India
# sales transactions (April–June 2022), 24 columns.
#
# **Goal**: produce a tidy `data/Amazon Sale Report Cleaned.csv` that
# downstream notebooks and the Streamlit dashboard consume.
#
# Every cleaning decision below is structured as **What / Why / How verified**.

# %%
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()
    if PROJECT_ROOT.name == "notebooks":
        PROJECT_ROOT = PROJECT_ROOT.parent
RAW = PROJECT_ROOT / "data" / "Amazon Sale Report.csv"
OUT = PROJECT_ROOT / "data" / "Amazon Sale Report Cleaned.csv"

# %% [markdown]
# ## 1. Load and inspect raw data
#
# **What**: read the CSV into a DataFrame, print shape, dtypes, and a head.
# **Why**: every cleaning decision needs evidence — we cannot fix what we
# haven't measured.

# %%
df_raw = pd.read_csv(RAW, low_memory=False)
print("Raw shape:", df_raw.shape)
print("\nColumn dtypes:")
print(df_raw.dtypes)
print("\nMissing-value counts (top 10):")
print(df_raw.isna().sum().sort_values(ascending=False).head(10))

# %% [markdown]
# ## 2. Column-name standardization
#
# **What**: lower-case, strip whitespace, replace spaces/dashes with underscores,
# drop the trailing space in `Sales Channel `.
# **Why**: spaces and inconsistent casing make downstream code fragile and
# error-prone. Standardized snake_case is the de-facto pandas convention.

# %%
df = df_raw.copy()
df.columns = [
    c.strip().lower().replace(" ", "_").replace("-", "_")
    for c in df.columns
]
print("Renamed columns:", df.columns.tolist())

# %% [markdown]
# ## 3. Drop artifact columns
#
# **What**: drop `unnamed:_22` and `index` — `unnamed:_22` is empty trailing
# delimiter from the CSV export; `index` duplicates the DataFrame index.

# %%
drop_cols = [c for c in ("unnamed:_22", "index") if c in df.columns]
df = df.drop(columns=drop_cols)
print("Dropped artifact columns:", drop_cols)

# %% [markdown]
# ## 4. Type conversion
#
# **What**: parse `date` (format `MM-DD-YY`), coerce `amount` and `qty` to
# numeric, normalize `b2b` to boolean.
# **Why**: arithmetic and grouping only behave correctly with the right dtypes.

# %%
df["date"] = pd.to_datetime(df["date"], format="%m-%d-%y", errors="coerce")
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
df["b2b"] = df["b2b"].astype("boolean").fillna(False).astype(bool)

bad_dates = df["date"].isna().sum()
print(f"Rows with un-parseable dates: {bad_dates}")
if bad_dates:
    df = df.dropna(subset=["date"])
print("After date filter:", df.shape)

# %% [markdown]
# ## 5. Missing-value handling
#
# Per-column rationale:
#
# | column | strategy | reason |
# |---|---|---|
# | `amount` | impute median by `(category, size)` → category → drop | preserves info; median is robust to skew |
# | `courier_status` | fill `'Unknown'` | NA = courier hasn't picked up yet — meaningful |
# | `promotion_ids` | binarize → `has_promotion`, drop original | the long ID string is not analytically useful |
# | `fulfilled_by` | fill `'Other'` | NA appears for Amazon-fulfilled orders |
# | `ship_*` | drop rows where all of `city/state/postal_code` are NA | un-locatable; small fraction |
# | `currency` | drop (always INR after filtering bad rows) | constant column |

# %%
before = df["amount"].isna().sum()
cat_size_median = df.groupby(["category", "size"])["amount"].transform("median")
df["amount"] = df["amount"].fillna(cat_size_median)
cat_median = df.groupby("category")["amount"].transform("median")
df["amount"] = df["amount"].fillna(cat_median)
df["amount"] = df["amount"].fillna(df["amount"].median())
after = df["amount"].isna().sum()
print(f"`amount` NaN before: {before:,} → after: {after:,}")

df["courier_status"] = df["courier_status"].fillna("Unknown")
df["has_promotion"] = df["promotion_ids"].notna()
df = df.drop(columns=["promotion_ids"])
df["fulfilled_by"] = df["fulfilled_by"].fillna("Other")

ship_cols = ["ship_city", "ship_state", "ship_postal_code"]
all_na = df[ship_cols].isna().all(axis=1)
print(f"Rows with no shipping info: {all_na.sum()} (dropping)")
df = df[~all_na].reset_index(drop=True)

if "currency" in df.columns:
    df = df.drop(columns=["currency"])

print("Post-NA-handling shape:", df.shape)

# %% [markdown]
# ## 6. Text standardization — cities and states
#
# **What**: upper-case, strip whitespace, unify duplicate-name pairs.
# **Why**: BANGALORE/BENGALURU, BOMBAY/MUMBAI etc. are the same place under
# different colonial/native names and must be merged for accurate geographic
# aggregation.

# %%
for col in ("ship_city", "ship_state"):
    df[col] = df[col].astype(str).str.strip().str.upper()

CITY_UNIFICATION = {
    "BANGALORE": "BENGALURU",
    "BOMBAY": "MUMBAI",
    "CALCUTTA": "KOLKATA",
    "MADRAS": "CHENNAI",
    "PONDICHERRY": "PUDUCHERRY",
    "BENARES": "VARANASI",
    "TRIVANDRUM": "THIRUVANANTHAPURAM",
    "POONA": "PUNE",
    "MYSORE": "MYSURU",
    "MANGALORE": "MANGALURU",
    "GURGAON": "GURUGRAM",
    "VIZAG": "VISAKHAPATNAM",
}
df["ship_city"] = df["ship_city"].replace(CITY_UNIFICATION)

STATE_FIX = {
    "RAJSHTHAN": "RAJASTHAN",
    "RAJSTHAN": "RAJASTHAN",
    "PB": "PUNJAB",
    "AR": "ARUNACHAL PRADESH",
    "NL": "NAGALAND",
    "ORISSA": "ODISHA",
    "PONDICHERRY": "PUDUCHERRY",
}
df["ship_state"] = df["ship_state"].replace(STATE_FIX)

print(f"Unique cities after unification: {df['ship_city'].nunique():,}")
print(f"Unique states after unification: {df['ship_state'].nunique():,}")

# %% [markdown]
# ## 7. Duplicate handling
#
# **What**: check for exact duplicates on the natural key
# `(order_id, sku, qty, amount)`.
# **Why**: a duplicate row would double-count revenue for that line item.

# %%
key = ["order_id", "sku", "qty", "amount"]
dupes = df.duplicated(subset=key).sum()
print(f"Exact duplicates on {key}: {dupes:,}")
df = df.drop_duplicates(subset=key).reset_index(drop=True)
print("Post-dedup shape:", df.shape)

# %% [markdown]
# ## 8. Outlier detection
#
# **What**: flag `amount` and `qty` outliers using a generous IQR fence
# (Q3 + 3·IQR). Keep them in the dataset — they are real high-value orders —
# but expose a boolean column so the dashboard can offer an opt-in filter.

# %%
def iqr_outlier_flag(s: pd.Series, k: float = 3.0) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + k * iqr
    return s > upper

df["is_amount_outlier"] = iqr_outlier_flag(df["amount"])
df["is_qty_outlier"] = iqr_outlier_flag(df["qty"])
print(f"Amount outliers: {df['is_amount_outlier'].sum():,} "
      f"({df['is_amount_outlier'].mean() * 100:.2f}%)")
print(f"Qty outliers:    {df['is_qty_outlier'].sum():,}")

# %% [markdown]
# ## 9. Feature engineering
#
# Add the time-feature columns the dashboard and forecasting notebook rely on,
# plus a `revenue` column that zeroes out cancelled orders so summing it gives
# realized revenue.

# %%
df["order_year"] = df["date"].dt.year
df["order_month"] = df["date"].dt.to_period("M").astype(str)
df["order_weekday"] = df["date"].dt.day_name()
df["order_day"] = df["date"].dt.day

df["is_cancelled"] = df["status"].str.contains("Cancel", case=False, na=False)
df["is_delivered"] = df["status"].str.contains("Delivered", case=False, na=False)
df["revenue"] = np.where(df["is_cancelled"], 0.0, df["amount"]).astype(float)

print(df[["date", "order_month", "order_weekday", "is_cancelled", "revenue"]].head())

# %% [markdown]
# ## 10. Validation
#
# Re-check NAs, dtypes, basic invariants.

# %%
print("\nFinal NA counts:")
print(df.isna().sum().sort_values(ascending=False).head(10))
print("\nFinal dtypes:")
print(df.dtypes)
assert df["amount"].notna().all(), "amount still has NAs"
assert df["date"].notna().all(), "date still has NAs"
assert (df["revenue"] >= 0).all(), "negative revenue detected"
print("\nValidation passed ✓")

# %% [markdown]
# ## 11. Export

# %%
df.to_csv(OUT, index=False)
print(f"Wrote {len(df):,} rows × {df.shape[1]} cols → {OUT}")
