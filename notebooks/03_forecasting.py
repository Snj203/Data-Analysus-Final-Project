"""ML revenue forecasting for the dashboard's Forecasting page.

Compares three models on the last 30 days and projects the next 30 days
using the best performer.

Run:
    python notebooks/03_forecasting.py
"""

# %% [markdown]
# # Notebook 03 — Revenue Forecasting
#
# **Goal**: produce a 30-day daily-revenue forecast that the Streamlit
# dashboard's Forecasting page can render.
#
# **Models compared**:
# 1. Naive baseline — 7-day moving average forward-fill.
# 2. Linear Regression on `[day_index, day_of_week_OH, month_OH]`.
# 3. Holt-Winters Exponential Smoothing (additive seasonality, 7-day cycle).
#
# Best model by MAPE is refit on the full series and used for the live forecast.

# %%
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path.cwd()
    if ROOT.name == "notebooks":
        ROOT = ROOT.parent
CLEAN = ROOT / "data" / "Amazon Sale Report Cleaned.csv"
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = ROOT / "data" / "forecast.csv"

# %% [markdown]
# ## 1. Build the daily revenue series

# %%
df = pd.read_csv(CLEAN, parse_dates=["date"])
daily = (df[~df["is_cancelled"]]
         .groupby(df.loc[~df["is_cancelled"], "date"].dt.date)["revenue"]
         .sum()
         .reset_index())
daily["date"] = pd.to_datetime(daily["date"])
daily = daily.sort_values("date").reset_index(drop=True)
print(f"Series length: {len(daily)} days "
      f"({daily['date'].min().date()} → {daily['date'].max().date()})")

# %% [markdown]
# ## 2. Stationarity check (ADF)

# %%
adf_stat, adf_p, *_ = adfuller(daily["revenue"], autolag="AIC")
print(f"ADF statistic: {adf_stat:.3f}, p-value: {adf_p:.4f}")
print("Series is", "stationary" if adf_p < 0.05 else "NOT stationary",
      "@ α=0.05")

# %% [markdown]
# ## 3. Train / test split (last 30 days = test)

# %%
TEST_DAYS = 30
train = daily.iloc[:-TEST_DAYS].copy()
test = daily.iloc[-TEST_DAYS:].copy()
print(f"Train: {len(train)}  Test: {len(test)}")


# %% [markdown]
# ## 4. Model 1 — Naive 7-day MA forward-fill

# %%
def naive_ma(train_series: pd.Series, horizon: int, window: int = 7):
    last_avg = train_series.tail(window).mean()
    return np.repeat(last_avg, horizon)


pred_naive = naive_ma(train["revenue"], len(test))


# %% [markdown]
# ## 5. Model 2 — Linear Regression on time features

# %%
def build_time_features(idx_dates: pd.Series, t0: pd.Timestamp) -> pd.DataFrame:
    day_index = (idx_dates - t0).dt.days.values
    dow = pd.get_dummies(idx_dates.dt.dayofweek, prefix="dow")
    mon = pd.get_dummies(idx_dates.dt.month, prefix="mon")
    feats = pd.concat([pd.DataFrame({"day_index": day_index}),
                       dow.reset_index(drop=True),
                       mon.reset_index(drop=True)], axis=1)
    return feats


t0 = train["date"].min()
X_train = build_time_features(train["date"], t0)
X_test = build_time_features(test["date"], t0)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

lr = LinearRegression().fit(X_train, train["revenue"])
pred_lr = lr.predict(X_test)

# %% [markdown]
# ## 6. Model 3 — Holt-Winters Exponential Smoothing

# %%
hw_model = ExponentialSmoothing(
    train["revenue"].values,
    trend="add",
    seasonal="add",
    seasonal_periods=7,
    initialization_method="estimated",
).fit(optimized=True)
pred_hw = hw_model.forecast(len(test))


# %% [markdown]
# ## 7. Evaluation

# %%
def metrics(name: str, y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = y_true_arr != 0
    mape = float(np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100)
    return {"model": name, "RMSE": rmse, "MAE": mae, "MAPE_pct": mape}


results = pd.DataFrame([
    metrics("Naive 7-day MA", test["revenue"], pred_naive),
    metrics("Linear Regression", test["revenue"], pred_lr),
    metrics("Holt-Winters", test["revenue"], pred_hw),
]).round(2)
print("\nModel comparison:")
print(results.to_string(index=False))

best_model_name = results.sort_values("MAPE_pct").iloc[0]["model"]
print(f"\nBest model by MAPE: {best_model_name}")

# %% [markdown]
# ## 8. Visualise validation fit

# %%
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(train["date"], train["revenue"], color="#1F77B4",
        alpha=0.45, label="Train")
ax.plot(test["date"], test["revenue"], color="#1F77B4",
        linewidth=2, label="Test (actual)")
ax.plot(test["date"], pred_naive, label="Naive MA",
        color="#9CA3AF", linestyle=":")
ax.plot(test["date"], pred_lr, label="Linear Regression",
        color="#059669", linestyle="--")
ax.plot(test["date"], pred_hw, label="Holt-Winters",
        color="#FF6B35", linewidth=2)
ax.set_title("Forecast validation on held-out 30-day window")
ax.set_xlabel("Date")
ax.set_ylabel("Revenue (INR)")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG_DIR / "10_forecast_validation.png")
plt.close(fig)

# %% [markdown]
# ## 9. Refit best model on full data and forecast next 30 days

# %%
HORIZON = 30
full_t0 = daily["date"].min()

if best_model_name == "Holt-Winters":
    final = ExponentialSmoothing(
        daily["revenue"].values,
        trend="add", seasonal="add", seasonal_periods=7,
        initialization_method="estimated",
    ).fit(optimized=True)
    point = final.forecast(HORIZON)
    residuals = daily["revenue"].values - final.fittedvalues
    sigma = float(np.std(residuals))
elif best_model_name == "Linear Regression":
    X_full = build_time_features(daily["date"], full_t0)
    lr_full = LinearRegression().fit(X_full, daily["revenue"])
    future_dates = pd.date_range(daily["date"].max() + pd.Timedelta(days=1),
                                 periods=HORIZON, freq="D")
    X_fut = build_time_features(pd.Series(future_dates), full_t0)
    X_fut = X_fut.reindex(columns=X_full.columns, fill_value=0)
    point = lr_full.predict(X_fut)
    sigma = float(np.std(daily["revenue"].values - lr_full.predict(X_full)))
else:
    point = naive_ma(daily["revenue"], HORIZON)
    sigma = float(np.std(daily["revenue"].diff().dropna()))

future_dates = pd.date_range(daily["date"].max() + pd.Timedelta(days=1),
                             periods=HORIZON, freq="D")
forecast = pd.DataFrame({
    "date": future_dates,
    "predicted_revenue": np.maximum(point, 0),
    "lower": np.maximum(point - 1.96 * sigma, 0),
    "upper": point + 1.96 * sigma,
    "model": best_model_name,
})
forecast.to_csv(OUT_CSV, index=False)
print(f"\nWrote 30-day forecast → {OUT_CSV}")
print(forecast.head().to_string(index=False))

# %% [markdown]
# ## 10. Final visualisation — history + forecast

# %%
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(daily["date"], daily["revenue"], color="#1F77B4",
        alpha=0.7, label="Historical")
ax.plot(forecast["date"], forecast["predicted_revenue"],
        color="#FF6B35", linewidth=2.4, label=f"Forecast ({best_model_name})")
ax.fill_between(forecast["date"], forecast["lower"], forecast["upper"],
                color="#FF6B35", alpha=0.2, label="95% interval")
ax.set_title("Daily revenue — history and 30-day forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Revenue (INR)")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG_DIR / "11_forecast_final.png")
plt.close(fig)

print("\nDone.")
