# Presentation Cheat-sheet

Glance-able reference for the 10–15 minute defense. **Not** a script —
your speech goes elsewhere ([DEFENSE_GUIDE.md](DEFENSE_GUIDE.md)).
Keep this open on a second screen or printed.

---

## ⏱ Timing skeleton

| Min | What you're doing | Slide / page |
|---|---|---|
| 0–2 | Intro: dataset, objective | Slide 1–2 |
| 2–5 | Cleaning walk-through | Slide 3 |
| 5–9 | Stats + EDA highlights | Slide 4–5 |
| 9–12 | Live dashboard demo + 3 insights | Slide 6–7 |
| 12–15 | Forecast + AI Insights + limitations | Slide 8 |
| +5 | Q&A | — |

---

## 🔢 Numbers to memorize

```
Total revenue           ₹71.79 M   (₹7.18 Cr)
Unique orders           120,350
Cancellation rate       14.21 %     ← above healthy 5–10 % band
Avg order value         ₹649
MoM growth              -10.7 %     ← latest full month
Top category            Set
Top state               Maharashtra
Unique cities reached   7,289
Date range              2022-03-31 → 2022-06-29 (91 days)
Raw rows                128,975  →  cleaned 128,936  (-0.03 %)
Forecast horizon        30 days
Best model              Holt-Winters, MAPE = 12.4 %
```

---

## 📐 Statistical tests — one-liners

| Test | Result | What I say |
|---|---|---|
| **Chi-square** cancellation × category | χ² = 61.5, p = 2×10⁻¹⁰ | "Cancellation is **not** independent of category." |
| **One-way ANOVA** amount across categories | F = 10,873, p ≈ 0 | "Category strongly affects order amount." |
| **Mann-Whitney U** B2B vs B2C amount | p = 8×10⁻⁶, B2B median ₹665 vs B2C ₹612 | "B2B orders are statistically larger." |
| **ADF** on daily revenue | p < 0.001 | "Series is stationary — fits Holt-Winters." |

All at α = 0.05.

---

## 🧹 Cleaning — three points to mention

1. **Amount imputation** — 7,795 NAs filled with median by `(category, size)`
   → category → global. Median is robust to skew.
2. **City unification** — BENGALURU/BANGALORE, MUMBAI/BOMBAY etc. merged
   via a dictionary. Without this, geographic aggregates double-count.
3. **Outliers** — flagged at `Q3 + 3·IQR`, **not removed** (high-value
   orders are real). Dashboard offers an opt-in filter.

---

## 📊 Per-page demo flow

| Page | What to click | What to say |
|---|---|---|
| **Overview** | Just open. Maybe shrink date range to last 30 days. | "Six KPIs at the top. Cancellation **14 %** is the headline concern. Notice the donut — about 60 % shipped." |
| **Sales Analytics** | Switch granularity dropdown to "Monthly". Show heatmap. | "Day-of-week × month heatmap shows Sundays peak. Cancellation table at the bottom is the actionable artefact." |
| **Geographic** | Show bubble map. Hover over Maharashtra. | "Maharashtra, Karnataka, Tamil Nadu dominate. 7,289 unique cities — long tail." |
| **Forecasting** | Open. Show 30-day forecast. | "Three models compared on a 30-day hold-out. Holt-Winters wins MAPE **12.4 %**. The orange band is the 95 % interval." |
| **AI Insights** | Show templated bullets. (Skip LLM if no key.) | "Templated bullets always render — pure Python. With an API key, Claude writes a 3-paragraph executive summary from the filtered KPIs only — no raw data leaves the app." |

---

## 🎯 Three key insights (memorize these)

1. **Cancellations are a real problem.** 14.2 % overall; chi-square confirms
   they vary by category. → Investigate the highest-cancel SKUs.
2. **Revenue is declining.** -10.7 % MoM in the last full month;
   Holt-Winters projects continued softness. → Demand-stimulation needed.
3. **Geographic concentration is risk.** Maharashtra + Karnataka +
   Tamil Nadu carry the topline. → Diversify, or invest deeper in this trio.

---

## ⚠️ Limitations — say these *first* before the examiner asks

- **No cost-of-goods column** → I report Revenue, not Profit. Substitution
  documented in the report.
- **No `customer_id`** → Classic RFM segmentation impossible; I use
  order-level analysis.
- **Only 91 days of data** → Forecast is directional, not commit-grade.
- **No exogenous variables** (holidays, marketing spend, prices).

---

## ❓ Q&A — short answers

| Q | A |
|---|---|
| Why this dataset? | Real, large (128k rows × 24 cols), messy enough to demonstrate cleaning, with geography + time + categories — covers every rubric point. |
| Biggest cleaning challenge? | Missing `amount` on 6 % of rows — instead of dropping, I imputed by `(category, size)` median to preserve cancellation context. |
| Why median, not mean? | Right-skewed amount distribution — mean would inflate from the long tail. |
| Why flag outliers instead of removing? | They are real bulk orders, not errors. Removing them would erase a legitimate B2B signal. |
| Why these charts specifically? | Each answers a defined question: bar = which is biggest, line = trend over time, heatmap = two-dimensional pattern, scatter = relationship, geo = concentration. |
| Correlation vs causation? | I never claim causation. Correlations are weak (|r| < 0.25). Chi-square shows *dependence*, not direction. |
| Why a dashboard, not just charts? | Filters let stakeholders explore. Same code answers "what about only Maharashtra in May?" without rerunning the analysis. |
| Why Holt-Winters? | Daily series is stationary (ADF p < 0.001), has weekly seasonality. Holt-Winters with `seasonal_periods=7` won MAPE on hold-out. |
| Why not Prophet? | Heavyweight; Windows wheel issues on Python 3.14. Statsmodels gave us a cleaner install + comparable accuracy. |
| Where does LLM data go? | Only **aggregated KPIs** (revenue, counts, top categories) go to Claude. No raw rows, no PII. The API key sits in Streamlit secrets, never in git. |
| If you had more time? | (a) Pull cost-of-goods to compute real profit. (b) Add holidays/marketing as exogenous regressors for forecasting. (c) Stream the LLM response. (d) Click-through drill-downs on the map. |

---

## 🛟 Crisis recovery

- **App won't load**: open a terminal, `streamlit run app.py` — check the
  log for the actual error. Most common: `forecast.csv` missing → run
  `python notebooks\03_forecasting.py`.
- **A chart looks empty**: a filter is too narrow. Reset filters in
  the sidebar.
- **AI Insights LLM section throws an error**: ignore — the templated
  bullets above it still answer the question.
- **PDF doesn't open**: it's a real PDF; on the project machine open it
  manually from `report\report.pdf`.

---

## 🗣 Opening line (memorize verbatim)

> "I built an end-to-end analytics solution on top of a real Amazon India
> sales dataset — 128 thousand orders, 91 days, all 28 Indian states. The
> project covers four layers: programmatic cleaning, applied statistics,
> an interactive Streamlit dashboard, and two bonus features — a 30-day
> revenue forecast and an LLM-powered insights generator."

## 🗣 Closing line

> "The headline takeaway is that revenue is softening at -10.7 % MoM and
> cancellation rate is 14.2 % — both flagged automatically by the
> dashboard. The forecast suggests continued pressure, so my
> recommendation is to investigate the highest-cancellation categories
> and stimulate demand in the top three states before the trend
> entrenches. I'd be happy to take questions."
