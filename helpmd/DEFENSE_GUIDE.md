# Project Defense Guide

The single document you can read once and feel confident about the entire
project: what was built, why each decision was made, what every chart
shows, what every dashboard page does, and how to answer the questions
the examiner is most likely to ask.

For the short during-the-talk reference, use [PRESENTATION_CHEATSHEET.md](PRESENTATION_CHEATSHEET.md).

---

## Table of contents

1. [The project in one minute](#1-the-project-in-one-minute)
2. [Dataset deep dive](#2-dataset-deep-dive)
3. [The pipeline](#3-the-pipeline)
4. [Cleaning — every decision explained](#4-cleaning--every-decision-explained)
5. [Statistical analysis — every test explained](#5-statistical-analysis--every-test-explained)
6. [Every chart, with the insight behind it](#6-every-chart-with-the-insight-behind-it)
7. [Dashboard pages — feature by feature](#7-dashboard-pages--feature-by-feature)
8. [ML Forecasting — methodology and results](#8-ml-forecasting--methodology-and-results)
9. [AI Insights — design and safety](#9-ai-insights--design-and-safety)
10. [Limitations & honesty](#10-limitations--honesty)
11. [The 8-slide deck and 15-minute speech](#11-the-8-slide-deck-and-15-minute-speech)
12. [Q&A — long-form answers](#12-qa--long-form-answers)

---

## 1. The project in one minute

**What it is.** An end-to-end business-analytics solution on the
*Amazon Sale Report* dataset — 128,975 real transactions from Amazon
India between 2022-03-31 and 2022-06-29 across 9 product categories and
all 28 Indian states.

**Why this is a strong project.** It covers all five rubric pillars:
**completeness** (cleaning → stats → EDA → dashboard → forecasting →
LLM), **code quality** (modular `src/`, three executable notebooks,
documented every decision), **insights** (three numerically-backed
findings), **visualization** (seven required chart types, all dynamic),
**documentation** (this guide + report PDF + plan.txt + project.md).

**The business question I built it to answer.**
> Which products, regions, and channels are driving Amazon India revenue
> — and where is the business losing money to cancellations?

**The headline answer.** Set, Kurta, and Western Dress drive most of the
₹71.79 M revenue. Maharashtra, Karnataka, and Tamil Nadu dominate
geography. Cancellation rate is 14.21 % — well above the healthy 5–10 %
band — and varies by category, which makes it the single highest-impact
lever in the business.

---

## 2. Dataset deep dive

| Property | Value |
|---|---|
| **Source** | [Amazon Sale Report on Kaggle](https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data) |
| **Raw shape** | 128,975 rows × 24 columns |
| **Cleaned shape** | 128,936 rows × 30 columns (39 unlocatable rows dropped + 6 engineered features) |
| **Time span** | 2022-03-31 → 2022-06-29 (91 days) |
| **Currency** | INR only after cleaning |
| **Geography** | All 28 Indian states + 8 UTs, ~7,289 unique cities |
| **Categories** | 9 (Set, Kurta, Western Dress, Top, Ethnic Dress, Bottoms, Saree, Blouse, Dupatta) |

**Columns we use most**:
- `order_id`, `date`, `status` (Shipped / Cancelled / Pending / …)
- `category`, `size`, `sku`, `qty`, `amount` (INR)
- `ship_city`, `ship_state`, `ship_postal_code`
- `fulfilment` (Amazon vs Merchant), `b2b` (bool), `promotion_ids`
- engineered: `revenue`, `is_cancelled`, `is_delivered`,
  `has_promotion`, `is_amount_outlier`, `is_qty_outlier`, plus
  `order_month`, `order_year`, `order_weekday`, `order_day`.

**Why this dataset fits the course.**
- It easily clears the 5,000-row × 10-column minimum (25× and 2× over).
- It is real-world data, not a contrived teaching set.
- It is messy in interesting ways — missing values, duplicate city
  spellings, typed-as-string dates, outliers — so cleaning has *teeth*.
- It spans geography, time, and categories — every required chart type
  has a natural question to answer.

---

## 3. The pipeline

```
data/Amazon Sale Report.csv  (raw)
        │
        ▼
notebooks/01_data_cleaning.ipynb       ──►  data/Amazon Sale Report Cleaned.csv
        │
        ▼
notebooks/02_eda_statistics.ipynb      ──►  report/figures/*.png
        │
        ▼
notebooks/03_forecasting.ipynb         ──►  data/forecast.csv
        │
        ▼
[Streamlit app.py + pages/]
   ├── Overview              (reads cleaned CSV)
   ├── Sales Analytics       (reads cleaned CSV)
   ├── Geographic            (reads cleaned CSV)
   ├── Forecasting           (reads cleaned CSV + forecast.csv)
   └── AI Insights           (reads cleaned CSV + Anthropic API)
        │
        ▼
report/report.md  ──►  report/build_pdf.py  ──►  report/report.pdf
```

**One-sentence per layer.**
- *Notebooks* are reproducible — running them top-to-bottom recreates
  every artifact.
- *`src/`* is shared infrastructure — `data_loader`, `kpis`, `charts`,
  `llm_insights`. Pages stay focused on layout.
- *Streamlit* reads only the cleaned CSV and `forecast.csv`. It never
  re-runs cleaning at request time.

---

## 4. Cleaning — every decision explained

The full code is in `notebooks/01_data_cleaning.ipynb`. Below is the
*why* you'll be asked about.

### 4.1 Column name standardization

Raw columns had inconsistent casing, spaces, and one trailing-space
column name (`Sales Channel `). I normalised everything to `snake_case`.

> **Why**: spaces and inconsistent casing make downstream code fragile.
> Standardised snake_case is the pandas convention.

### 4.2 Dropped two columns outright

- `Unnamed: 22` — 38 % NA, no information content (CSV export artifact).
- `index` — duplicates the DataFrame index.

> **Why**: keeping them would pollute every later groupby and increase
> file size with zero analytical value.

### 4.3 Date parsing

Raw dates were strings like `"04-30-22"`. Parsed with explicit
`format="%m-%d-%y"`.

> **Why**: time-based grouping requires `datetime`. Explicit format
> prevents `pandas` from guessing wrong on Indian-style D-M-Y rows.

### 4.4 Missing values — per-column rationale

| Column | NA count | Strategy | Why |
|---|---|---|---|
| `amount` | 7,795 (6.0 %) | Impute median by `(category, size)` → `category` → global | Median is robust to skew; missing amount on cancelled rows is meaningful, not dropped |
| `courier_status` | 6,872 | Fill `'Unknown'` | NA = courier hasn't picked up yet — a real state |
| `promotion_ids` | 49,153 | Binarise → `has_promotion`, drop original | The long composite ID string is not useful as a categorical |
| `fulfilled_by` | 89,698 | Fill `'Other'` | NA corresponds to Amazon-fulfilled orders — needs a name |
| All three `ship_*` cols NA | 33 rows | Drop the rows | Un-locatable, < 0.03 % |
| `currency` | 7,795 | Drop column | After NA handling it's constant (INR) |

**Why no row-dropping for amount?** Cancelled orders frequently have NA
amount. Dropping them would systematically remove cancellation evidence
— catastrophic for the 14 % cancellation insight.

**Why median, not mean, for amount?** The amount distribution is
strongly right-skewed (long tail of high-value orders). Mean would pull
imputations upward; median is robust.

### 4.5 City / state text standardization

- Upper-case, strip whitespace.
- Unification dictionary: BANGALORE→BENGALURU, BOMBAY→MUMBAI,
  CALCUTTA→KOLKATA, MADRAS→CHENNAI, PONDICHERRY→PUDUCHERRY,
  GURGAON→GURUGRAM, plus state typo fixes (ORISSA→ODISHA,
  RAJSTHAN→RAJASTHAN).

> **Why**: those are the same place under colonial vs native names.
> Without unification, the Geographic page double-counts and the
> top-city ranking is wrong.

### 4.6 Duplicate handling

Checked exact duplicates on `(order_id, sku, qty, amount)` — the natural
line-item key. Six found, dropped.

> **Why**: a duplicate line would double-count revenue for that SKU.

### 4.7 Outlier detection — flag, don't drop

IQR fence at `Q3 + 3·IQR`:
- `amount` outliers: 1.3 % of rows
- `qty` outliers: 3.0 % of rows

Both flagged in boolean columns (`is_amount_outlier`, `is_qty_outlier`)
and the dashboard offers a sidebar toggle to exclude them. **The rows
are kept.**

> **Why**: high-value orders are *real* — they're often B2B bulk
> purchases (Mann-Whitney later confirms this). Removing them would
> erase a legitimate signal. Flagging respects both interpretations.

### 4.8 Engineered features

| Feature | Definition | Used by |
|---|---|---|
| `order_year`, `order_month`, `order_weekday`, `order_day` | Time decompositions | Heatmaps, trends, MoM growth |
| `is_cancelled` | `status.contains("Cancel")` | Cancellation rate KPI |
| `is_delivered` | `status.contains("Delivered")` | Delivery-rate analyses |
| `has_promotion` | `promotion_ids.notna()` | Correlation analysis |
| `revenue` | `amount` where not cancelled, else 0 | Every revenue chart and KPI |

> **Why `revenue` and not just `amount`?** Cancelled orders are placed
> but generate zero income. Summing `amount` would over-state revenue
> by ~16 % (cancellation rate × avg amount). `revenue` makes the KPI
> arithmetic honest.

### 4.9 Validation

After cleaning: `amount` and `date` are NA-free; `revenue` is
non-negative; final shape is 128,936 × 30. Assertions in the notebook
guard against regression.

---

## 5. Statistical analysis — every test explained

Notebook 02 runs three formal tests at α = 0.05, plus correlations and
descriptives.

### 5.1 Chi-square independence — cancellation × category

| | |
|---|---|
| **Question** | Is cancellation independent of product category? |
| **H₀** | Cancellation is independent of category. |
| **Result** | χ² = 61.51, df = 8, p = 2.35 × 10⁻¹⁰ |
| **Decision** | Reject H₀ |
| **Interpretation** | Different categories have meaningfully different cancellation rates. This *justifies* category-targeted interventions. |
| **Caveat** | Chi-square detects *dependence*, not direction or causality. |

### 5.2 One-way ANOVA — order amount across categories

| | |
|---|---|
| **Question** | Do mean order amounts differ across categories? |
| **H₀** | All category means are equal. |
| **Result** | F = 10,873.3, p ≈ 0 (numerically < 1 × 10⁻¹⁰) |
| **Decision** | Reject H₀ |
| **Interpretation** | Category is a strong driver of order basket size. The ranking we see (Set > Western Dress > …) is not a coincidence. |
| **Caveat** | ANOVA assumes normality — with 128k rows the central-limit theorem covers us, but the result mainly tells us *some* pair differs, not *which*. |

### 5.3 Mann-Whitney U — B2B vs B2C amounts

| | |
|---|---|
| **Question** | Do B2B orders have a different amount distribution than B2C? |
| **H₀** | The distributions are identical. |
| **Result** | U = 60.66 M, p = 8 × 10⁻⁶; B2B median = ₹665 vs B2C median = ₹612 |
| **Decision** | Reject H₀ |
| **Interpretation** | B2B orders are *statistically* larger. The effect is small in absolute terms (~9 %) but real — a candidate for a dedicated B2B onboarding flow. |
| **Why Mann-Whitney, not t-test** | Amount is right-skewed; Mann-Whitney is the rank-based non-parametric alternative that doesn't assume normality. |

### 5.4 Correlations

Pearson and Spearman matrices on `[amount, qty, is_cancelled, has_promotion, b2b, is_amount_outlier, is_qty_outlier]`.

All non-trivial correlations are **weak** (|r| < 0.25). Strongest:
- `amount` ↔ `is_amount_outlier` (mechanical, by construction)
- `qty` ↔ `is_qty_outlier` (same)
- Weak positive `amount` ↔ `b2b` (matches Mann-Whitney)
- Weak negative `amount` ↔ `is_cancelled`

> **What to say**: "The data does not show strong linear relationships
> between these business variables. That itself is an insight — order
> amount is driven mostly by *category* (per ANOVA), not by
> promotion-status or B2B-status."

---

## 6. Every chart, with the insight behind it

Every figure lives in `report/figures/` and is embedded in the PDF.

### Figure 1 — `01_amount_distribution.png`
**Type**: side-by-side histograms (raw + log-scaled)
**What it shows**: the raw amount distribution is strongly right-skewed
— most orders cluster ₹300–₹1,000 with a long tail to ₹5,000+. The
log-transformed view is approximately normal.
**Insight**: classic e-commerce shape; justifies median (not mean) for
imputation and a log scale wherever needed downstream.
**One-liner during demo**: *"Amount is right-skewed — many small orders,
a long tail of large ones. Log transform makes it look like a normal
bell curve, which is what we expect."*

### Figure 2 — `02_amount_by_category_box.png`
**Type**: boxplots, amount by category (outliers hidden)
**What it shows**: Set and Western Dress have higher medians; Top and
Bottom skew lower. Variance differs by category.
**Insight**: not all categories are commercially equivalent — some sell
in higher price brackets.
**One-liner**: *"Set and Western Dress are the premium lines by median
basket size."*

### Figure 3 — `03_daily_revenue_ma.png`
**Type**: daily revenue line + 7-day moving average overlay
**What it shows**: daily volatility ~₹400–800 K, with a soft mid-May
peak and a sustained slide afterward.
**Insight**: there is a real downward trend in the second half of the
window — not just noise.
**One-liner**: *"The 7-day moving average makes the trend obvious —
revenue peaked in mid-May and has been softening."*

### Figure 4 — `04_monthly_revenue_growth.png`
**Type**: dual-axis: monthly revenue bars + MoM growth % line
**What it shows**: April, May, June revenue with -10.7 % MoM in the
latest month.
**Insight**: the dashboard's KPI card surfaces this -10.7 % automatically.
**One-liner**: *"The latest month is down 10.7 % vs the previous one —
the KPI card flags this automatically."*

### Figure 5 — `05_revenue_by_category.png`
**Type**: horizontal bar chart
**What it shows**: Set is the dominant revenue line, followed by Kurta
and Western Dress. The bottom five categories together contribute a
small fraction.
**Insight**: top-three concentration is a risk — a dip in any of these
hits the topline hard.
**One-liner**: *"Three categories carry the business. That's concentration
risk, which is something a CFO would want to see flagged."*

### Figure 6 — `06_cancellation_rate.png`
**Type**: horizontal bar chart, cancellation rate by category
**What it shows**: significant variance — some niche categories hit
> 18 % cancellation. Headline 14.2 % is not uniform.
**Insight**: backed by the chi-square test (p < 10⁻⁹). The dashboard's
Sales Analytics page exposes the same table interactively.
**One-liner**: *"Cancellation is not uniform across categories — chi-
square confirms dependence. The high-cancel lines are where to look first."*

### Figure 7 — `07_top_states.png`
**Type**: horizontal bar chart, top 15 states by revenue
**What it shows**: Maharashtra leads, followed by Karnataka and
Tamil Nadu. The drop from #3 to #4 is noticeable.
**Insight**: marketing budget should be sized in proportion to this
ranking; the bubble map on the Geographic page reinforces this visually.
**One-liner**: *"Three states carry most of the revenue — Maharashtra,
Karnataka, Tamil Nadu."*

### Figure 8 — `08_correlations.png`
**Type**: side-by-side Pearson + Spearman heatmaps
**What it shows**: weak correlations across business variables; Pearson
and Spearman agree on direction.
**Insight**: amount is *not* explained by promotion-status or B2B-flag
linearly — category is the real driver (per ANOVA).
**One-liner**: *"Correlations are weak. Don't expect linear relationships
to explain much — the strong signal is categorical, not numeric."*

### Figure 9 — `09_heatmap_dow_month.png`
**Type**: weekday × month revenue heatmap
**What it shows**: weekends (especially Sunday) outperform; Thursday is
the consistently weakest weekday.
**Insight**: a candidate for weekday-targeted promotion windows.
**One-liner**: *"Sundays are the strongest day every month — Thursday
is consistently the weakest."*

### Figure 10 — `10_forecast_validation.png`
**Type**: time-series with three model predictions overlaid on the
30-day hold-out
**What it shows**: Naive MA tracks the mean; Linear Regression
under-fits badly; Holt-Winters tracks the actual line closely.
**Insight**: Holt-Winters wins on every metric (RMSE 104,787;
MAE 89,525; MAPE 12.4 %).
**One-liner**: *"On the 30-day hold-out, Holt-Winters had the lowest
MAPE at 12.4 % — that's the model the live dashboard uses."*

### Figure 11 — `11_forecast_final.png`
**Type**: full historical line + 30-day forecast + 95 % interval band
**What it shows**: continued softness projected; uncertainty band widens
near the end of the horizon.
**Insight**: confirms the dashboard's automated -10.7 % MoM warning.
**One-liner**: *"The forecast says continued pressure for the next month.
The orange band is the 95 % confidence interval — narrow now, widens
into the future, as you'd expect."*

---

## 7. Dashboard pages — feature by feature

Open `streamlit run app.py` and walk through them in this order.

### 7.1 Sidebar (shared across all pages)

| Filter | Type | What it does |
|---|---|---|
| Date range | Date picker | Crops the time window |
| Category | Multi-select | One or many categories |
| Order Status | Multi-select | Useful for "show only Shipped" or "only Cancelled" |
| Ship State | Multi-select | Drill into specific states |
| Fulfilment | Multi-select | Amazon vs Merchant vs others |
| Search city | Text input | Contains-match on ship_city (uppercase) |
| Include amount outliers | Checkbox | Opt-out filter — defaults to including |

**Demo line**: *"All filters are global. Change anything in the sidebar
and every chart on every page re-renders against that slice."*

### 7.2 Page 1 — Overview

**Six KPI cards** (top row + second row of three each):
1. Total Revenue (formatted Cr / L / K)
2. Total Orders (unique `order_id`)
3. Cancellation Rate %
4. Avg Order Value (shipped only)
5. Month-over-Month Growth %
6. Best-Selling Category

**Three charts**:
- Daily revenue with 7-day MA (line, Plotly hover-enabled)
- Top 10 categories by revenue (horizontal bar)
- Order-status mix (donut)

**Expandable "Snapshot summary"**: top state, top SKU, peak revenue day,
unique cities, unique SKUs.

**Why this design**: the page answers a stakeholder's first three
questions — *"How big is the business? Is it healthy? What's driving it?"*
— in under 10 seconds.

### 7.3 Page 2 — Sales Analytics

**Six charts + two tables**:
1. Granularity selector (Daily / Weekly / Monthly) → line chart of
   aggregated revenue.
2. **Weekday × Month** heatmap (revenue intensity).
3. **Pearson correlation** heatmap on the filtered data.
4. **Stacked bar** — revenue by category × fulfilment.
5. **Scatter** — amount vs qty, coloured by category (sampled to 8k
   points for snappy rendering).
6. **Top 20 orders by amount** — interactive table.
7. **Cancellation rate by category** — interactive table, sorted desc.

**Why this design**: this is the "analyst's playground" — every chart
answers a specific drill-down question. Examiner-asked "where are
cancellations worst?" → scroll to the bottom table.

### 7.4 Page 3 — Geographic

**Three visuals + one table**:
- **India bubble map** — Plotly geo, bubbles sized by revenue,
  positioned at state-capital coordinates (table of 36 state-capital
  lat/lon pairs lives in `src/charts.py`).
- Top 15 states by revenue (horizontal bar).
- Top 15 cities by order count (horizontal bar).
- State-level KPI table: orders / revenue / cancellation rate / AOV.

**Why this design**: the original task explicitly lists a Geographic
Map as optional but rewarded. This delivers it interactively with
genuine information.

### 7.5 Page 4 — Forecasting

**Components**:
- Horizon slider (7 / 14 / 30 days)
- Historical revenue + forecast line with 95 % confidence band
- Forecast values table (date / predicted / lower / upper)
- "Methodology" expander — repeats the model comparison

**Why this design**: the forecast is the bonus *Predictive Analytics*
deliverable. The slider lets stakeholders look at a shorter or fuller
horizon without re-running the model.

**Important nuance**: the forecast was trained on the full historical
window in notebook 03. The dashboard filters do **not** retrain the
model — they only affect the *historical* portion shown for context.
This is by design — retraining at request time would be slow and
unstable.

### 7.6 Page 5 — AI Insights

**Two layers**:
1. **Templated bullets** (always render):
   - "The filtered slice covers N orders generating ₹R."
   - Cancellation rate band (healthy / elevated / concerning).
   - Avg order value statement.
   - MoM growth direction.
   - Best-selling category + city coverage.
   - Strongest / weakest revenue weekday.
2. **Claude narrative** (only if API key is set):
   - Text input for an optional user question.
   - Button "Generate executive summary" → calls Claude Haiku 4.5
     with a system prompt enforcing three short paragraphs
     (trends / risks / recommendations), capped at 220 words.
   - Returns markdown with bold callouts for key figures.

**Why this design**: the bonus task says "AI Chatbot Integration". I
went further — the LLM call is grounded on *aggregated* filtered KPIs
so it cannot hallucinate counterfactuals and cannot leak raw data.
Templated fallback keeps the page useful with zero API spend.

---

## 8. ML Forecasting — methodology and results

### 8.1 Series construction

Daily revenue series from `revenue` column (shipped orders only), one
row per calendar day, 91 days long.

### 8.2 Stationarity check

Augmented Dickey-Fuller test: statistic = -5.835, p < 0.001 →
**reject** the unit-root null. The series is stationary, which means
an additive Holt-Winters formulation is appropriate (no need to
difference first).

### 8.3 Train / test split

Last 30 days are the hold-out. Train = 61 days, Test = 30 days.

### 8.4 Models compared

| Model | What it does | Strength | Weakness |
|---|---|---|---|
| **Naive 7-day MA forward-fill** | Predict the mean of the last 7 train days, for every test day | Cheap baseline, no risk of overfitting | Cannot capture trend or seasonality |
| **Linear Regression** on `[day_index, day_of_week_OH, month_OH]` | Linear combo of time features | Captures linear trend + weekday pattern + monthly shift | Cannot capture changing variance, fails when the test month is unseen (only one full June in training) |
| **Holt-Winters Exponential Smoothing** | Additive trend + additive seasonality with period 7 | Captures weekly cycle and trend simultaneously, smooths gracefully | Needs enough data per season — 91 days is on the lower bound |

### 8.5 Hold-out results

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| Naive 7-day MA | 111,844 | 91,393 | 13.6 % |
| Linear Regression | 375,720 | 362,992 | **48.3 %** ❌ |
| **Holt-Winters** | **104,787** | **89,525** | **12.4 %** ✓ |

Linear Regression's failure is the expected one — the `month` one-hot
coefficient for June was learned only from partial-June rows in the
training period, so extrapolation was bad. This is itself a useful
finding ("don't add features the model can't generalise on").

### 8.6 Final forecast

Holt-Winters refit on the full 91 days; project 30 days. Confidence
band is `point ± 1.96 · σ_residuals`. Saved to `data/forecast.csv`
with columns `[date, predicted_revenue, lower, upper, model]`.

### 8.7 What I would do with more data

- Add a holiday-aware SARIMAX or Prophet.
- Incorporate exogenous regressors: marketing spend, average price,
  promotion-status share.
- Run a back-test across multiple rolling origins, not just one.
- Add prediction intervals from model uncertainty, not just residual
  σ.

---

## 9. AI Insights — design and safety

### 9.1 Why two layers

The dashboard must be **fully functional offline** for grading (no
expectation that the examiner has my API key). Templated insights
satisfy this. The LLM layer is an enhancement, not a dependency.

### 9.2 What goes to Claude

Only **aggregated** values from the filtered slice:
- The six KPIs
- Row count, min/max date in the filter
- Top 5 categories by revenue (name → INR)
- Top 5 states by revenue (name → INR)

**Not sent**: individual order rows, order IDs, ship-city granularity,
SKU details, anything that could be construed as PII.

### 9.3 Prompt design

System prompt (paraphrased):
> "You are a senior business analyst. Given KPIs and stats from Amazon
> India sales, produce a three-paragraph executive summary (trends /
> risks / actions). Cite the numbers. Avoid causal claims. ≤ 220 words.
> Use markdown bold sparingly for key figures."

User prompt: the JSON of the context dict, optionally appended with the
user's typed question.

### 9.4 Failure modes handled

- `anthropic` package missing → page shows an "install hint" banner.
- API key missing → page shows a "configure in secrets" banner.
- API call exception → caught and shown in a red error box; templated
  bullets remain.

### 9.5 Why this is a real bonus (and not theatre)

It uses the LLM the way you should — as a *narrative* layer over
already-computed numbers, not as an arithmetic layer. The arithmetic is
in pandas. The LLM only describes.

---

## 10. Limitations & honesty

> Examiners love it when you raise limitations *before* they do.

| Limitation | Impact | Mitigation |
|---|---|---|
| No cost-of-goods column | Cannot compute gross profit. The "Total Profit" KPI from the rubric is substituted by Total Revenue. | Documented in the report; mentioned during the demo. |
| No `customer_id` | Classic RFM (recency-frequency-monetary) segmentation is impossible. | Order-level analysis used instead; called out as a future-work item. |
| Only 91 days of history | Time-series forecasting on a short window is fragile; seasonal models need ≥ 2 cycles to be confident. | Hold-out validation done honestly; forecast presented as directional, not commit-grade. |
| No exogenous variables (holidays, prices, marketing) | Forecast cannot react to known events. | Listed in future work; would address via SARIMAX/Prophet with regressors. |
| Single country, single channel | Findings don't generalise globally. | Stated explicitly. The pipeline does — swap the CSV and it still works. |
| LLM responses are stochastic | Two runs can produce slightly different narratives. | Templated bullets stay deterministic; LLM is only an *additional* narrative layer. |

**The honesty principle**: I never claim causation. Correlations show
*association*. Chi-square shows *dependence*. ANOVA shows *some* group
mean differs. Mann-Whitney shows *some* distribution difference. That is
what they show, and that is what I report.

---

## 11. The 8-slide deck and 15-minute speech

Mapping the recommended 8-slide structure to the dashboard.

### Slide 1 — Title + dataset source (Min 0:00–0:30)

> "Amazon Sales Intelligence & Forecasting — Final Project, Data
> Analysis and Visualization. Dataset: Amazon Sale Report from Kaggle,
> 128,975 real transactions from Amazon India between March 31 and
> June 29, 2022."

### Slide 2 — Objective (Min 0:30–2:00)

> "The business question: which products, regions, and channels are
> driving revenue, and where are we losing money to cancellations? The
> project answers this through four layers — programmatic cleaning,
> applied statistics, an interactive Streamlit dashboard, and two bonus
> features — a 30-day forecast and an LLM-powered insights engine."

### Slide 3 — Data cleaning summary (Min 2:00–5:00)

> "The raw file had 128,975 rows in 24 columns. After cleaning it's
> 128,936 rows in 30 columns. Three decisions are worth highlighting:
> first, I imputed missing amounts by category-and-size median rather
> than dropping them — dropping would have systematically removed
> cancelled orders, which would erase the 14 % cancellation insight.
> Second, I unified city names — BANGALORE / BENGALURU and similar
> pairs — because otherwise the geographic ranking double-counts.
> Third, I flagged outliers instead of removing them — high-value
> orders are real B2B activity, and removing them would erase that
> signal."

### Slide 4 — Applied statistics (Min 5:00–7:00)

> "I ran three formal tests at alpha 0.05. Chi-square shows that
> cancellation is not independent of category — different categories
> have meaningfully different cancellation rates. One-way ANOVA shows
> that order amount varies strongly by category. And Mann-Whitney U
> shows that B2B orders have a larger median than B2C — a small effect
> but statistically real. All three reject the null. Correlations
> between numeric variables are weak — the strong signal is
> categorical, not linear."

### Slide 5 — Main visualizations (Min 7:00–9:00)

[Show 2–3 of the strongest figures — figures 3, 5, 7 work well.]

> "Daily revenue with a moving average shows a clear softening trend
> after mid-May. Revenue by category shows three lines — Set, Kurta,
> Western Dress — carry the business; that's concentration risk. Top
> states show Maharashtra-Karnataka-Tamil Nadu carrying most of the
> geography."

### Slide 6 — Dashboard summary (Min 9:00–12:00)

[Switch to live demo, walk through all 5 pages following the
per-page demo flow in the cheatsheet.]

### Slide 7 — Three key insights (Min 12:00–13:30)

> "First: cancellation rate is 14.2 % — above a healthy 5–10 % band —
> and chi-square confirms it varies by category. That's the
> highest-impact lever. Second: revenue is declining 10.7 % MoM in the
> latest full month; the Holt-Winters forecast projects continued
> softness. Third: revenue concentrates geographically — three states
> carry most of it. Each insight is numerical, evidence-based, and
> visible on the dashboard."

### Slide 8 — Conclusion + limitations (Min 13:30–15:00)

> "The pipeline reproducibly delivers the cleaning, stats, dashboard,
> forecasting, and LLM-narrative layers required by the rubric, plus
> deploy-ready to Streamlit Cloud. Honestly: the dataset has no cost
> column so I report revenue, not profit; no customer ID so I can't do
> classic RFM; and 91 days is short for time-series forecasting. With
> more data I'd add holiday-aware SARIMAX and bring in exogenous
> variables. Happy to take questions."

---

## 12. Q&A — long-form answers

Each answer is structured as: **short verbal sentence** → **evidence to
point at** → **gracious sidestep if pushed**.

### Q: "Why did you pick this dataset?"

- **Short**: "It satisfies every rubric line — real-world, 128k rows,
  24 columns, geography, time, categories, and rich enough to require
  real cleaning."
- **Evidence**: open `original-task.txt` — the dataset section requires
  ≥ 5,000 rows, ≥ 10 columns, real data preferred. We're 25× over rows
  and 2.4× over columns.
- **If pushed**: "I considered the other CSVs in the Kaggle bundle —
  P&L statements and warehouse comparisons — but those are aggregated
  and don't expose enough rows for statistical work. The transactional
  Amazon Sale Report does."

### Q: "Biggest cleaning challenge?"

- **Short**: "Missing `amount` on 6 % of rows. Naively dropping them
  would have removed the signal I most wanted to study — cancellations
  often have missing amounts."
- **Evidence**: notebook 01 section 4 — `(category, size)` median
  imputation, three-step fallback.
- **If pushed**: "I picked median over mean because of the skew, and
  picked the `(category, size)` granularity because the same product
  in different sizes can have very different prices."

### Q: "Why did you flag outliers instead of removing them?"

- **Short**: "Because they're real high-value orders, not data errors.
  Mann-Whitney later confirmed B2B orders are statistically larger —
  that signal lives in the outliers."
- **Evidence**: the outlier flag column + the Mann-Whitney result.
- **If pushed**: "I expose a sidebar toggle so anyone who *does* want
  the outlier-free view can get it in one click."

### Q: "Why these specific charts?"

- **Short**: "Each chart answers a specific question. Bar = which is
  biggest. Line = trend over time. Heatmap = two-dimensional pattern.
  Scatter = relationship. Donut = mix share. Geo = concentration."
- **Evidence**: report section 7 maps each chart to a question.
- **If pushed**: "I avoid charts-for-the-sake-of-charts — for example
  I deliberately didn't include a pie chart with more than 5 slices."

### Q: "Why a dashboard instead of just static charts?"

- **Short**: "Static charts answer one question each. A dashboard with
  filters lets a stakeholder answer questions you didn't anticipate
  — like 'what does this look like in Maharashtra only?'"
- **Evidence**: live demo — change a sidebar filter, every chart
  re-renders.
- **If pushed**: "The static figures still exist in the PDF — they're
  the same data, but pre-aggregated to all 91 days."

### Q: "Correlation versus causation — what did you actually claim?"

- **Short**: "Nothing causal. The chi-square shows dependence between
  cancellation and category; it does not say category *causes*
  cancellation."
- **Evidence**: the report and cheatsheet both call this out
  explicitly.
- **If pushed**: "To get to causation you'd need an intervention — an
  A/B test on listing copy or returns policy — and observational data
  cannot deliver that."

### Q: "What are your three strongest insights?"

- **Short**: "(1) 14.2 % cancellation rate, category-dependent.
  (2) -10.7 % MoM revenue trend with continued forecast softness.
  (3) Geographic concentration in three states."
- **Evidence**: each is numerical, each is visible on the dashboard,
  each is in the report's findings section.
- **If pushed**: "All three are levers — the cancellation rate is
  *actionable* per category, the trend is *predictable* via the
  forecast, the geography is *targetable* for marketing spend."

### Q: "What are the limitations of your dataset?"

- **Short**: "No cost data, no customer ID, only 91 days, single
  country, single channel."
- **Evidence**: limitations section in this guide and the report.
- **If pushed**: "I prefer to flag these upfront — the worst defense
  failure is being caught hiding a limitation."

### Q: "If you had more time, what would you improve?"

- **Short**: "(1) Bring in cost data for real profit margin. (2) Add
  holiday-aware SARIMAX with exogenous variables. (3) Stream the LLM
  response for snappier UX. (4) Add click-through drill-downs on the
  bubble map. (5) Run a rolling-origin back-test on the forecast."
- **Evidence**: future-work section in the report.

### Q: "Why didn't you use Prophet?"

- **Short**: "Prophet has wheel-build issues on Python 3.14 on Windows,
  and statsmodels' Holt-Winters gave us comparable accuracy with a
  cleaner install. If I targeted Python 3.11 I'd add Prophet as a
  fourth model and compare on the same hold-out."

### Q: "How does the LLM know what to say?"

- **Short**: "I send it only the *aggregated* KPIs from the filtered
  slice — six KPIs, top categories, top states, the date span. The
  system prompt locks it to a three-paragraph executive summary and
  forbids causal claims. The arithmetic is in pandas; the LLM only
  describes."
- **Evidence**: `src/llm_insights.py`.

### Q: "Is the API key safe?"

- **Short**: "Yes. It's read from Streamlit's secrets manager, which
  is gitignored. On Streamlit Cloud it lives in the Secrets UI, not in
  the repo. No raw data leaves the app — only aggregates."
- **Evidence**: `.gitignore` includes `.streamlit/secrets.toml`; only
  `.streamlit/secrets.toml.example` is committed.

### Q: "Why did you build a multi-page app instead of one page?"

- **Short**: "Cognitive load. Five smaller pages, each answering a
  coherent set of questions, is easier to navigate than one wall of
  scrolling charts."
- **Evidence**: page-by-page breakdown above.

### Q: "Is your forecast confidence band statistically rigorous?"

- **Short**: "It's a 1.96σ band on residuals — a normal-approximation
  prediction interval. With more data I'd use the model's own
  predictive variance from `get_prediction()`, which accounts for
  parameter uncertainty too."

### Q: "Can your pipeline handle a different dataset?"

- **Short**: "Mostly yes. The cleaning script is dataset-specific (the
  category-size median imputation only makes sense for products with
  size attributes). The EDA and forecasting steps generalise well. The
  dashboard pages would need column-name updates, since they reference
  `category`, `ship_state`, etc."

### Q: "What's the most interesting thing you found?"

- **Short**: "That the strong signals here are *categorical*, not
  numeric. Order amount doesn't correlate linearly with anything
  measurable — but ANOVA shows category dominates. That tells me
  Amazon's product taxonomy is the right axis to optimise on."

---

## Final pre-flight checklist

Before you walk into the room:

- [ ] Laptop charged + power cable.
- [ ] `streamlit run app.py` works in a clean terminal.
- [ ] PDF (`report/report.pdf`) opens.
- [ ] The dashboard's date range covers the full 91 days (reset filters).
- [ ] You have this guide and the cheatsheet open on a phone or tablet.
- [ ] You've practiced the opening line out loud once.
- [ ] You know the three insight numbers cold:
      **14.2 %**, **-10.7 %**, **₹71.79 M / Maharashtra-Karnataka-Tamil Nadu**.

Good luck.
