## Submition Files and access

**The Project is deployed and is accesible through link : https://sanz-minb-da-project.streamlit.app/**

**Used dataset: https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data**

Main Files:

**notebooks/** - Notebooks with data cleaning, eda and forecasting, python scripts that do exactly the same things. 

**report/figures** - All Dashboard files 

Other:

**.streamlit/** - configuration files, enviroment variables

**assets/styles.css** - css styles

**data/** - input dataset,cleaned dataset and forecasting on revenue

**pages/** - streamlit pages

**report/report.html** - report summary

**src/** - Python modules shared by app.py and every page in pages/

**app.py** The Streamlit entrypoint. Renders the Overview page

---

**Final project for the Data Analysis & Visualization course.**

End-to-end data analytics solution on the **Amazon India Sale Report**
dataset (128,975 transactions, April–June 2022).
---

## What's in the box

- **3 Jupyter notebooks** — programmatic cleaning, EDA + statistics,
  ML forecasting.
- **Streamlit dashboard** with 5 pages — Overview, Sales Analytics,
  Geographic, Forecasting, AI Insights.
- **AI Insights** with deterministic fallback.
- **30-day revenue forecast** (Holt-Winters wins MAPE 12.4%).
- **PDF report** with 11 figures, three hypothesis tests, full methodology.

For a per-file index, see **[project.md](project.md)**.
For the design rationale and corrections to Gemini's task summary, see
**[plan.txt](plan.txt)**.

---

## Run it locally

```bash
# 1. Install deps (Python 3.11+ recommended)
pip install -r requirements.txt

# 2. Build the cleaned CSV, figures, and forecast
python notebooks/01_cleaning.py
python notebooks/02_eda.py
python notebooks/03_forecasting.py

# 3. (Optional) build the .ipynb notebooks
python notebooks/build_notebooks.py

# 4. Launch the dashboard
streamlit run app.py
```

The dashboard opens at <http://localhost:8501>.

---

## Enable AI Insights (optional)

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml and paste your ANTHROPIC_API_KEY
```

Without a key, the AI Insights page still renders deterministic templated
bullets — no crashes.

---

## Project highlights

| KPI (full dataset) | Value |
|---|---|
| Total revenue | ₹7.18 Cr |
| Unique orders | 120,350 |
| Cancellation rate | 14.21 % |
| Avg order value | ₹649 |
| Top category | Set |
| Top state | Maharashtra |
| Best forecast model | Holt-Winters (MAPE 12.4 %) |

---

## License & credits

Coursework project. Dataset: [Amazon Sale Report on Kaggle](https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data) (uploader: Anandshaw123).
