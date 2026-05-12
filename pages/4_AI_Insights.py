"""AI Insights page — templated bullets + optional Claude-powered narrative."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_data, sidebar_filters, inject_css
from src.kpis import compute_kpis
from src.llm_insights import template_insights, llm_narrative, llm_available


st.set_page_config(page_title="AI Insights", page_icon="🤖", layout="wide")
inject_css()

st.title("AI-Generated Insights")
st.caption(
    "Automatic interpretation of the filtered data. "
    "Templated insights always available; LLM narrative when an API key is configured."
)

df = load_data()
filtered = sidebar_filters(df)

if filtered.empty:
    st.warning("No rows match the current filters.")
    st.stop()

kpis = compute_kpis(filtered)

st.markdown('<div class="section-header">Automatic insights</div>',
            unsafe_allow_html=True)
for bullet in template_insights(kpis, filtered):
    st.markdown(f"- {bullet}")

st.markdown('<div class="section-header">Ask the AI analyst</div>',
            unsafe_allow_html=True)

api_key = st.secrets.get("ANTHROPIC_API_KEY") if hasattr(st, "secrets") else None
try:
    if api_key is None and hasattr(st, "secrets"):
        api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
except Exception:
    api_key = None

if not llm_available():
    st.info(
        "The `anthropic` Python package is not installed in this environment. "
        "Run `pip install anthropic` to enable the LLM narrative feature."
    )
elif not api_key:
    st.info(
        "No `ANTHROPIC_API_KEY` found in Streamlit secrets. "
        "Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` "
        "and add your key to enable AI narrative generation."
    )
else:
    question = st.text_input(
        "Optional: ask a specific question (e.g. 'Which category should we double down on?')",
        value="",
    )
    if st.button("Generate executive summary", type="primary"):
        with st.spinner("Claude is analysing the filtered slice…"):
            try:
                context = {
                    **kpis,
                    "row_count": int(len(filtered)),
                    "date_min": str(filtered["date"].min().date()),
                    "date_max": str(filtered["date"].max().date()),
                    "top_5_categories_by_revenue": (
                        filtered.groupby("category")["revenue"].sum()
                        .sort_values(ascending=False).head(5).round(0).to_dict()
                    ),
                    "top_5_states_by_revenue": (
                        filtered.groupby("ship_state")["revenue"].sum()
                        .sort_values(ascending=False).head(5).round(0).to_dict()
                    ),
                }
                narrative = llm_narrative(context, api_key, question=question or None)
                st.markdown(narrative)
            except Exception as exc:
                st.error(f"LLM call failed: {exc}")

with st.expander("How does this work?"):
    st.markdown(
        """
        - **Templated insights** are generated from KPI thresholds in pure Python —
          they require no network access and always render.
        - **LLM narrative** sends the filtered KPIs (no raw rows, no PII) to
          Anthropic's Claude API for a 3-paragraph executive summary. The model
          identifier and a system prompt enforcing analytical caution are
          defined in `src/llm_insights.py`.
        - API key is read from `st.secrets["ANTHROPIC_API_KEY"]` — never
          committed to git.
        """
    )
