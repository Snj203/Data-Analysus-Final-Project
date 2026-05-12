"""LLM-powered insights (Anthropic Claude) with a deterministic fallback.

Design:
  * `template_insights()` always works — pure Python, no network.
  * `llm_narrative()` is only invoked if the user supplies an API key.
  * Import of `anthropic` is lazy so the dashboard does not crash if the
    package is missing.
"""

from __future__ import annotations

import json
from typing import Any


def template_insights(kpis: dict, df=None) -> list[str]:
    """Generate ~6 deterministic business insights from KPIs and (optional)
    filtered dataframe. Used as a baseline and as a fallback when no LLM key
    is configured."""
    out: list[str] = []

    rev = kpis.get("total_revenue", 0)
    orders = kpis.get("total_orders", 0)
    out.append(
        f"The filtered slice covers {orders:,} orders generating ₹{rev:,.0f} in revenue."
    )

    cancel = kpis.get("cancellation_rate", 0)
    if cancel > 15:
        out.append(
            f"Cancellation rate is **{cancel:.1f}%** — well above a healthy 5–10% band; "
            "investigate inventory accuracy and customer-fit issues."
        )
    elif cancel > 8:
        out.append(
            f"Cancellation rate at **{cancel:.1f}%** is elevated; worth a deeper look "
            "into top-cancelled categories."
        )
    else:
        out.append(f"Cancellation rate is healthy at **{cancel:.1f}%**.")

    aov = kpis.get("avg_order_value", 0)
    out.append(f"Average order value sits at **₹{aov:,.0f}** per shipped order.")

    mom = kpis.get("mom_growth")
    if mom is None:
        out.append("Insufficient history to compute month-over-month growth.")
    elif mom > 5:
        out.append(f"Revenue is trending **up {mom:.1f}% MoM** in the latest month.")
    elif mom < -5:
        out.append(f"Revenue is trending **down {abs(mom):.1f}% MoM** — flag for review.")
    else:
        out.append(f"Revenue is roughly flat ({mom:+.1f}% MoM).")

    out.append(
        f"Best-selling category is **{kpis.get('best_category', '—')}**, "
        f"shipping to **{kpis.get('unique_cities', 0):,}** distinct cities."
    )

    if df is not None and not df.empty and "order_weekday" in df.columns:
        dow = (df.groupby("order_weekday")["revenue"].sum()
               .reindex(["Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday"]))
        if dow.notna().any():
            best_dow = str(dow.idxmax())
            worst_dow = str(dow.idxmin())
            out.append(
                f"Strongest revenue weekday is **{best_dow}**; weakest is **{worst_dow}**."
            )

    return out


def llm_available() -> bool:
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def llm_narrative(context: dict[str, Any], api_key: str,
                  model: str = "claude-haiku-4-5-20251001",
                  question: str | None = None) -> str:
    """Call Claude for a richer narrative. Returns the model's text response.

    Raises on API errors so the page can surface a helpful message.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    system_msg = (
        "You are a senior business analyst working with Amazon India sales data. "
        "Given the KPIs and summary statistics provided, write a concise executive "
        "summary in three short paragraphs:\n"
        "1) Headline trends — what the numbers show.\n"
        "2) Risks or anomalies — what stands out as concerning.\n"
        "3) Recommended next actions — specific, data-driven suggestions.\n"
        "Cite the numeric values you reference. Avoid causal claims unless the "
        "data clearly supports them. Keep the total under 220 words. "
        "Use markdown bold sparingly for key figures."
    )
    payload = json.dumps(context, default=str, indent=2)
    user_msg = f"Sales KPIs and stats:\n```json\n{payload}\n```"
    if question:
        user_msg += f"\n\nUser question: {question}"

    response = client.messages.create(
        model=model,
        max_tokens=600,
        system=system_msg,
        messages=[{"role": "user", "content": user_msg}],
    )
    parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()
