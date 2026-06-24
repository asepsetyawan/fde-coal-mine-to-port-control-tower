"""Natural-language ops assistant over the control tower.

Executives ask questions; they don't read dashboards. This assistant
exposes the control tower's analytics as tools. With an OPENAI_API_KEY it
uses function-calling; without one it falls back to a deterministic
keyword router so the demo always works (incl. Bahasa Indonesia).
"""
from __future__ import annotations

import json
from typing import Callable

from ..config import OPENAI_API_KEY, OPENAI_MODEL


# --- Tools (the assistant's "hands" on the data) ---------------------------
def tool_demurrage_exposure() -> dict:
    from .demurrage import predict_vessels

    df = predict_vessels()
    if df.empty:
        return {"error": "no vessel data"}
    top = df.head(3)[["vessel_id", "pred_demurrage_days", "risk_usd", "risk_level"]]
    return {
        "total_exposure_usd": float(df["risk_usd"].sum()),
        "high_risk_vessels": int((df["risk_level"] == "HIGH").sum()),
        "top_vessels": top.to_dict(orient="records"),
    }


def tool_blend_recommendation(target_tons: float = 65_000.0) -> dict:
    from .blending import optimize_blend

    r = optimize_blend(target_tons=target_tons)
    return {
        "feasible": r.feasible,
        "mix_tons": r.mix,
        "blended_quality": r.blended,
        "cost_per_ton_usd": r.cost_per_ton,
        "quality_giveaway_cv": r.quality_giveaway_cv,
        "giveaway_value_usd": r.giveaway_value_usd,
        "message": r.message,
    }


def tool_fleet_health() -> dict:
    from .predictive_maintenance import score_fleet

    df = score_fleet()
    if df.empty:
        return {"error": "no fleet data"}
    crit = df[df["health"] == "CRITICAL"]
    return {
        "critical_trucks": crit["truck_id"].tolist(),
        "watch_trucks": df[df["health"] == "WATCH"]["truck_id"].tolist(),
        "top_risk": df.head(5)[["truck_id", "failure_prob_48h", "health"]].to_dict("records"),
    }


def tool_reconciliation_anomalies() -> dict:
    from .reconciliation_anomaly import detect

    df = detect()
    if df.empty:
        return {"error": "no reconciliation data"}
    flagged = df[df["anomaly"] == 1]
    return {
        "n_flagged_days": int(len(flagged)),
        "flagged_dates": [str(d)[:10] for d in flagged["date"].tolist()],
        "avg_pit_to_vessel_loss_pct": float(df["pit_to_vessel_loss_pct"].mean()),
    }


def tool_production_summary() -> dict:
    from ..pipeline.reconciliation import summary

    return summary()


TOOLS: dict[str, Callable[..., dict]] = {
    "demurrage_exposure": tool_demurrage_exposure,
    "blend_recommendation": tool_blend_recommendation,
    "fleet_health": tool_fleet_health,
    "reconciliation_anomalies": tool_reconciliation_anomalies,
    "production_summary": tool_production_summary,
}

OPENAI_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "demurrage_exposure",
            "description": "Total demurrage USD exposure and the highest-risk vessels.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "blend_recommendation",
            "description": "Optimal coal blend to meet contract spec at minimum cost.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_tons": {"type": "number", "description": "Cargo size in tons"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fleet_health",
            "description": "Haul trucks predicted to fail within 48h (critical/watch).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reconciliation_anomalies",
            "description": "Days with abnormal tonnage loss (possible drift/theft).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "production_summary",
            "description": "Pit-to-vessel tonnage and average loss summary.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM_PROMPT = (
    "You are the operations assistant for a coal mine-to-port control tower. "
    "Answer concisely with concrete numbers. Use the provided tools to fetch "
    "live data; never invent figures. Reply in the user's language "
    "(English or Bahasa Indonesia)."
)


def ask(question: str) -> dict:
    """Return {'answer': str, 'data': dict, 'mode': 'llm'|'rule'}."""
    if OPENAI_API_KEY:
        try:
            return _ask_llm(question)
        except Exception as exc:  # noqa: BLE001 - fall back gracefully in a demo
            return _ask_rule(question, note=f"(LLM error, used fallback: {exc})")
    return _ask_rule(question)


def _ask_llm(question: str) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    first = client.chat.completions.create(
        model=OPENAI_MODEL, messages=messages, tools=OPENAI_TOOL_SCHEMA, tool_choice="auto"
    )
    msg = first.choices[0].message
    collected: dict = {}
    if msg.tool_calls:
        messages.append(msg.model_dump())
        for call in msg.tool_calls:
            fn = TOOLS.get(call.function.name)
            args = json.loads(call.function.arguments or "{}")
            result = fn(**args) if fn else {"error": "unknown tool"}
            collected[call.function.name] = result
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, default=str),
            })
        final = client.chat.completions.create(model=OPENAI_MODEL, messages=messages)
        return {"answer": final.choices[0].message.content, "data": collected, "mode": "llm"}
    return {"answer": msg.content, "data": {}, "mode": "llm"}


def _ask_rule(question: str, note: str = "") -> dict:
    """Keyword router used when no LLM key is configured."""
    q = question.lower()
    data: dict = {}

    def has(*words: str) -> bool:
        return any(w in q for w in words)

    if has("demurrage", "vessel", "kapal", "laycan", "denda"):
        data = tool_demurrage_exposure()
        answer = (
            f"Total demurrage exposure ~${data.get('total_exposure_usd', 0):,.0f} "
            f"across scheduled vessels; {data.get('high_risk_vessels', 0)} high-risk. "
            f"Top: {', '.join(v['vessel_id'] for v in data.get('top_vessels', []))}."
        )
    elif has("blend", "blending", "quality", "kualitas", "spec", "cv", "campuran"):
        data = tool_blend_recommendation()
        if data.get("feasible"):
            answer = (
                f"Optimal blend at ${data['cost_per_ton_usd']}/t meeting spec. "
                f"Quality giveaway {data['quality_giveaway_cv']} CV "
                f"(~${data['giveaway_value_usd']:,.0f})."
            )
        else:
            answer = data.get("message", "Blend infeasible.")
    elif has("truck", "fleet", "maintenance", "truk", "armada", "rusak", "breakdown"):
        data = tool_fleet_health()
        answer = (
            f"{len(data.get('critical_trucks', []))} trucks CRITICAL "
            f"({', '.join(data.get('critical_trucks', [])[:5]) or 'none'}), "
            f"{len(data.get('watch_trucks', []))} on WATCH."
        )
    elif has("anomaly", "reconcil", "loss", "theft", "selisih", "anomali", "kehilangan"):
        data = tool_reconciliation_anomalies()
        answer = (
            f"{data.get('n_flagged_days', 0)} day(s) flagged for abnormal loss; "
            f"avg pit-to-vessel loss {data.get('avg_pit_to_vessel_loss_pct', 0):.2f}%. "
            f"Dates: {', '.join(data.get('flagged_dates', [])[:5]) or 'none'}."
        )
    elif has("production", "tons", "produksi", "tonase", "summary", "ringkasan"):
        data = tool_production_summary()
        answer = (
            f"Pit {data.get('total_pit_tons', 0):,.0f} t -> vessel "
            f"{data.get('total_vessel_tons', 0):,.0f} t over {data.get('days', 0)} days; "
            f"avg loss {data.get('avg_pit_to_vessel_loss_pct', 0):.2f}%."
        )
    else:
        answer = (
            "I can answer about: demurrage exposure, blending/quality, fleet health, "
            "reconciliation anomalies, or production summary. "
            "(Tanya soal demurrage, blending, armada truk, anomali, atau produksi.)"
        )
    if note:
        answer = f"{answer}  {note}"
    return {"answer": answer, "data": data, "mode": "rule"}
