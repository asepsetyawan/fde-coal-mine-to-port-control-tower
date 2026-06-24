"""End-to-end smoke test: seed -> train -> run every analytic + assistant.

Run with:  python -m control_tower.smoke
Exits non-zero on any failure so it can gate CI.
"""
from __future__ import annotations

import sys


def main() -> int:
    from .pipeline import seed
    from .ai import train as train_all
    from .ai import blending, demurrage, predictive_maintenance, reconciliation_anomaly
    from .ai.assistant import ask
    from .pipeline.reconciliation import summary

    print("== Seeding ==")
    seed.main(days=20)

    print("\n== Training ==")
    train_all.main()

    print("\n== Reconciliation summary ==")
    print(summary())

    print("\n== Demurrage (top vessels) ==")
    print(demurrage.predict_vessels().head().to_string(index=False))

    print("\n== Blending ==")
    r = blending.optimize_blend(65_000)
    print(f"feasible={r.feasible} cost/t=${r.cost_per_ton} "
          f"giveaway=${r.giveaway_value_usd:,.0f} mix={r.mix}")

    print("\n== Predictive maintenance (top trucks) ==")
    print(predictive_maintenance.score_fleet().head().to_string(index=False))

    print("\n== Anomalies ==")
    an = reconciliation_anomaly.detect()
    print(f"flagged days: {int(an['anomaly'].sum())}")

    print("\n== Assistant (rule/LLM) ==")
    for q in ["What is our demurrage exposure?",
              "Recommend a blend for 70000 tons",
              "Truk mana yang berisiko rusak?"]:
        resp = ask(q)
        print(f"[{resp['mode']}] Q: {q}\n      A: {resp['answer']}")

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
