"""Coal blending optimization (the biggest margin lever).

Given current stockpile quality/availability and a buyer's contract spec,
choose how many tons to draw from each source to build the cargo at minimum
cost while meeting every spec constraint. Over-delivering quality ("quality
giveaway") is wasted margin, so minimizing cost naturally pushes the blend
to *just meet* spec.

Pure operations research (linear programming via PuLP) - not ML - which is
exactly the right tool for hard constraints.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pulp

from ..config import (
    DEFAULT_CONTRACT,
    REFERENCE_CV,
    REFERENCE_PRICE_USD_PER_TON,
    STOCKPILES,
    ContractSpec,
)
from ..db import read_table


def latest_stockpile_quality() -> pd.DataFrame:
    """Most recent assay per source stockpile (falls back to config profile)."""
    df = read_table("lab_quality")
    if df.empty:
        rows = [
            dict(stockpile_id=k, cv=v["cv"], ash_pct=v["ash"],
                 sulphur_pct=v["sulphur"], tm_pct=v["tm"],
                 tons_available=50_000.0)
            for k, v in STOCKPILES.items()
        ]
        return pd.DataFrame(rows)
    df = df.sort_values("ts").groupby("stockpile_id", as_index=False).last()
    return df


@dataclass
class BlendResult:
    feasible: bool
    target_tons: float
    mix: dict[str, float]                 # stockpile_id -> tons
    blended: dict[str, float]             # resulting cv/ash/sulphur/tm
    total_cost_usd: float
    cost_per_ton: float
    quality_giveaway_cv: float            # CV delivered above the minimum
    giveaway_value_usd: float             # $ value of that giveaway
    message: str


def optimize_blend(target_tons: float = 65_000.0,
                   spec: ContractSpec = DEFAULT_CONTRACT) -> BlendResult:
    q = latest_stockpile_quality()
    sources = q["stockpile_id"].tolist()
    cv = dict(zip(q["stockpile_id"], q["cv"]))
    ash = dict(zip(q["stockpile_id"], q["ash_pct"]))
    sul = dict(zip(q["stockpile_id"], q["sulphur_pct"]))
    tm = dict(zip(q["stockpile_id"], q["tm_pct"]))
    avail = dict(zip(q["stockpile_id"], q["tons_available"]))
    cost = {s: STOCKPILES.get(s, {}).get("cost_per_ton", 35.0) for s in sources}

    prob = pulp.LpProblem("coal_blend", pulp.LpMinimize)
    x = {s: pulp.LpVariable(f"x_{s}", lowBound=0, upBound=avail[s]) for s in sources}

    # Objective: minimize procurement cost (penalizes over-using premium coal).
    prob += pulp.lpSum(cost[s] * x[s] for s in sources)

    # Mass balance.
    prob += pulp.lpSum(x[s] for s in sources) == target_tons
    # Quality constraints (linear in tons).
    prob += pulp.lpSum(cv[s] * x[s] for s in sources) >= spec.min_cv * target_tons
    prob += pulp.lpSum(ash[s] * x[s] for s in sources) <= spec.max_ash * target_tons
    prob += pulp.lpSum(sul[s] * x[s] for s in sources) <= spec.max_sulphur * target_tons
    prob += pulp.lpSum(tm[s] * x[s] for s in sources) <= spec.max_tm * target_tons

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        return BlendResult(
            feasible=False, target_tons=target_tons, mix={}, blended={},
            total_cost_usd=0.0, cost_per_ton=0.0, quality_giveaway_cv=0.0,
            giveaway_value_usd=0.0,
            message=("Infeasible: available stockpiles cannot meet the contract "
                     "spec at this tonnage. Source higher-grade coal or relax spec."),
        )

    mix = {s: round(x[s].value(), 1) for s in sources if x[s].value() and x[s].value() > 1}
    tot = sum(mix.values()) or target_tons
    blended = {
        "cv": round(sum(cv[s] * t for s, t in mix.items()) / tot, 1),
        "ash_pct": round(sum(ash[s] * t for s, t in mix.items()) / tot, 2),
        "sulphur_pct": round(sum(sul[s] * t for s, t in mix.items()) / tot, 3),
        "tm_pct": round(sum(tm[s] * t for s, t in mix.items()) / tot, 2),
    }
    total_cost = sum(cost[s] * t for s, t in mix.items())
    giveaway_cv = max(0.0, blended["cv"] - spec.min_cv)
    # Value the giveaway at the reference $/ton per CV point.
    giveaway_value = giveaway_cv / REFERENCE_CV * REFERENCE_PRICE_USD_PER_TON * tot

    return BlendResult(
        feasible=True,
        target_tons=target_tons,
        mix=mix,
        blended=blended,
        total_cost_usd=round(total_cost, 0),
        cost_per_ton=round(total_cost / tot, 2),
        quality_giveaway_cv=round(giveaway_cv, 1),
        giveaway_value_usd=round(giveaway_value, 0),
        message="Optimal blend found (just meets spec at minimum cost).",
    )
