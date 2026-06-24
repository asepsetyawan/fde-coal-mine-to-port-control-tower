"""Tonnage reconciliation pit -> ROM -> CRUSHER -> PORT -> VESSEL.

Pure data engineering (no AI): pivots stage tonnage by date and computes
stage-to-stage loss %. The AI layer (anomaly detection) sits on top of
this output to separate normal variance from abnormal leakage.
"""
from __future__ import annotations

import pandas as pd

from ..config import VALUE_CHAIN_STAGES
from ..db import read_table


def reconciliation_frame() -> pd.DataFrame:
    df = read_table("stage_tonnage")
    if df.empty:
        return df
    wide = df.pivot_table(index="date", columns="stage", values="tons", aggfunc="sum")
    wide = wide.reindex(columns=VALUE_CHAIN_STAGES)

    out = wide.copy()
    # Stage-to-stage loss relative to the previous stage.
    for prev, cur in zip(VALUE_CHAIN_STAGES[:-1], VALUE_CHAIN_STAGES[1:]):
        out[f"loss_{prev}_to_{cur}_pct"] = (
            (wide[prev] - wide[cur]) / wide[prev] * 100
        ).round(3)

    # End-to-end retention and total loss.
    out["pit_to_vessel_loss_pct"] = (
        (wide["PIT"] - wide["VESSEL"]) / wide["PIT"] * 100
    ).round(3)
    return out.reset_index()


def summary() -> dict:
    rf = reconciliation_frame()
    if rf.empty:
        return {}
    return {
        "days": int(len(rf)),
        "total_pit_tons": float(rf["PIT"].sum()),
        "total_vessel_tons": float(rf["VESSEL"].sum()),
        "avg_pit_to_vessel_loss_pct": float(rf["pit_to_vessel_loss_pct"].mean()),
        "worst_day_loss_pct": float(rf["pit_to_vessel_loss_pct"].max()),
    }
