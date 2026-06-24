"""Reconciliation anomaly detection.

Tonnage loss between stages is always non-zero (moisture, spillage,
measurement variance). The question is which days show *abnormal* loss
that may indicate calibration drift or theft. IsolationForest learns the
normal loss distribution and flags outliers for investigation.
"""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..config import MODELS_DIR
from ..pipeline.reconciliation import reconciliation_frame

MODEL_PATH = MODELS_DIR / "recon_anomaly.joblib"

LOSS_COLS = [
    "loss_PIT_to_ROM_pct",
    "loss_ROM_to_CRUSHER_pct",
    "loss_CRUSHER_to_PORT_pct",
    "loss_PORT_to_VESSEL_pct",
]


def train(contamination: float = 0.06) -> dict:
    rf = reconciliation_frame()
    if rf.empty:
        raise RuntimeError("No reconciliation data. Run the seed pipeline first.")
    X = rf[LOSS_COLS].fillna(0.0)
    model = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=0
    )
    model.fit(X)
    joblib.dump({"model": model, "features": LOSS_COLS}, MODEL_PATH)
    flagged = detect()
    return {"n_days": int(len(rf)), "n_flagged": int(flagged["anomaly"].sum())}


def _load():
    if not MODEL_PATH.exists():
        raise RuntimeError("Anomaly model not trained. Run: python -m control_tower.ai.train")
    return joblib.load(MODEL_PATH)


def detect() -> pd.DataFrame:
    bundle = _load()
    model, feats = bundle["model"], bundle["features"]
    rf = reconciliation_frame()
    if rf.empty:
        return rf
    X = rf[feats].fillna(0.0)
    rf = rf.copy()
    rf["anomaly_score"] = model.decision_function(X).round(4)
    rf["anomaly"] = (model.predict(X) == -1).astype(int)
    return rf.sort_values("anomaly_score").reset_index(drop=True)
