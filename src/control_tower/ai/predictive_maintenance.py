"""Predictive maintenance for the haul fleet.

Equipment failure is probabilistic and hidden in sensor patterns (engine
temperature, vibration, oil pressure). We train a classifier to predict
whether a truck will fail within 48h, then score the latest reading for
each truck so maintenance can shift unplanned -> planned downtime.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ..config import MODELS_DIR
from ..db import read_table

MODEL_PATH = MODELS_DIR / "predictive_maintenance.joblib"

FEATURES = [
    "engine_temp_c",
    "oil_pressure_kpa",
    "vibration_mm_s",
    "fuel_rate_lph",
    "cycle_time_min",
    "payload_t",
]


def _running(df: pd.DataFrame) -> pd.DataFrame:
    # Only running samples carry a meaningful sensor signature.
    return df[df["status"] == "RUNNING"].copy()


@dataclass
class PdMMetrics:
    roc_auc: float
    pr_auc: float
    positive_rate: float
    n_train: int
    n_test: int


def train() -> PdMMetrics:
    df = _running(read_table("fleet_telemetry"))
    if df.empty:
        raise RuntimeError("No fleet telemetry. Run the seed pipeline first.")

    X = df[FEATURES].astype(float)
    y = df["failed_within_48h"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced", random_state=0, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    metrics = PdMMetrics(
        roc_auc=float(roc_auc_score(y_te, proba)),
        pr_auc=float(average_precision_score(y_te, proba)),
        positive_rate=float(y.mean()),
        n_train=len(X_tr),
        n_test=len(X_te),
    )
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    return metrics


def _load():
    if not MODEL_PATH.exists():
        raise RuntimeError("PdM model not trained. Run: python -m control_tower.ai.train")
    return joblib.load(MODEL_PATH)


def score_fleet() -> pd.DataFrame:
    """Failure probability on each truck's most recent running reading."""
    bundle = _load()
    model, feats = bundle["model"], bundle["features"]
    df = _running(read_table("fleet_telemetry"))
    if df.empty:
        return df
    latest = df.sort_values("ts").groupby("truck_id", as_index=False).last()
    latest["failure_prob_48h"] = model.predict_proba(latest[feats].astype(float))[:, 1]
    latest["health"] = pd.cut(
        latest["failure_prob_48h"],
        bins=[-0.01, 0.2, 0.5, 1.01],
        labels=["OK", "WATCH", "CRITICAL"],
    ).astype(str)
    cols = ["truck_id", "ts", *feats, "failure_prob_48h", "health"]
    return latest[cols].sort_values("failure_prob_48h", ascending=False).reset_index(drop=True)
