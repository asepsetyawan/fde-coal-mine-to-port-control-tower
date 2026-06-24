"""Demurrage risk prediction (the cleanest $ story in the project).

Whether a vessel waits depends on uncertain, interacting drivers (stockpile
build rate, rain, port congestion, quality margin). We learn the expected
demurrage *days* per vessel and translate it to USD exposure.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict

from ..config import DEMURRAGE_USD_PER_DAY, MODELS_DIR
from ..db import read_table
from ..simulators.vessels import FEATURE_COLS

MODEL_PATH = MODELS_DIR / "demurrage.joblib"


def _make_model() -> RandomForestRegressor:
    # RandomForest generalises better than boosting on this small,
    # zero-inflated target (most parcels load on time).
    return RandomForestRegressor(n_estimators=300, max_depth=6, random_state=0, n_jobs=-1)


@dataclass
class DemurrageMetrics:
    mae_days: float
    r2: float
    n_train: int
    n_test: int


def train() -> DemurrageMetrics:
    df = read_table("vessel")
    if df.empty:
        raise RuntimeError("No vessel data. Run the seed pipeline first.")

    X = df[FEATURE_COLS].astype(float)
    y = df["demurrage_days"].astype(float)

    # Honest out-of-fold metrics (shuffled K-fold) instead of one tiny
    # train/test split. MAE (in days) is the headline metric since the
    # target is zero-heavy and R^2 is unstable on small samples.
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.clip(cross_val_predict(_make_model(), X, y, cv=kf), 0, None)
    ss_res = float(np.sum((y - oof) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = float(np.mean(np.abs(y - oof)))

    # Fit the final model on all data for serving.
    model = _make_model().fit(X, y)
    joblib.dump({"model": model, "features": FEATURE_COLS}, MODEL_PATH)

    n_test = len(y) // 5
    return DemurrageMetrics(mae_days=mae, r2=float(r2), n_train=len(y) - n_test, n_test=n_test)


def _load():
    if not MODEL_PATH.exists():
        raise RuntimeError("Demurrage model not trained. Run: python -m control_tower.ai.train")
    return joblib.load(MODEL_PATH)


def predict_vessels() -> pd.DataFrame:
    """Score every scheduled vessel with predicted demurrage days + USD risk."""
    bundle = _load()
    model, feats = bundle["model"], bundle["features"]
    df = read_table("vessel")
    if df.empty:
        return df

    pred_days = np.clip(model.predict(df[feats].astype(float)), 0, None)
    out = df[["vessel_id", "laycan_start", "laycan_end", "cargo_tons"]].copy()
    out["pred_demurrage_days"] = pred_days.round(2)
    out["risk_usd"] = (pred_days * DEMURRAGE_USD_PER_DAY).round(0)
    out["risk_level"] = pd.cut(
        out["pred_demurrage_days"],
        bins=[-0.01, 0.25, 1.0, np.inf],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)
    return out.sort_values("risk_usd", ascending=False).reset_index(drop=True)


def feature_importances() -> pd.DataFrame:
    bundle = _load()
    model, feats = bundle["model"], bundle["features"]
    return (
        pd.DataFrame({"feature": feats, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def summary() -> dict:
    """Expected-vs-delivered reconciliation for the demurrage programme.

    Makes the demurrage story auditable: how much cargo is scheduled vs how
    much coal the port actually delivered, the resulting berth utilisation,
    and the risk distribution behind the headline USD exposure.
    """
    from ..pipeline.reconciliation import summary as recon_summary

    scored = predict_vessels()
    if scored.empty:
        return {}

    n = int(len(scored))
    scheduled_cargo = float(scored["cargo_tons"].sum())
    delivered = float(recon_summary().get("total_vessel_tons", 0.0))
    counts = scored["risk_level"].value_counts().to_dict()

    return {
        "n_vessels": n,
        "scheduled_cargo_tons": round(scheduled_cargo, 0),
        "delivered_to_vessel_tons": round(delivered, 0),
        "berth_utilisation_pct": round(scheduled_cargo / delivered * 100, 1) if delivered else None,
        "total_exposure_usd": float(scored["risk_usd"].sum()),
        "avg_demurrage_days": round(float(scored["pred_demurrage_days"].mean()), 2),
        "high_risk": int(counts.get("HIGH", 0)),
        "medium_risk": int(counts.get("MEDIUM", 0)),
        "low_risk": int(counts.get("LOW", 0)),
        "high_risk_pct": round(int(counts.get("HIGH", 0)) / n * 100, 1),
    }
