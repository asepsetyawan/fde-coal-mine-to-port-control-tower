"""FastAPI backend for the Coal Mine-to-Port Control Tower.

Run with:  uvicorn control_tower.api.main:app --reload --port 8000
Docs at:   http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..ai import blending, demurrage, predictive_maintenance, reconciliation_anomaly
from ..ai.assistant import ask
from ..db import read_table
from ..pipeline.reconciliation import reconciliation_frame, summary

app = FastAPI(
    title="Coal Mine-to-Port Control Tower",
    description="Simulated enterprise control tower with surgical AI (FDE portfolio).",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/reconciliation/summary")
def recon_summary() -> dict:
    s = summary()
    if not s:
        raise HTTPException(404, "No data. Seed the database first.")
    return s


@app.get("/reconciliation/daily")
def recon_daily() -> list[dict]:
    return reconciliation_frame().to_dict(orient="records")


@app.get("/reconciliation/anomalies")
def recon_anomalies() -> list[dict]:
    df = reconciliation_anomaly.detect()
    return df[df["anomaly"] == 1].to_dict(orient="records")


@app.get("/demurrage/vessels")
def demurrage_vessels() -> list[dict]:
    return demurrage.predict_vessels().to_dict(orient="records")


@app.get("/demurrage/feature-importance")
def demurrage_features() -> list[dict]:
    return demurrage.feature_importances().to_dict(orient="records")


@app.get("/demurrage/summary")
def demurrage_summary() -> dict:
    return demurrage.summary()


class BlendRequest(BaseModel):
    target_tons: float = 65_000.0
    min_cv: int | None = None
    max_ash: float | None = None
    max_sulphur: float | None = None
    max_tm: float | None = None


@app.post("/blend/optimize")
def blend_optimize(req: BlendRequest) -> dict:
    from ..config import DEFAULT_CONTRACT, ContractSpec

    spec = ContractSpec(
        min_cv=req.min_cv or DEFAULT_CONTRACT.min_cv,
        max_ash=req.max_ash or DEFAULT_CONTRACT.max_ash,
        max_sulphur=req.max_sulphur or DEFAULT_CONTRACT.max_sulphur,
        max_tm=req.max_tm or DEFAULT_CONTRACT.max_tm,
    )
    r = blending.optimize_blend(target_tons=req.target_tons, spec=spec)
    return r.__dict__


@app.get("/fleet/health")
def fleet_health() -> list[dict]:
    return predictive_maintenance.score_fleet().to_dict(orient="records")


@app.get("/barge/positions")
def barge_positions() -> list[dict]:
    df = read_table("barge_position")
    if df.empty:
        return []
    latest = df.sort_values("ts").groupby("barge_id", as_index=False).last()
    return latest.to_dict(orient="records")


@app.get("/barge/tracks")
def barge_tracks(points: int = 10) -> list[dict]:
    """Recent AIS points per barge for movement trail visualization."""
    df = read_table("barge_position")
    if df.empty:
        return []
    points = max(2, min(points, 30))
    trails: list[dict] = []
    for barge_id, group in df.sort_values("ts").groupby("barge_id"):
        trail = group.tail(points)
        trails.append({
            "barge_id": barge_id,
            "points": trail[["lat", "lon", "ts", "status", "speed_kn", "tons"]].to_dict(orient="records"),
        })
    return trails


class AskRequest(BaseModel):
    question: str


@app.post("/assistant/ask")
def assistant_ask(req: AskRequest) -> dict:
    return ask(req.question)
