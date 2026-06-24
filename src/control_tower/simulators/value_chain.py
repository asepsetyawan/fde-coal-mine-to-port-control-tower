"""Stage tonnage + lab quality + barge AIS simulators.

Tonnage flows PIT -> ROM -> CRUSHER -> PORT -> VESSEL. Each downstream
stage loses a little mass (moisture loss, spillage, measurement variance),
and we inject occasional anomalies (calibration drift / suspected theft)
for the reconciliation anomaly detector to catch.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..config import STOCKPILES, VALUE_CHAIN_STAGES

# Normal cumulative retention at each stage relative to PIT.
STAGE_RETENTION = {
    "PIT": 1.000,
    "ROM": 0.985,
    "CRUSHER": 0.972,
    "PORT": 0.955,
    "VESSEL": 0.948,
}


def simulate_stage_tonnage(rng: np.random.Generator, start: datetime, days: int) -> pd.DataFrame:
    rows: list[dict] = []
    for d in range(days):
        date = start + timedelta(days=d)
        # Daily pit production with weekly seasonality + rain disruption.
        rain = rng.random() < 0.25
        base = rng.normal(42_000, 3_500) * (0.6 if rain else 1.0)
        pit_tons = max(5_000.0, base)

        for stage in VALUE_CHAIN_STAGES:
            retention = STAGE_RETENTION[stage]
            tons = pit_tons * retention * (1 + rng.normal(0, 0.004))

            # Inject anomalies: ~4% of port readings have abnormal shrinkage
            # (calibration drift or suspected theft) -> reconciliation flags it.
            if stage == "PORT" and rng.random() < 0.04:
                tons *= rng.uniform(0.88, 0.94)

            rows.append(dict(date=date, stage=stage, tons=round(tons, 1)))
    return pd.DataFrame(rows)


def simulate_lab_quality(rng: np.random.Generator, start: datetime, days: int) -> pd.DataFrame:
    """Sparse, lagged assays per source stockpile (a few per day)."""
    rows: list[dict] = []
    for d in range(days):
        for sp_id, prof in STOCKPILES.items():
            # 1-3 assays per stockpile per day at random times.
            for _ in range(int(rng.integers(1, 4))):
                ts = start + timedelta(days=d, hours=float(rng.uniform(0, 24)))
                rows.append(
                    dict(
                        ts=ts,
                        stockpile_id=sp_id,
                        tons_available=round(float(rng.uniform(8_000, 60_000)), 1),
                        cv=round(prof["cv"] + rng.normal(0, 120), 1),
                        ash_pct=round(prof["ash"] + rng.normal(0, 0.8), 2),
                        sulphur_pct=round(prof["sulphur"] + rng.normal(0, 0.05), 3),
                        tm_pct=round(prof["tm"] + rng.normal(0, 1.2), 2),
                    )
                )
    return pd.DataFrame(rows)


# Rough geo path: Kalimantan river mouth -> anchorage (illustrative coords).
_ROUTE = [
    (-3.55, 114.62),  # loading point (river)
    (-3.62, 114.55),
    (-3.71, 114.45),
    (-3.80, 114.38),  # anchorage / transshipment
]


def simulate_barge_positions(rng: np.random.Generator, start: datetime, days: int) -> pd.DataFrame:
    """Simulate barge AIS cycles (load -> transit -> anchor -> discharge).

    Each barge starts at a random phase and we stop emitting as soon as we
    pass the period end *mid-cycle*. Because barges are staggered, the final
    position per barge (what the live snapshot shows) lands on a realistic
    mix of statuses instead of everyone idle at DISCHARGED.
    """
    rows: list[dict] = []
    n_barges = 8
    end = start + timedelta(days=days)

    for b in range(1, n_barges + 1):
        barge_id = f"BG-{b:02d}"
        # Stagger each barge across a couple of days so phases differ.
        t = start + timedelta(hours=float(rng.uniform(0, 48)))
        tons = round(float(rng.uniform(7_000, 9_000)), 1)
        done = False

        while not done:
            # LOADING at the river point.
            if t >= end:
                break
            rows.append(_pos(barge_id, t, _ROUTE[0], 0.0, tons, "LOADING"))
            t += timedelta(hours=float(rng.uniform(6, 12)))

            # IN_TRANSIT along the route to the anchorage.
            for lat, lon in _ROUTE[1:]:
                if t >= end:
                    done = True
                    break
                jitter = (rng.normal(0, 0.01), rng.normal(0, 0.01))
                rows.append(
                    _pos(barge_id, t, (lat + jitter[0], lon + jitter[1]),
                         float(rng.uniform(6, 10)), tons, "IN_TRANSIT")
                )
                t += timedelta(hours=float(rng.uniform(3, 6)))
            if done:
                break

            # ANCHORED waiting to discharge.
            if t >= end:
                break
            rows.append(_pos(barge_id, t, _ROUTE[-1], 0.0, tons, "ANCHORED"))
            t += timedelta(hours=float(rng.uniform(8, 20)))

            # DISCHARGED, then return leg before the next cycle.
            if t >= end:
                break
            rows.append(_pos(barge_id, t, _ROUTE[-1], 0.0, 0.0, "DISCHARGED"))
            t += timedelta(hours=float(rng.uniform(4, 10)))

    return pd.DataFrame(rows)


def _pos(barge_id, ts, latlon, speed, tons, status) -> dict:
    return dict(
        ts=ts,
        barge_id=barge_id,
        lat=round(latlon[0], 5),
        lon=round(latlon[1], 5),
        speed_kn=round(speed, 1),
        tons=tons,
        status=status,
        dest_port="Taboneo Anchorage",
    )
