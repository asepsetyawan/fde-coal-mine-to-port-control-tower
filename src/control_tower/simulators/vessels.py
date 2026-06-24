"""Export vessel simulator with demurrage outcome labels.

Each vessel has a laycan window and a contract spec. Whether it incurs
demurrage depends on latent operational drivers (stockpile build rate,
rain during the window, port congestion, and whether the available coal
can meet spec without rework). These same drivers are exposed as features
so the demurrage model has something real to learn.

Physical realism: the number of vessels and their cargo parcels are sized
so that total scheduled cargo ~= coal actually delivered to the port over
the period (see PORT_THROUGHPUT_PER_DAY). This keeps berth utilisation
near 100% instead of 6x oversubscribed, so demurrage stays realistic
(~15-30% high risk) rather than flagging almost every vessel.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..config import DEFAULT_CONTRACT

# Approx. coal reaching the port stage per day (tons). Mirrors the stage
# tonnage simulator (pit ~42k/day, ~36k after rain + chain retention).
PORT_THROUGHPUT_PER_DAY = 36_000.0

# Coal parcels loaded at anchorage by barges (realistic for the Taboneo
# transshipment model where barges feed vessels). Average ~11.5k tons.
# Smaller parcels keep total demand ~= port supply while giving enough
# shipments for the demurrage model to learn from.
CARGO_CHOICES = [8_000, 10_000, 12_000, 16_000]
AVG_CARGO = float(np.mean(CARGO_CHOICES))


def simulate_vessels(rng: np.random.Generator, start: datetime, days: int) -> pd.DataFrame:
    rows: list[dict] = []

    # Size the vessel programme to the port's actual throughput so total
    # demand ~= supply (berth utilisation ~95-108%).
    utilisation = float(rng.uniform(0.95, 1.08))
    capacity_tons = PORT_THROUGHPUT_PER_DAY * days * utilisation
    n_vessels = max(24, round(capacity_tons / AVG_CARGO))

    for i in range(1, n_vessels + 1):
        vessel_id = f"MV-{2600 + i}"
        laycan_start = start + timedelta(days=float(rng.uniform(0, days)))
        window = float(rng.uniform(3, 6))
        laycan_end = laycan_start + timedelta(days=window)
        cargo = float(rng.choice(CARGO_CHOICES))

        # --- Latent operational drivers (also used as features) -----------
        # Build/load rate to the vessel's parcel (tons/day via barges).
        planned_build_rate = max(1_800.0, float(rng.normal(4_000, 1_200)))
        forecast_rain_days = float(rng.poisson(1.0))
        port_congestion = float(np.clip(rng.normal(0.45, 0.2), 0, 1))
        # Quality margin: how comfortably sources beat min CV (higher = easier).
        quality_margin = float(rng.normal(250, 180))

        # --- Outcome: days needed to build a spec parcel ------------------
        effective_rate = planned_build_rate * (1 - 0.06 * forecast_rain_days) \
            * (1 - 0.30 * port_congestion)
        effective_rate = max(1_500.0, effective_rate)
        days_needed = cargo / effective_rate
        # Off-spec coal forces re-blending/rework -> extra time.
        rework_days = 0.0 if quality_margin > 0 else float(rng.uniform(0.5, 2.0))

        ready_offset = days_needed + rework_days + float(rng.normal(0, 0.2))

        # Demurrage accrues if the parcel isn't ready within the laycan
        # window, plus a small penalty when the berth is very congested.
        demurrage_days = max(0.0, ready_offset - window) + max(0.0, port_congestion - 0.7) * 1.2
        demurrage_days = round(max(0.0, demurrage_days + rng.normal(0, 0.05)), 2)

        rows.append(
            dict(
                vessel_id=vessel_id,
                laycan_start=laycan_start,
                laycan_end=laycan_end,
                cargo_tons=cargo,
                min_cv=DEFAULT_CONTRACT.min_cv,
                max_ash=DEFAULT_CONTRACT.max_ash,
                max_sulphur=DEFAULT_CONTRACT.max_sulphur,
                max_tm=DEFAULT_CONTRACT.max_tm,
                stockpile_ready_date=laycan_start + timedelta(days=max(0.2, ready_offset)),
                demurrage_days=demurrage_days,
                status="SCHEDULED",
                # --- feature columns (available before loading) ----------
                planned_build_rate=round(planned_build_rate, 1),
                forecast_rain_days=forecast_rain_days,
                port_congestion=round(port_congestion, 3),
                quality_margin=round(quality_margin, 1),
                laycan_window_days=round(window, 2),
            )
        )
    return pd.DataFrame(rows)


# Feature columns used by the demurrage model.
FEATURE_COLS = [
    "cargo_tons",
    "planned_build_rate",
    "forecast_rain_days",
    "port_congestion",
    "quality_margin",
    "laycan_window_days",
]
