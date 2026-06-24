"""Haul-fleet telemetry simulator.

Generates hourly telemetry for a fleet of haul trucks with realistic
degradation: a subset of trucks drift into a "pre-failure" regime where
engine temperature, vibration, and oil pressure deteriorate before an
actual DOWN event. The `failed_within_48h` label powers the predictive
maintenance model.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..config import FLEET, SIM_DAYS


def simulate_fleet(rng: np.random.Generator, start: datetime, days: int = SIM_DAYS) -> pd.DataFrame:
    rows: list[dict] = []
    n = FLEET.n_trucks
    truck_ids = [f"HT-{i:03d}" for i in range(1, n + 1)]

    # Each truck has a hidden "degradation clock"; some trucks will fail.
    # time_to_failure_h = hours from start until a DOWN event (np.inf = healthy run)
    will_fail = rng.random(n) < 0.45
    ttf_hours = np.where(
        will_fail,
        rng.uniform(48, days * 24, size=n),
        np.inf,
    )

    total_hours = days * 24
    for h in range(total_hours):
        ts = start + timedelta(hours=h)
        for ti, truck_id in enumerate(truck_ids):
            ttf = ttf_hours[ti]
            hours_to_fail = ttf - h

            # Degradation factor ramps up in the 48h before failure.
            if np.isfinite(ttf) and hours_to_fail <= 48:
                degr = float(np.clip((48 - hours_to_fail) / 48, 0, 1))
            else:
                degr = 0.0

            down = np.isfinite(ttf) and h >= ttf and h < ttf + rng.integers(6, 24)

            if down:
                status = "DOWN"
                payload = 0.0
                cycle = 0.0
                fuel = 0.0
                engine_temp = 0.0
                oil = 0.0
                vib = 0.0
            else:
                # Idle vs running pattern (shift changes, queueing).
                idle = rng.random() < 0.12
                status = "IDLE" if idle else "RUNNING"
                base_load = 0.0 if idle else FLEET.nominal_payload_t
                payload = max(0.0, base_load + rng.normal(0, 8) if not idle else 0.0)
                cycle = 0.0 if idle else max(
                    12.0, FLEET.nominal_cycle_min + rng.normal(0, 3) + degr * 6
                )
                fuel = 0.0 if idle else max(20.0, 95 + rng.normal(0, 10) + degr * 25)
                engine_temp = 88 + rng.normal(0, 2.5) + degr * 18
                oil = 340 + rng.normal(0, 12) - degr * 70
                vib = 2.5 + rng.normal(0, 0.4) + degr * 4.5

            rows.append(
                dict(
                    ts=ts,
                    truck_id=truck_id,
                    payload_t=round(payload, 1),
                    engine_temp_c=round(engine_temp, 1),
                    oil_pressure_kpa=round(oil, 1),
                    vibration_mm_s=round(vib, 2),
                    fuel_rate_lph=round(fuel, 1),
                    cycle_time_min=round(cycle, 1),
                    status=status,
                    failed_within_48h=int(np.isfinite(ttf) and 0 <= (ttf - h) <= 48),
                )
            )

    return pd.DataFrame(rows)
