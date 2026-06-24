"""Seed the database with a full simulated mine-to-port dataset.

Run with:  python -m control_tower.pipeline.seed
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from ..config import SIM_DAYS, SIM_SEED
from ..db import reset_schema, write_df
from ..simulators.fleet import simulate_fleet
from ..simulators.value_chain import (
    simulate_barge_positions,
    simulate_lab_quality,
    simulate_stage_tonnage,
)
from ..simulators.vessels import simulate_vessels


def main(days: int = SIM_DAYS, seed: int = SIM_SEED) -> None:
    rng = np.random.default_rng(seed)
    start = (datetime.utcnow() - timedelta(days=days)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    print(f"Resetting schema and simulating {days} days from {start:%Y-%m-%d} ...")
    reset_schema()

    print("  - fleet telemetry")
    write_df(simulate_fleet(rng, start, days), "fleet_telemetry")

    print("  - stage tonnage")
    write_df(simulate_stage_tonnage(rng, start, days), "stage_tonnage")

    print("  - lab quality")
    write_df(simulate_lab_quality(rng, start, days), "lab_quality")

    print("  - barge positions (AIS)")
    write_df(simulate_barge_positions(rng, start, days), "barge_position")

    print("  - vessels")
    write_df(simulate_vessels(rng, start, days), "vessel")

    print("Seed complete. Next: python -m control_tower.ai.train")


if __name__ == "__main__":
    main()
