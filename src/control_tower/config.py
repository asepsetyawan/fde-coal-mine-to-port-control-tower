"""Central configuration loaded from environment / .env.

Domain constants (contract specs, demurrage rates, value-chain stages) live
here so the simulators, AI models, and frontend share one source of truth.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'control_tower.db'}")

# --- LLM assistant ---------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- Simulation knobs ------------------------------------------------------
SIM_SEED = int(os.getenv("SIM_SEED", "42"))
SIM_DAYS = int(os.getenv("SIM_DAYS", "30"))
SIM_TRUCKS = int(os.getenv("SIM_TRUCKS", "50"))

# --- Domain model ----------------------------------------------------------
# Tonnage flows through these stages; reconciliation compares adjacent stages.
VALUE_CHAIN_STAGES = ["PIT", "ROM", "CRUSHER", "PORT", "VESSEL"]

# Source stockpiles feeding the blend, each with a typical quality profile.
# CV is calorific value (kcal/kg, GAR). Higher CV = higher price.
STOCKPILES = {
    "SP-HIGH": {"name": "High CV seam", "cv": 5200, "ash": 8.0, "sulphur": 0.55, "tm": 18.0, "cost_per_ton": 42.0},
    "SP-MID":  {"name": "Mid CV seam",  "cv": 4400, "ash": 11.0, "sulphur": 0.70, "tm": 24.0, "cost_per_ton": 33.0},
    "SP-LOW":  {"name": "Low CV seam",  "cv": 3800, "ash": 14.0, "sulphur": 0.95, "tm": 30.0, "cost_per_ton": 26.0},
}

# Demurrage: penalty (USD/day) when a vessel waits beyond laytime.
DEMURRAGE_USD_PER_DAY = 22_000.0

# Reference selling price anchor (USD/ton) at a reference CV, used to
# value "quality giveaway" in the blending optimizer.
REFERENCE_PRICE_USD_PER_TON = 52.0
REFERENCE_CV = 4200


@dataclass(frozen=True)
class ContractSpec:
    """Buyer contract specification for a coal cargo."""
    min_cv: int = 4200          # minimum calorific value (kcal/kg GAR)
    max_ash: float = 12.0       # maximum ash %
    max_sulphur: float = 0.8    # maximum sulphur %
    max_tm: float = 26.0        # maximum total moisture %


DEFAULT_CONTRACT = ContractSpec()


@dataclass(frozen=True)
class FleetConfig:
    """Haul-fleet simulation parameters."""
    n_trucks: int = SIM_TRUCKS
    nominal_payload_t: float = 220.0      # Komatsu 730E-class
    nominal_cycle_min: float = 25.0
    base_failure_rate: float = 0.03       # per truck-shift baseline
    samples_per_truck_per_day: int = 24   # hourly telemetry


FLEET = FleetConfig()
