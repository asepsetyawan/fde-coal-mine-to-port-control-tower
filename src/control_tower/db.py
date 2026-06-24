"""Database layer: SQLAlchemy schema + helpers.

SQLite by default (zero-setup), but DATABASE_URL can point at Postgres to
mirror a real enterprise deployment.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import DATABASE_URL

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, future=True)
Base = declarative_base()


class FleetTelemetry(Base):
    __tablename__ = "fleet_telemetry"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, index=True)
    truck_id = Column(String, index=True)
    payload_t = Column(Float)
    engine_temp_c = Column(Float)
    oil_pressure_kpa = Column(Float)
    vibration_mm_s = Column(Float)
    fuel_rate_lph = Column(Float)
    cycle_time_min = Column(Float)
    status = Column(String)          # RUNNING / IDLE / DOWN
    failed_within_48h = Column(Integer)  # label for predictive maintenance


class StageTonnage(Base):
    """Weighbridge / measured tonnage at each value-chain stage per day."""
    __tablename__ = "stage_tonnage"
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, index=True)
    stage = Column(String, index=True)   # PIT/ROM/CRUSHER/PORT/VESSEL
    tons = Column(Float)


class LabQuality(Base):
    """Lab assay results per source stockpile (sparse, lagged)."""
    __tablename__ = "lab_quality"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, index=True)
    stockpile_id = Column(String, index=True)
    tons_available = Column(Float)
    cv = Column(Float)
    ash_pct = Column(Float)
    sulphur_pct = Column(Float)
    tm_pct = Column(Float)


class BargePosition(Base):
    """Simulated AIS positions for barges moving ROM/port -> anchorage."""
    __tablename__ = "barge_position"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, index=True)
    barge_id = Column(String, index=True)
    lat = Column(Float)
    lon = Column(Float)
    speed_kn = Column(Float)
    tons = Column(Float)
    status = Column(String)           # LOADING / IN_TRANSIT / ANCHORED / DISCHARGED
    dest_port = Column(String)


class Vessel(Base):
    """Export vessels with laycan window and contract specification."""
    __tablename__ = "vessel"
    id = Column(Integer, primary_key=True)
    vessel_id = Column(String, index=True)
    laycan_start = Column(DateTime)
    laycan_end = Column(DateTime)
    cargo_tons = Column(Float)
    min_cv = Column(Float)
    max_ash = Column(Float)
    max_sulphur = Column(Float)
    max_tm = Column(Float)
    # Outcome fields (filled by the simulator to create training labels)
    stockpile_ready_date = Column(DateTime)
    demurrage_days = Column(Float)
    status = Column(String)           # SCHEDULED / LOADING / SAILED
    # Operational drivers available before loading (demurrage model features)
    planned_build_rate = Column(Float)
    forecast_rain_days = Column(Float)
    port_congestion = Column(Float)
    quality_margin = Column(Float)
    laycan_window_days = Column(Float)


ALL_TABLES = [FleetTelemetry, StageTonnage, LabQuality, BargePosition, Vessel]


def reset_schema() -> None:
    """Drop and recreate all tables (used by the seed pipeline)."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def init_schema() -> None:
    Base.metadata.create_all(engine)


def write_df(df: pd.DataFrame, table: str) -> None:
    """Bulk-append a DataFrame to a table."""
    if df.empty:
        return
    df.to_sql(table, engine, if_exists="append", index=False)


def read_table(table: str) -> pd.DataFrame:
    """Read an entire table into a DataFrame."""
    try:
        return pd.read_sql_table(table, engine)
    except ValueError:
        return pd.DataFrame()


def query(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Run an arbitrary read query (used by the LLM assistant tools)."""
    with engine.connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)
