"""Realistic data simulators for the mine-to-port value chain.

Each module returns a tidy DataFrame whose columns match a table in db.py.
All randomness flows from a single seeded numpy Generator so the whole
dataset is reproducible (SIM_SEED).
"""
