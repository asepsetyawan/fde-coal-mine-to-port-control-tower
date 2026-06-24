"""AI layer applied surgically to three uncertainty problems.

  - demurrage.py               : forecast vessel waiting cost (ML regression)
  - blending.py                : optimize blend to meet spec at min cost (LP)
  - predictive_maintenance.py  : flag haul trucks likely to fail (ML classifier)
  - reconciliation_anomaly.py  : separate normal loss from leakage (anomaly det.)
  - assistant.py               : natural-language access (LLM + rule fallback)

Everything else in the control tower is deterministic data engineering;
AI is the scalpel, not the hammer.
"""
