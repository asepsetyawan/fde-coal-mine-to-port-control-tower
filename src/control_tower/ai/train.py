"""Train all AI models in one shot.

Run with:  python -m control_tower.ai.train
"""
from __future__ import annotations

from . import demurrage, predictive_maintenance, reconciliation_anomaly


def main() -> None:
    print("Training demurrage model ...")
    dm = demurrage.train()
    print(f"  MAE={dm.mae_days:.2f} days | R2={dm.r2:.2f} "
          f"| train/test={dm.n_train}/{dm.n_test}")

    print("Training predictive maintenance model ...")
    pm = predictive_maintenance.train()
    print(f"  ROC-AUC={pm.roc_auc:.3f} | PR-AUC={pm.pr_auc:.3f} "
          f"| positive_rate={pm.positive_rate:.3f}")

    print("Training reconciliation anomaly detector ...")
    an = reconciliation_anomaly.train()
    print(f"  days={an['n_days']} | flagged={an['n_flagged']}")

    print("All models trained and saved to ./models/")


if __name__ == "__main__":
    main()
