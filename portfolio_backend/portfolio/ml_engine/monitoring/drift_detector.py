from __future__ import annotations

"""Feature drift detection utilities."""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from scipy import stats


class FeatureDriftDetector:
    """Detects feature distribution drift via KS tests."""

    ALERT_THRESHOLD = 0.15

    def __init__(self, baseline_path: Path) -> None:
        self.baseline: Optional[dict] = None
        if baseline_path.exists():
            self.baseline = json.loads(baseline_path.read_text())

    def save_baseline(self, X: pd.DataFrame, path: Path) -> None:
        """Persist baseline feature distributions.

        Args:
            X: Training feature dataframe.
            path: Output path for baseline json.
        """
        baseline = {
            col: {
                "mean": float(X[col].mean()),
                "std": float(X[col].std()),
                "values_sample": X[col].dropna().tolist()[:500],
            }
            for col in X.columns
        }
        path.write_text(json.dumps(baseline, indent=2))
        self.baseline = baseline

    def check(self, X_live: pd.DataFrame) -> dict[str, float]:
        """Check for drift with KS statistic.

        Args:
            X_live: Live feature dataframe.

        Returns:
            Dict of features with drift stats.
        """
        if self.baseline is None:
            return {}
        alerts: dict[str, float] = {}
        for col in X_live.columns:
            if col not in self.baseline:
                continue
            baseline_vals = self.baseline[col]["values_sample"]
            live_vals = X_live[col].dropna().tolist()
            if len(live_vals) < 30:
                continue
            ks_stat, _ = stats.ks_2samp(baseline_vals, live_vals)
            if ks_stat > self.ALERT_THRESHOLD:
                alerts[col] = float(ks_stat)
        if alerts:
            print(f"⚠️  Feature drift detected: {alerts}")
        return alerts
