from __future__ import annotations

"""Validation metrics and gates."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from portfolio.ml_engine.config import config


@dataclass(frozen=True)
class GateResult:
    """Result of a quality gate check."""

    passed: bool
    reason: Optional[str] = None


def validate_gate(cv_mean: float, wf_f1: float) -> GateResult:
    """Check model quality gate thresholds."""
    if cv_mean < config.model.min_cv_mean:
        return GateResult(False, f"CV mean {cv_mean:.3f} < {config.model.min_cv_mean}")
    if wf_f1 < config.model.min_wf_f1:
        return GateResult(False, f"Walk-forward F1 {wf_f1:.3f} < {config.model.min_wf_f1}")
    return GateResult(True, None)


def mean_score(values: list[float]) -> float:
    """Compute mean safely."""
    return float(np.mean(values)) if values else 0.0
