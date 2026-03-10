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


def validate_gate(
    cv_mean: float,
    wf_f1: float,
    min_cv_mean: float | None = None,
    min_wf_f1: float | None = None,
) -> GateResult:
    """Check model quality gate thresholds."""
    cv_threshold = config.model.min_cv_mean if min_cv_mean is None else min_cv_mean
    f1_threshold = config.model.min_wf_f1 if min_wf_f1 is None else min_wf_f1
    if cv_mean < cv_threshold:
        return GateResult(False, f"CV mean {cv_mean:.3f} < {cv_threshold}")
    if wf_f1 < f1_threshold:
        return GateResult(False, f"Walk-forward F1 {wf_f1:.3f} < {f1_threshold}")
    return GateResult(True, None)


def mean_score(values: list[float]) -> float:
    """Compute mean safely."""
    return float(np.mean(values)) if values else 0.0
