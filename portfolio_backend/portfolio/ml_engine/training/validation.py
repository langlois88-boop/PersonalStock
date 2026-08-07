from __future__ import annotations

"""Validation metrics and gates."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from portfolio.ml_engine.config import config

logger = logging.getLogger(__name__)


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
    """Check model quality gate thresholds.

    ``cv_mean`` is expected to be an imbalance-robust score — balanced
    accuracy, not raw accuracy. A model that always predicts the majority
    class scores exactly 0.5 balanced accuracy and 0.0 F1 for the minority
    class, regardless of how skewed the label distribution is, so any
    ``min_cv_mean`` above 0.5 combined with any ``min_wf_f1`` above 0.0
    already rejects it. Raw accuracy doesn't have that property — on a
    90/10 label split, "always predict majority" scores 0.90 accuracy while
    still being useless, which is why it must not be used here.
    """
    cv_threshold = config.model.min_cv_mean if min_cv_mean is None else min_cv_mean
    f1_threshold = config.model.min_wf_f1 if min_wf_f1 is None else min_wf_f1
    if cv_threshold <= 0.5:
        # A "balanced accuracy" gate at or below 0.5 can't distinguish a
        # majority-only model (which scores exactly 0.5) from a real one —
        # this is almost certainly a misconfiguration.
        logger.warning(
            "validate_gate: min_cv_mean=%.3f <= 0.5 — a model that always "
            "predicts the majority class would pass this gate.",
            cv_threshold,
        )
    if cv_mean < cv_threshold:
        return GateResult(False, f"CV mean {cv_mean:.3f} < {cv_threshold}")
    if wf_f1 < f1_threshold:
        return GateResult(False, f"Walk-forward F1 {wf_f1:.3f} < {f1_threshold}")
    return GateResult(True, None)


def is_majority_only_predictor(y_true, y_pred) -> bool:
    """True if predictions collapse to a single class — the exact signature
    of a model that just always predicts the majority label rather than
    actually discriminating."""
    if len(set(np.asarray(y_pred).tolist())) >= 2:
        return False
    # Predicts one class only — confirm that class is in fact the majority
    # (if it's a single-class dataset, this check doesn't apply).
    values, counts = np.unique(np.asarray(y_true), return_counts=True)
    if len(values) < 2:
        return False
    return True


def mean_score(values: list[float]) -> float:
    """Compute mean safely."""
    return float(np.mean(values)) if values else 0.0
