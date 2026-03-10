from __future__ import annotations

"""Generic training loop with time-series validation."""

import logging
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

from portfolio.ml_engine.training.validation import validate_gate

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Training result payload."""

    model: object
    feature_names: list[str]
    cv_scores: list[float]
    cv_mean: float
    wf_f1: float
    n_samples: int
    label_balance: float
    passed_gate: bool
    gate_reason: Optional[str] = None


class Trainer:
    """Generic walk-forward trainer for any sklearn pipeline."""

    def __init__(self, pipeline_factory: Callable[[], object], n_splits: int = 5) -> None:
        self.pipeline_factory = pipeline_factory
        self.n_splits = n_splits

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        min_cv_mean: float | None = None,
        min_wf_f1: float | None = None,
    ) -> TrainResult:
        """Fit model with time-series cross validation.

        Args:
            X: Feature dataframe.
            y: Label series.
            sample_weight: Optional sample weights.

        Returns:
            TrainResult summary.
        """
        if len(set(y)) < 2:
            raise ValueError(f"Only one class in labels — cannot train. Balance: {y.mean():.3f}")
        if len(y) < 50:
            raise ValueError(f"Too few samples: {len(y)}. Minimum 50 required.")

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        cv_scores: list[float] = []
        wf_f1_scores: list[float] = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            pipe = clone(self.pipeline_factory())
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
            sw_tr = None if sample_weight is None else np.array(sample_weight)[train_idx]

            fit_params = {}
            if sw_tr is not None:
                fit_params["classifier__sample_weight"] = sw_tr

            pipe.fit(X_tr, y_tr, **fit_params)
            preds = pipe.predict(X_te)
            cv_scores.append(float(accuracy_score(y_te, preds)))
            wf_f1_scores.append(float(f1_score(y_te, preds, zero_division=0)))

            logger.info("Fold %s/%s: acc=%.3f f1=%.3f", fold + 1, self.n_splits, cv_scores[-1], wf_f1_scores[-1])

        final_pipe = self.pipeline_factory()
        final_fit_params = {}
        if sample_weight is not None:
            final_fit_params["classifier__sample_weight"] = sample_weight
        final_pipe.fit(X, y, **final_fit_params)

        cv_mean = float(np.mean(cv_scores)) if cv_scores else 0.0
        wf_f1 = float(np.mean(wf_f1_scores)) if wf_f1_scores else 0.0
        gate = validate_gate(cv_mean, wf_f1, min_cv_mean=min_cv_mean, min_wf_f1=min_wf_f1)

        return TrainResult(
            model=final_pipe,
            feature_names=list(X.columns),
            cv_scores=cv_scores,
            cv_mean=cv_mean,
            wf_f1=wf_f1,
            n_samples=len(y),
            label_balance=float(y.mean()),
            passed_gate=gate.passed,
            gate_reason=gate.reason,
        )
