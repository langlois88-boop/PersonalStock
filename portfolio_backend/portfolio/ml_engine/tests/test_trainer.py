import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from portfolio.ml_engine.training.trainer import Trainer


def test_trainer_runs() -> None:
    X = pd.DataFrame({"a": np.random.randn(200), "b": np.random.randn(200)})
    y = pd.Series([0, 1] * 100)

    def pipeline_factory() -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
        ])

    trainer = Trainer(pipeline_factory)
    result = trainer.fit(X, y)
    assert result.cv_mean >= 0.0
    assert result.wf_f1 >= 0.0


def test_trainer_rejects_majority_only_model_on_imbalanced_labels() -> None:
    """Regression test: cv_mean must be an imbalance-robust score, not raw
    accuracy. A model that always predicts the majority class here would
    score 0.90 raw accuracy on this 90/10 split — comfortably above the
    default 0.55 threshold — yet it must still fail the gate."""
    n = 300
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n)})
    # Positives evenly spaced (1 every 10 rows) so every purged CV fold sees
    # both classes, keeping the metrics below deterministic across folds.
    y = pd.Series([1 if i % 10 == 9 else 0 for i in range(n)])
    assert y.mean() == 0.1

    def pipeline_factory() -> Pipeline:
        return Pipeline([("classifier", DummyClassifier(strategy="most_frequent"))])

    trainer = Trainer(pipeline_factory)
    result = trainer.fit(X, y)

    assert result.cv_mean == pytest.approx(0.5)
    assert result.wf_f1 == 0.0
    assert result.passed_gate is False
    assert result.gate_reason is not None
