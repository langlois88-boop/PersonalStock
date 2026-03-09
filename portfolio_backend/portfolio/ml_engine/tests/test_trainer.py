import numpy as np
import pandas as pd
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
