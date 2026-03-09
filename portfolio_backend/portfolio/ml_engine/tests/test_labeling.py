import numpy as np
import pandas as pd

from portfolio.ml_engine.training.labeling import triple_barrier_labels


def test_labels_are_binary() -> None:
    close = pd.Series(100 + np.random.randn(200).cumsum())
    labels = triple_barrier_labels(close, up_pct=0.10, down_pct=0.05, max_days=10)
    valid = labels.dropna()
    assert set(valid.unique()).issubset({0, 1})
