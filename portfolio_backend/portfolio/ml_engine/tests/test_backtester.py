from pathlib import Path

import numpy as np
import pandas as pd

from portfolio.ml_engine.backtester import train_fusion_model


def test_train_fusion_model_rejects_uninformative_majority_only_model(tmp_path: Path) -> None:
    """Regression test: train_fusion_model used to have no quality gate at
    all — it always joblib.dump()'d and returned whatever RandomForest it
    fit, regardless of whether it had actually learned anything. With no
    real feature signal (every engineered feature column is left unset, so
    `_ensure_features` fills them all with a constant 0.0), the model can't
    outperform a majority-only predictor, and must now be rejected (return
    None) instead of silently deployed."""
    rng = np.random.RandomState(0)
    n = 400
    close = 100 + np.cumsum(rng.randn(n) * 0.3 + 0.05)
    df = pd.DataFrame({
        "Close": close,
        "Returns": pd.Series(close).pct_change().fillna(0),
    })

    payload = train_fusion_model(df, model_path=tmp_path / "fusion_test.pkl")
    assert payload is None
