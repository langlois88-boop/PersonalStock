from unittest.mock import patch

import numpy as np
import pandas as pd

from portfolio.ml_engine.feature_registry import CRYPTO_FEATURE_NAMES, PENNY_FEATURE_NAMES


def _synthetic_ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = 100 + np.cumsum(rng.randn(n) * 0.5)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": rng.uniform(100, 1000, n),
    })


def test_crypto_feature_names_match_what_crypto_training_computes() -> None:
    """Regression test: CRYPTO_FEATURE_NAMES used to list renamed/aspirational
    columns (rubber_band_20, price_to_vwap_20, macd_hist, bb_pct_b_20,
    stoch_k_14, obv_zscore) that crypto_training.py::_build_crypto_features
    never actually produced, so build_crypto_dataset()'s
    `dataset.dropna(subset=feature_cols + ['label'])` raised a guaranteed
    KeyError on any real run."""
    from portfolio.ml_engine.crypto_training import build_crypto_dataset

    df = _synthetic_ohlcv()
    with patch("portfolio.ml_engine.crypto_training.fetch_crypto_history", return_value=df):
        dataset, labels, feature_cols = build_crypto_dataset(["BTC-USD"], days=8)

    assert feature_cols == list(CRYPTO_FEATURE_NAMES)
    assert not dataset.empty
    for col in feature_cols:
        assert col in dataset.columns


def test_penny_feature_names_match_what_penny_pipeline_computes() -> None:
    """Regression test: PENNY_FEATURE_NAMES used to list 20 features, 13 of
    which train_penny_model.py never computed and silently zero-padded
    instead of erroring — diluting the model with dead constant columns."""
    from portfolio.ml_engine.pipelines.penny_pipeline import PENNY_FEATURES

    assert PENNY_FEATURES == list(PENNY_FEATURE_NAMES)
    assert len(PENNY_FEATURE_NAMES) == 7
