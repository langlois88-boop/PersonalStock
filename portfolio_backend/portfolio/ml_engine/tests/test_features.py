import numpy as np
import pandas as pd

from portfolio.ml_engine.features.technical import rsi, sma_ratio, volume_zscore, rvol


def make_close(n: int = 100, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 + rng.normal(0, 2, n).cumsum())


def test_rsi_range() -> None:
    close = make_close()
    result = rsi(close, 14).dropna()
    assert (result >= 0).all() and (result <= 100).all()


def test_rsi_wilder_not_simple() -> None:
    close = make_close()
    wilder = rsi(close, 14).dropna()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    simple = (100 - 100 / (1 + rs)).dropna()
    assert not np.allclose(wilder.values[-20:], simple.values[-20:], atol=0.1)


def test_volume_zscore_clipped() -> None:
    vol = pd.Series([1e6] * 80 + [1e9, 1e9])
    result = volume_zscore(vol, 20).dropna()
    assert result.max() <= 3.0 and result.min() >= -3.0


def test_rsi_too_short_returns_nan() -> None:
    close = make_close(5)
    result = rsi(close, 14)
    assert result.isna().all()


def test_sma_ratio_neutral_when_equal() -> None:
    close = pd.Series([100.0] * 100)
    result = sma_ratio(close, 10, 50).dropna()
    assert np.allclose(result.values, 1.0)


def test_rvol_clipped() -> None:
    vol = pd.Series([100] * 50 + [1000] * 50)
    result = rvol(vol, 20).dropna()
    assert result.max() <= 10.0 and result.min() >= 0.0
