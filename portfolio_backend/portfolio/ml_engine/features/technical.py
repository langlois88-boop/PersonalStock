from __future__ import annotations

"""Technical indicator feature functions."""

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder EMA smoothing.

    Args:
        close: Close prices.
        period: RSI period.

    Returns:
        RSI series.
    """
    if len(close) < period + 1:
        return pd.Series(np.nan, index=close.index, name="rsi")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("rsi")


def sma_ratio(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """SMA ratio feature.

    Args:
        close: Close prices.
        fast: Fast window.
        slow: Slow window.

    Returns:
        SMA ratio series.
    """
    sma_fast = close.rolling(fast).mean()
    sma_slow = close.rolling(slow).mean().replace(0, np.nan)
    return (sma_fast / sma_slow).fillna(1.0).rename(f"sma_ratio_{fast}_{slow}")


def volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume z-score clipped to [-3, 3]."""
    mean = volume.rolling(period).mean()
    std = volume.rolling(period).std().replace(0, np.nan)
    return ((volume - mean) / std).clip(-3, 3).fillna(0).rename("volume_zscore")


def rvol(volume: pd.Series, period: int = 20) -> pd.Series:
    """Relative volume clipped to [0, 10]."""
    mean = volume.rolling(period).mean().replace(0, np.nan)
    return (volume / mean).clip(0, 10).fillna(1.0).rename("rvol")


def volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """Rolling volatility of returns."""
    return close.pct_change().rolling(period).std().rename("volatility")


def rubber_band_index(close: pd.Series, period: int = 20) -> pd.Series:
    """Distance from SMA20 normalized by 2*std20."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std().replace(0, np.nan)
    return ((close - sma) / (2 * std)).fillna(0).rename("rubber_band_index")


def macd_signal(close: pd.Series) -> pd.Series:
    """MACD line minus signal line."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return (macd_line - signal_line).rename("macd_signal")
