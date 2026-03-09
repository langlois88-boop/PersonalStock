from __future__ import annotations

"""Labeling utilities for training datasets."""

import numpy as np
import pandas as pd


def triple_barrier_labels(
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    up_pct: float = 0.15,
    down_pct: float = 0.07,
    max_days: int = 20,
) -> pd.Series:
    """Generate triple-barrier labels.

    Args:
        close: Close price series.
        high: High price series.
        low: Low price series.
        up_pct: Upper barrier percentage.
        down_pct: Down barrier percentage.
        max_days: Max look-ahead days.

    Returns:
        Series of labels (0/1) with NaNs for insufficient horizon.
    """
    if close is None or close.empty:
        return pd.Series(dtype=float)
    prices = close.values
    highs = high.values if high is not None and not high.empty else prices
    lows = low.values if low is not None and not low.empty else prices
    labels = np.full(len(prices), np.nan)
    for i in range(len(prices) - 1):
        entry = prices[i]
        if entry <= 0:
            continue
        upper = entry * (1 + up_pct)
        lower = entry * (1 - down_pct)
        end = min(len(prices), i + max_days + 1)
        hit = 0
        for j in range(i + 1, end):
            if highs[j] >= upper:
                hit = 1
                break
            if lows[j] <= lower:
                hit = 0
                break
        labels[i] = hit
    return pd.Series(labels, index=close.index)


def forward_return_labels(close: pd.Series, horizon: int = 8, target_pct: float = 0.02) -> pd.Series:
    """Simple forward return label.

    Args:
        close: Close price series.
        horizon: Forward steps.
        target_pct: Target return threshold.

    Returns:
        Series of binary labels.
    """
    future = close.shift(-horizon)
    future_return = (future / close) - 1.0
    return (future_return >= target_pct).astype(int)
