from __future__ import annotations

import numpy as np
import pandas as pd


def fractional_diff_weights(d: float, size: int) -> np.ndarray:
    weights = [1.0]
    for k in range(1, size):
        weights.append(weights[-1] * (d - k + 1) / k)
    return np.array(weights[::-1], dtype=float)


def fractional_diff_series(series: pd.Series, d: float = 0.4, threshold: float = 1e-3) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=float)
    series = pd.to_numeric(series, errors="coerce")
    weights = fractional_diff_weights(d, len(series))
    weights = weights[np.abs(weights) > threshold]
    width = len(weights)
    if width <= 1:
        return pd.Series(0.0, index=series.index)
    diff = np.full(len(series), np.nan)
    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1 : i + 1].values
        if np.isnan(window).any():
            continue
        diff[i] = np.dot(weights, window)
    return pd.Series(diff, index=series.index).fillna(0.0)
