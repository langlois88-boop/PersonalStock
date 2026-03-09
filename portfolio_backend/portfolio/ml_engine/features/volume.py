from __future__ import annotations

"""Volume-related features."""

import pandas as pd

from portfolio.ml_engine.features.technical import rvol, volume_zscore


def volume_features(volume: pd.Series, period: int = 20) -> pd.DataFrame:
    """Compute volume z-score and RVOL features.

    Args:
        volume: Volume series.
        period: Rolling period.

    Returns:
        Dataframe with volume_zscore and rvol.
    """
    return pd.DataFrame({
        "volume_zscore_20": volume_zscore(volume, period=period),
        "rvol_20": rvol(volume, period=period),
    })
