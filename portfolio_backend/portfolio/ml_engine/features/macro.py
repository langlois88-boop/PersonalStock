from __future__ import annotations

"""Macro feature helpers."""

from typing import Optional

from portfolio.ml_engine.data.fred import fetch_series_latest


def fred_latest(series_id: str) -> Optional[float]:
    """Fetch latest FRED value for a series."""
    return fetch_series_latest(series_id)
