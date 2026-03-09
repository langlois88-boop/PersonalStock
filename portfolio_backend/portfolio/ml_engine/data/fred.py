from __future__ import annotations

"""FRED data retrieval helpers."""

from typing import Optional
import requests

from portfolio.ml_engine.config import config


def fetch_series_latest(series_id: str) -> Optional[float]:
    """Fetch the latest value for a FRED series.

    Args:
        series_id: FRED series code (e.g., GS10).

    Returns:
        Latest float value or None.
    """
    if not config.integration.fred_api_key:
        return None
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "api_key": config.integration.fred_api_key,
        "series_id": series_id,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        observations = data.get("observations") or []
        if not observations:
            return None
        value = observations[0].get("value")
        return float(value) if value not in (None, ".") else None
    except Exception:
        return None
