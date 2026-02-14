import os
from typing import Optional

import requests


BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_fred_latest(series_id: str) -> Optional[float]:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key or not series_id:
        return None

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1,
    }

    try:
        resp = requests.get(BASE_URL, params=params, timeout=12)
        if resp.status_code != 200:
            return None
        payload = resp.json() or {}
    except Exception:
        return None

    observations = payload.get("observations") or []
    if not observations:
        return None

    value = observations[0].get("value")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
