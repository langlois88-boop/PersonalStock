import os
from typing import Any, Dict, Optional

import requests


BASE_URL = "https://financialmodelingprep.com/api/v3"


def _get_json(url: str) -> Any:
    try:
        resp = requests.get(url, timeout=12)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def fetch_fmp_fundamentals(symbol: str) -> Dict[str, Optional[float]]:
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        return {"roe": None, "debt_to_equity": None}

    symbol = (symbol or "").upper().strip()
    if not symbol:
        return {"roe": None, "debt_to_equity": None}

    ratios_url = f"{BASE_URL}/ratios-ttm/{symbol}?apikey={api_key}"
    ratios = _get_json(ratios_url) or []
    if not isinstance(ratios, list) or not ratios:
        return {"roe": None, "debt_to_equity": None}

    latest = ratios[0]
    return {
        "roe": _safe_float(latest.get("returnOnEquityTTM")),
        "debt_to_equity": _safe_float(latest.get("debtEquityRatioTTM")),
    }


def fetch_fmp_sentiment(symbol: str) -> Dict[str, Optional[float]]:
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        return {"news_sentiment": None}

    symbol = (symbol or "").upper().strip()
    if not symbol:
        return {"news_sentiment": None}

    news_url = f"{BASE_URL}/stock_news?tickers={symbol}&limit=50&apikey={api_key}"
    items = _get_json(news_url) or []
    if not isinstance(items, list) or not items:
        return {"news_sentiment": 0.0}

    scores = []
    for item in items:
        score = _safe_float(item.get("sentiment"))
        if score is not None:
            scores.append(score)

    if not scores:
        return {"news_sentiment": 0.0}

    return {"news_sentiment": float(sum(scores) / len(scores))}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
