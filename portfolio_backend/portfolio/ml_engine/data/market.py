from __future__ import annotations

"""Market data fetching with cache and retries."""

import hashlib
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from portfolio.ml_engine.config import config
from portfolio.ml_engine.data.cache import atomic_write_parquet


def _cache_key(symbol: str, period: str, interval: str) -> str:
    """Build a stable cache key for a symbol and window."""
    return hashlib.md5(f"{symbol}_{period}_{interval}".encode()).hexdigest()[:12]


def _cache_path(key: str) -> Path:
    """Return cache file path for a given key."""
    config.data.cache_dir.mkdir(parents=True, exist_ok=True)
    return config.data.cache_dir / f"{key}.parquet"


def fetch_history(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
    force_refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV history with local parquet cache.

    Args:
        symbol: Ticker symbol.
        period: yfinance period.
        interval: yfinance interval.
        force_refresh: Bypass cache if True.

    Returns:
        Dataframe of history or None if unavailable.
    """
    key = _cache_key(symbol, period, interval)
    cache_file = _cache_path(key)
    ttl_seconds = config.data.cache_ttl_hours * 3600

    if not force_refresh and cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < ttl_seconds:
            return pd.read_parquet(cache_file)

    for attempt in range(3):
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval, timeout=config.data.yfinance_timeout)
            if data is None or data.empty:
                return None
            if "Adj Close" in data.columns:
                data = data.rename(columns={"Adj Close": "Close"})
            if len(data) < 40:
                return None
            atomic_write_parquet(data, cache_file)
            return data
        except Exception as exc:
            if attempt == 2:
                import logging

                logging.getLogger(__name__).error("Failed to fetch %s after 3 attempts: %s", symbol, exc)
                return None
            time.sleep(2 ** attempt)

    return None
