from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Iterable

import pandas as pd

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except Exception:  # pragma: no cover - optional dependency
    StockHistoricalDataClient = None
    StockBarsRequest = None
    StockSnapshotRequest = None
    TimeFrame = None
    TimeFrameUnit = None

try:
    import yfinance as _yfinance
except Exception:  # pragma: no cover - optional dependency
    _yfinance = None

from .alpaca_data import get_daily_bars, get_intraday_bars_range, get_latest_trade_price


SYMBOL_ALIASES = {
    '^GSPC': 'SPY',
    '^IXIC': 'QQQ',
    '^DJI': 'DIA',
    '^VIX': 'VIXY',
    'DX-Y.NYB': 'UUP',
    'CL=F': 'USO',
    'GC=F': 'GLD',
}


def _yf_timeout() -> float:
    try:
        return float(os.getenv('YF_TIMEOUT_SEC', '4'))
    except Exception:
        return 4.0


def _allow_yf_price_fallback(symbol: str) -> bool:
    flag = str(os.getenv('ALLOW_YF_PRICE_FALLBACK', '')).strip().lower()
    if flag in {'1', 'true', 'yes', 'y'}:
        return True
    symbol = (symbol or '').upper()
    return '.' in symbol


def _with_timeout(func, timeout: float, default):
    if func is None:
        return default
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            return future.result(timeout=timeout)
    except TimeoutError:
        return default
    except Exception:
        return default


def _alpaca_client() -> StockHistoricalDataClient | None:
    if StockHistoricalDataClient is None:
        return None
    from os import getenv

    api_key = getenv('ALPACA_API_KEY')
    api_secret = getenv('ALPACA_SECRET_KEY')
    if not api_key or not api_secret:
        return None
    return StockHistoricalDataClient(api_key, api_secret)


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _period_to_days(period: str | None) -> int:
    if not period:
        return 365
    period = str(period).strip().lower()
    if period.endswith('d'):
        return int(float(period[:-1]) or 1)
    if period.endswith('mo'):
        return int(float(period[:-2]) * 30)
    if period.endswith('y'):
        return int(float(period[:-1]) * 365)
    return 365


def _timeframe_from_interval(interval: str | None) -> Any:
    if TimeFrame is None:
        return None
    interval = (interval or '1d').strip().lower()
    if interval.endswith('m'):
        amount = int(interval[:-1] or 1)
        if TimeFrameUnit is None:
            return TimeFrame.Minute
        return TimeFrame(amount, TimeFrameUnit.Minute)
    if interval.endswith('h'):
        amount = int(interval[:-1] or 1)
        if TimeFrameUnit is None:
            return TimeFrame.Hour
        return TimeFrame(amount, TimeFrameUnit.Hour)
    return TimeFrame.Day


def _normalize_symbols(tickers: str | Iterable[str]) -> list[str]:
    if isinstance(tickers, str):
        symbols = [s.strip().upper() for s in tickers.replace(',', ' ').split() if s.strip()]
    else:
        symbols = [str(s).strip().upper() for s in tickers if str(s).strip()]
    return [s for s in symbols if s]


def _map_symbol(symbol: str) -> str:
    symbol = (symbol or '').strip().upper()
    if not symbol:
        return symbol
    return SYMBOL_ALIASES.get(symbol, symbol)


def _bars_for_symbols(symbols: list[str], start: datetime, end: datetime, interval: str) -> pd.DataFrame:
    client = _alpaca_client()
    if client is None or StockBarsRequest is None:
        return pd.DataFrame()
    timeframe = _timeframe_from_interval(interval)
    if timeframe is None:
        return pd.DataFrame()
    try:
        request = StockBarsRequest(symbol_or_symbols=symbols, timeframe=timeframe, start=start, end=end)
        bars = client.get_stock_bars(request)
        df = bars.df if bars is not None else None
    except Exception:
        df = None
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    if 'timestamp' not in df.columns and 'time' in df.columns:
        df = df.rename(columns={'time': 'timestamp'})
    return df


def _frame_from_bars(symbol: str, bars: pd.DataFrame) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame()
    frame = bars[bars.get('symbol', symbol) == symbol] if 'symbol' in bars.columns else bars
    if frame.empty:
        return pd.DataFrame()
    frame = frame.copy()
    if 'timestamp' in frame.columns:
        frame = frame.set_index('timestamp')
    frame = frame.sort_index()
    rename_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
    }
    for key, value in rename_map.items():
        if key in frame.columns:
            frame[value] = frame[key]
    cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in frame.columns]
    return frame[cols]


def download(
    tickers: str | Iterable[str],
    period: str | None = None,
    interval: str | None = '1d',
    start: Any | None = None,
    end: Any | None = None,
    group_by: str | None = None,
    threads: bool | None = None,
    auto_adjust: bool | None = None,
    progress: bool | None = None,
) -> pd.DataFrame:
    symbols = _normalize_symbols(tickers)
    if not symbols:
        return pd.DataFrame()
    symbol_pairs = [(symbol, _map_symbol(symbol)) for symbol in symbols]
    mapped_symbols = sorted({mapped for _, mapped in symbol_pairs if mapped})
    start_dt = _parse_datetime(start)
    end_dt = _parse_datetime(end) or datetime.now(timezone.utc)
    if start_dt is None:
        days = _period_to_days(period or '1y')
        start_dt = end_dt - timedelta(days=days)
    bars = _bars_for_symbols(mapped_symbols, start_dt, end_dt, interval or '1d')
    if not symbols:
        return pd.DataFrame()
    if len(symbols) == 1:
        return _frame_from_bars(symbol_pairs[0][1], bars)
    frames = {}
    for original, mapped in symbol_pairs:
        frame = _frame_from_bars(mapped, bars)
        if frame is not None and not frame.empty:
            frames[original] = frame
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def screen(scr_id: str, count: int = 200) -> dict[str, Any]:
    if _yfinance is None:
        return {'quotes': []}
    timeout = _yf_timeout()
    return _with_timeout(lambda: _yfinance.screen(scr_id, count=count) or {'quotes': []}, timeout, {'quotes': []})


class Ticker:
    def __init__(self, symbol: str):
        self.symbol = (symbol or '').strip().upper()
        self._mapped = _map_symbol(self.symbol)

    @property
    def info(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if _yfinance is not None:
            timeout = _yf_timeout()
            payload = _with_timeout(lambda: _yfinance.Ticker(self.symbol).info or {}, timeout, {})
        price = get_latest_trade_price(self._mapped)
        if price is not None:
            payload['regularMarketPrice'] = price
        return payload

    @property
    def news(self) -> list[dict[str, Any]]:
        if _yfinance is None:
            return []
        timeout = _yf_timeout()
        return _with_timeout(lambda: _yfinance.Ticker(self.symbol).news or [], timeout, [])

    @property
    def calendar(self) -> pd.DataFrame:
        if _yfinance is None:
            return pd.DataFrame()
        timeout = _yf_timeout()
        return _with_timeout(lambda: _yfinance.Ticker(self.symbol).calendar, timeout, pd.DataFrame())

    @property
    def dividends(self) -> pd.Series:
        if _yfinance is None:
            return pd.Series(dtype='float64')
        timeout = _yf_timeout()
        return _with_timeout(lambda: _yfinance.Ticker(self.symbol).dividends, timeout, pd.Series(dtype='float64'))

    @property
    def balance_sheet(self) -> pd.DataFrame:
        if _yfinance is None:
            return pd.DataFrame()
        timeout = _yf_timeout()
        return _with_timeout(lambda: _yfinance.Ticker(self.symbol).balance_sheet, timeout, pd.DataFrame())

    @property
    def financials(self) -> pd.DataFrame:
        if _yfinance is None:
            return pd.DataFrame()
        timeout = _yf_timeout()
        return _with_timeout(lambda: _yfinance.Ticker(self.symbol).financials, timeout, pd.DataFrame())

    def history(
        self,
        period: str | None = None,
        interval: str | None = '1d',
        start: Any | None = None,
        end: Any | None = None,
        timeout: int | None = None,
        auto_adjust: bool | None = None,
        prepost: bool | None = None,
    ) -> pd.DataFrame:
        data = download(
            self._mapped,
            period=period,
            interval=interval,
            start=start,
            end=end,
        )
        if isinstance(data, pd.DataFrame) and not data.empty:
            return data
        if _yfinance is None or not _allow_yf_price_fallback(self.symbol):
            return pd.DataFrame()
        timeout = _yf_timeout()
        fallback = _with_timeout(
            lambda: _yfinance.Ticker(self.symbol).history(
                period=period,
                interval=interval,
                start=start,
                end=end,
                timeout=int(timeout),
                auto_adjust=auto_adjust,
                prepost=prepost,
            ),
            timeout,
            pd.DataFrame(),
        )
        return fallback if isinstance(fallback, pd.DataFrame) else pd.DataFrame()
