from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.requests import StockSnapshotRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass
except Exception:  # pragma: no cover - optional dependency
    StockHistoricalDataClient = None
    StockBarsRequest = None
    StockLatestTradeRequest = None
    TimeFrame = None
    StockSnapshotRequest = None
    TradingClient = None
    GetAssetsRequest = None
    AssetClass = None

from .patterns import enrich_bars_with_patterns
from . import market_data as market_data


def _alpaca_client() -> StockHistoricalDataClient | None:
    if StockHistoricalDataClient is None:
        return None
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    if not api_key or not api_secret:
        return None
    return StockHistoricalDataClient(api_key, api_secret)


def _alpaca_trading_client() -> TradingClient | None:
    if TradingClient is None:
        return None
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    if not api_key or not api_secret:
        return None
    paper = os.getenv('ALPACA_PAPER', 'true').strip().lower() in {'1', 'true', 'yes', 'y'}
    return TradingClient(api_key, api_secret, paper=paper)


def get_tradable_symbols(limit: int = 500) -> list[str]:
    client = _alpaca_trading_client()
    if client is None or GetAssetsRequest is None or AssetClass is None:
        return []
    try:
        request = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)
        assets = client.get_all_assets(request)
        symbols = [a.symbol for a in assets if getattr(a, 'tradable', False)]
        return symbols[:limit]
    except Exception:
        return []


def get_stock_snapshots(symbols: list[str]) -> dict[str, Any]:
    client = _alpaca_client()
    if client is None or StockSnapshotRequest is None:
        return {}
    if not symbols:
        return {}
    try:
        request = StockSnapshotRequest(symbol_or_symbols=symbols)
        snapshots = client.get_stock_snapshot(request)
        return snapshots or {}
    except Exception:
        return {}


def _bars_to_frame(bars: Any) -> pd.DataFrame:
    if bars is None:
        return pd.DataFrame()
    try:
        df = bars.df
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    if 'timestamp' not in df.columns and 'time' in df.columns:
        df = df.rename(columns={'time': 'timestamp'})
    return df


def get_intraday_bars_range(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    client = _alpaca_client()
    if client is None or TimeFrame is None:
        return pd.DataFrame()
    symbol = (symbol or '').strip().upper()
    if not symbol:
        return pd.DataFrame()
    try:
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
        )
        bars = client.get_stock_bars(request)
        df = _bars_to_frame(bars)
        if df.empty:
            return df
        df = df[df.get('symbol', symbol) == symbol] if 'symbol' in df.columns else df
        return df.sort_values('timestamp')
    except Exception:
        return pd.DataFrame()


def get_intraday_bars(symbol: str, minutes: int = 390) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes)
    df = get_intraday_bars_range(symbol, start=start, end=end)
    if df is not None and not df.empty:
        return df
    # fallback: last 5 trading days, then take last N minutes
    fallback_start = end - timedelta(days=5)
    fallback_df = get_intraday_bars_range(symbol, start=fallback_start, end=end)
    if fallback_df is None or fallback_df.empty:
        try:
            hist = market_data.Ticker(symbol).history(period='5d', interval='1m', timeout=8)
            if hist is None or hist.empty:
                return pd.DataFrame()
            hist = hist.reset_index()
            if 'Datetime' in hist.columns and 'timestamp' not in hist.columns:
                hist = hist.rename(columns={'Datetime': 'timestamp'})
            if 'Date' in hist.columns and 'timestamp' not in hist.columns:
                hist = hist.rename(columns={'Date': 'timestamp'})
            if 'Open' in hist.columns:
                hist = hist.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                })
            cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in hist.columns for col in cols):
                return pd.DataFrame()
            return hist[cols].tail(minutes)
        except Exception:
            return pd.DataFrame()
    return fallback_df.tail(minutes)


def get_daily_bars(symbol: str, days: int = 30) -> pd.DataFrame:
    client = _alpaca_client()
    if client is None or TimeFrame is None:
        return pd.DataFrame()
    symbol = (symbol or '').strip().upper()
    if not symbol:
        return pd.DataFrame()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    try:
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        bars = client.get_stock_bars(request)
        df = _bars_to_frame(bars)
        if df.empty:
            return df
        df = df[df.get('symbol', symbol) == symbol] if 'symbol' in df.columns else df
        return df.sort_values('timestamp')
    except Exception:
        return pd.DataFrame()


def get_latest_trade_price(symbol: str) -> float | None:
    client = _alpaca_client()
    if client is None or StockLatestTradeRequest is None:
        return None
    symbol = (symbol or '').strip().upper()
    if not symbol:
        return None
    try:
        request = StockLatestTradeRequest(symbol_or_symbols=[symbol])
        trade = client.get_stock_latest_trade(request)
        if isinstance(trade, dict):
            trade = trade.get(symbol) or next(iter(trade.values()), None)
        if trade is None:
            return None
        price = getattr(trade, 'price', None)
        if price is None:
            price = getattr(trade, 'p', None)
        return float(price) if price is not None else None
    except Exception:
        return None


def get_intraday_context(symbol: str, minutes: int = 390, rvol_window: int = 20) -> dict[str, Any] | None:
    df = get_intraday_bars(symbol, minutes=minutes)
    if df.empty:
        return None
    enriched = enrich_bars_with_patterns(df, rvol_window=rvol_window)
    if enriched.empty:
        return None
    last = enriched.iloc[-1]
    return {
        'bars': enriched,
        'rvol': float(last.get('rvol') or 0.0),
        'pattern_signal': float(last.get('pattern_signal') or 0.0),
        'patterns': last.get('patterns') or [],
        'volatility': float(last.get('volatility') or 0.0),
        'rsi14': float(last.get('rsi14') or 0.0),
        'ema20': float(last.get('ema20') or 0.0),
        'ema50': float(last.get('ema50') or 0.0),
        'last_close': float(last.get('close') or 0.0),
    }
