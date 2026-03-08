from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

# On regroupe tous les imports Alpaca dans le bloc sécurisé
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest, StockSnapshotRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed  # <--- Crucial pour le plan gratuit
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest, MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
    from alpaca.trading.enums import AssetClass, OrderSide, TimeInForce
except Exception:  # pragma: no cover
    StockHistoricalDataClient = None
    StockBarsRequest = None
    StockLatestTradeRequest = None
    TimeFrame = None
    StockSnapshotRequest = None
    DataFeed = None
    TradingClient = None
    GetAssetsRequest = None
    MarketOrderRequest = None
    LimitOrderRequest = None
    GetOrdersRequest = None
    AssetClass = None
    OrderSide = None
    TimeInForce = None
    OrderStatus = None

from .patterns import enrich_bars_with_patterns


def _alpaca_data_feed() -> Any:
    if DataFeed is None:
        return None
    feed = os.getenv('ALPACA_DATA_FEED', 'IEX').strip().upper()
    if feed == 'SIP':
        return DataFeed.SIP
    return DataFeed.IEX


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


def get_trading_client() -> TradingClient | None:
    return _alpaca_trading_client()


def get_account() -> Any | None:
    client = _alpaca_trading_client()
    if client is None:
        return None
    try:
        return client.get_account()
    except Exception:
        return None


def get_open_positions() -> list[Any]:
    client = _alpaca_trading_client()
    if client is None:
        return []
    try:
        return client.get_all_positions() or []
    except Exception:
        return []


def submit_market_order(symbol: str, qty: int, side: str) -> Any | None:
    client = _alpaca_trading_client()
    if client is None or MarketOrderRequest is None or OrderSide is None or TimeInForce is None:
        return None
    symbol = (symbol or '').strip().upper()
    if not symbol or qty <= 0:
        return None
    try:
        side_enum = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side_enum,
            time_in_force=TimeInForce.DAY,
        )
        return client.submit_order(request)
    except Exception:
        return None


def submit_limit_order(symbol: str, qty: int, side: str, limit_price: float) -> Any | None:
    client = _alpaca_trading_client()
    if client is None or LimitOrderRequest is None or OrderSide is None or TimeInForce is None:
        return None
    symbol = (symbol or '').strip().upper()
    if not symbol or qty <= 0 or limit_price <= 0:
        return None
    try:
        side_enum = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side_enum,
            time_in_force=TimeInForce.DAY,
            limit_price=round(float(limit_price), 4),
        )
        return client.submit_order(request)
    except Exception:
        return None


def get_order_by_id(order_id: str) -> Any | None:
    client = _alpaca_trading_client()
    if client is None:
        return None
    try:
        return client.get_order_by_id(order_id)
    except Exception:
        return None


def close_position(symbol: str) -> Any | None:
    client = _alpaca_trading_client()
    if client is None:
        return None
    try:
        return client.close_position(symbol)
    except Exception:
        return None


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


def get_recent_orders(lookback_days: int = 7) -> list[Any]:
    client = _alpaca_trading_client()
    if client is None:
        return []
    try:
        if GetOrdersRequest is not None:
            after = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            return list(client.get_orders(GetOrdersRequest(status='all', after=after)) or [])
        return client.get_orders() or []
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


def get_latest_bid_ask_spread_pct(symbol: str) -> float | None:
    symbol = (symbol or '').strip().upper()
    if not symbol:
        return None
    snapshots = get_stock_snapshots([symbol])
    if not snapshots:
        return None
    snapshot = snapshots.get(symbol) or next(iter(snapshots.values()), None)
    if snapshot is None:
        return None
    quote = getattr(snapshot, 'latest_quote', None) or getattr(snapshot, 'quote', None)
    if quote is None:
        return None
    bid = getattr(quote, 'bid_price', None)
    ask = getattr(quote, 'ask_price', None)
    if bid is None:
        bid = getattr(quote, 'bp', None)
    if ask is None:
        ask = getattr(quote, 'ap', None)
    try:
        bid = float(bid) if bid is not None else None
        ask = float(ask) if ask is not None else None
    except Exception:
        bid = None
        ask = None
    if bid is None or ask is None or ask <= 0:
        return None
    mid = (bid + ask) / 2
    if mid <= 0:
        return None
    return float((ask - bid) / mid)


def get_order_book_imbalance(symbol: str) -> float | None:
    """Return bid/ask size ratio from latest quote if available."""
    symbol = (symbol or '').strip().upper()
    if not symbol:
        return None
    snapshots = get_stock_snapshots([symbol])
    if not snapshots:
        return None
    snapshot = snapshots.get(symbol) or next(iter(snapshots.values()), None)
    if snapshot is None:
        return None
    quote = getattr(snapshot, 'latest_quote', None) or getattr(snapshot, 'quote', None)
    if quote is None:
        return None
    bid_size = getattr(quote, 'bid_size', None)
    ask_size = getattr(quote, 'ask_size', None)
    if bid_size is None:
        bid_size = getattr(quote, 'bs', None)
    if ask_size is None:
        ask_size = getattr(quote, 'as', None)
    try:
        bid_size = float(bid_size) if bid_size is not None else None
        ask_size = float(ask_size) if ask_size is not None else None
    except Exception:
        return None
    if bid_size is None or ask_size is None or ask_size <= 0:
        return None
    return float(bid_size / ask_size)


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
    # On vérifie que DataFeed est disponible pour le plan gratuit
    if client is None or TimeFrame is None or DataFeed is None:
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
            feed=_alpaca_data_feed()
        )
        bars = client.get_stock_bars(request)
        df = _bars_to_frame(bars)
        if df.empty:
            return df
        
        # Filtrage par symbole pour éviter les résidus de MultiIndex
        if 'symbol' in df.columns:
            df = df[df['symbol'] == symbol]
            
        return df.sort_values('timestamp')
    except Exception:
        return pd.DataFrame()


def get_intraday_bars(symbol: str, minutes: int = 390) -> pd.DataFrame:
    """
    Récupère les bougies intraday avec fallback automatique sur Yahoo Finance
    si Alpaca ne renvoie rien (week-end ou actions hors IEX).
    """
    symbol = (symbol or '').strip().upper()
    
    # 1. Gestion de la fenêtre de temps
    end = datetime.now(timezone.utc)
    if end.weekday() >= 5: # Si samedi ou dimanche, on recule au vendredi
        end = end - timedelta(days=end.weekday() - 4)
    
    max_days = 3
    start = max(end - timedelta(minutes=minutes), end - timedelta(days=max_days))
    
    # 2. Tentative via Alpaca
    df = get_intraday_bars_range(symbol, start=start, end=end)
    if df is None or df.empty:
        # Re-tente avec une fenêtre plus large (pré-marché / avant ouverture)
        start = end - timedelta(days=max_days)
        df = get_intraday_bars_range(symbol, start=start, end=end)
    
    # 3. Si Alpaca échoue ou est vide, on utilise le Fallback Yahoo
    if df is None or df.empty:
        return _get_yahoo_fallback(symbol, minutes, max_days, start=start, end=end)
    
    return df.tail(minutes)


def _get_yahoo_fallback(
    symbol: str,
    minutes: int,
    max_days: int,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Fonction de secours robuste utilisant Yahoo Finance via market_data"""
    try:
        from . import market_data

        symbol = (symbol or '').strip().upper()
        if not symbol:
            return pd.DataFrame()

        period = f"{max_days}d"
        df = market_data.download(
            symbol,
            period=period,
            interval='1m',
            start=start,
            end=end,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            if symbol in df.columns.get_level_values(0):
                df = df[symbol]
            else:
                df = df.xs(df.columns.levels[0][0], axis=1)

        df = df.copy()
        if 'timestamp' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df['timestamp'] = df.index
            elif 'Datetime' in df.columns:
                df = df.rename(columns={'Datetime': 'timestamp'})
            elif 'Date' in df.columns:
                df = df.rename(columns={'Date': 'timestamp'})

        if 'Adj Close' in df.columns and 'Close' in df.columns:
            df = df.drop(columns=['Adj Close'])

        rename_map = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'close',
            'Volume': 'volume',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
            df = df.dropna(subset=['timestamp'])

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                df[col] = pd.NA

        df = df.sort_values('timestamp')
        return df.tail(minutes)
    except Exception:
        return pd.DataFrame()


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
        # Note: Pour les daily bars, feed=IEX est moins critique mais recommandé pour la cohérence
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=_alpaca_data_feed()
        )
        bars = client.get_stock_bars(request)
        df = _bars_to_frame(bars)
        if df.empty:
            return df
        
        if 'symbol' in df.columns:
            df = df[df['symbol'] == symbol]
            
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
        # Pour le prix "latest", Alpaca gratuit a souvent 15min de délai sur IEX
        request = StockLatestTradeRequest(symbol_or_symbols=[symbol], feed=_alpaca_data_feed())
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
    
    # Enrichissement avec les indicateurs techniques (RSI, EMA, Patterns)
    enriched = enrich_bars_with_patterns(df, rvol_window=rvol_window)
    if enriched.empty:
        return None
    
    last = enriched.iloc[-1]
    spread_pct = get_latest_bid_ask_spread_pct(symbol)
    imbalance = get_order_book_imbalance(symbol)
    try:
        vol_series = enriched['volume'].rolling(20, min_periods=5).mean()
        trade_velocity = float((last.get('volume') or 0.0) / (vol_series.iloc[-1] or 1.0))
    except Exception:
        trade_velocity = 0.0
    return {
        'bars': enriched,
        'rvol': float(last.get('rvol') or 0.0),
        'pattern_signal': float(last.get('pattern_signal') or 0.0),
        'patterns': last.get('patterns') or [],
        'volatility': float(last.get('volatility') or 0.0),
        'rsi14': float(last.get('rsi14') or 0.0),
        'ema20': float(last.get('ema20') or 0.0),
        'ema50': float(last.get('ema50') or 0.0),
        'vwap': float(last.get('vwap') or 0.0),
        'price_to_vwap': float(last.get('price_to_vwap') or 0.0),
        'last_close': float(last.get('close') or 0.0),
        'bid_ask_spread_pct': float(spread_pct or 0.0),
        'order_book_imbalance': float(imbalance or 0.0),
        'trade_velocity': float(trade_velocity or 0.0),
    }