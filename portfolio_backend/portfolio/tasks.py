from __future__ import annotations

import os
from uuid import uuid4
from pathlib import Path
from functools import lru_cache
from datetime import date, datetime, timedelta, time as dt_time, timezone as dt_timezone
from math import isfinite
from typing import Any

from . import market_data as yf
import yfinance as yfin
import requests
import random
import json
import pandas as pd
from time import sleep
from zoneinfo import ZoneInfo
from celery import shared_task
from django.conf import settings
from django.core.mail import send_mail
from django.core.management import call_command
from django.db import models
from django.core.cache import cache
from django.utils import timezone
import finnhub
import feedparser
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
import re
import praw
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
try:
    from google import genai
except Exception:  # pragma: no cover - optional dependency
    genai = None
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfgen import canvas as pdf_canvas
except Exception:  # pragma: no cover - optional dependency
    letter = (612.0, 792.0)
    inch = 72.0
    pdfmetrics = None
    pdf_canvas = None

from .models import (
    Account,
    AccountTransaction,
    AlertEvent,
    ActiveSignal,
    DividendHistory,
    DripSnapshot,
    MacroIndicator,
    DataQADaily,
    ModelCalibrationDaily,
    ModelDriftDaily,
    ModelRegistry,
    ModelEvaluationDaily,
    Portfolio,
    PortfolioHolding,
    PortfolioDigest,
    Transaction,
    PennyStockSnapshot,
    PennyStockUniverse,
    PennySignal,
    PriceHistory,
    Prediction,
    PaperTrade,
    SandboxWatchlist,
    MasterWatchlistEntry,
    Stock,
    StockNews,
    TaskRunLog,
    SystemLog,
    UserPreference,
)
from .services.danas_broker import DanasMLRouter
from .ai_module import run_predictions
from .alpaca_data import (
    get_daily_bars,
    get_intraday_context,
    get_intraday_bars,
    get_intraday_bars_range,
    get_latest_trade_price,
    get_latest_bid_ask_spread_pct,
    get_order_book_imbalance,
    get_account,
    get_open_positions,
    get_recent_orders,
    submit_market_order,
    submit_limit_order,
    get_order_by_id,
    close_position,
    get_stock_snapshots,
    get_trading_client,
    get_tradable_symbols,
)
from .patterns import enrich_bars_with_patterns
from .crypto_processor import scan_crypto_drip
from .ml_engine.engine.data_fusion import DataFusionEngine
from .ml_engine.backtester import (
    AIBacktester,
    FEATURE_COLUMNS,
    apply_feature_weighting_to_signal,
    get_model_version,
    get_model_path,
    load_or_train_model,
    save_model_payload,
    train_fusion_model,
    train_fusion_model_from_labels,
)
from .ai_advisor import DeepSeekAdvisor


DEFAULT_YIELD = 0.02
NON_FUNDAMENTAL_SYMBOLS = {
    'TEC.TO',
    'BTC-CAD',
}


def _is_crypto_symbol(symbol: str) -> bool:
    symbol_upper = (symbol or '').upper()
    return '-' in symbol_upper and symbol_upper.endswith(('CAD', 'USD', 'USDT'))


def _symbol_currency(symbol: str) -> str:
    symbol_upper = (symbol or '').upper()
    if symbol_upper.endswith(('.TO', '.V')):
        return 'CAD'
    if _is_crypto_symbol(symbol_upper):
        return 'CAD' if symbol_upper.endswith('CAD') else 'USD'
    return 'USD'


def _cache_key(symbol: str, suffix: str) -> str:
    date_key = _ny_time_now().strftime('%Y%m%d')
    return f"trade_monitor:{symbol}:{suffix}:{date_key}"


def _latest_bid_ask(symbol: str) -> tuple[float | None, float | None]:
    try:
        info = yf.Ticker(symbol).info or {}
        bid = info.get('bid')
        ask = info.get('ask')
        bid = float(bid) if bid is not None else None
        ask = float(ask) if ask is not None else None
        return bid, ask
    except Exception:
        return None, None


def _gemini_trade_reason(symbol: str, signal_payload: dict[str, Any] | None) -> str | None:
    api_key = getattr(settings, 'GEMINI_AI_API_KEY', None)
    if not api_key:
        return None
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        model_name = getattr(settings, 'GEMINI_AI_MODEL', 'models/gemini-2.5-flash')
        features = (signal_payload or {}).get('features') or {}
        explanations = (signal_payload or {}).get('explanations') or []
        prompt = (
            "Explique en 2 phrases max pourquoi ce trade paper est pris. "
            "Sois concis et factuel.\n"
            f"Ticker: {symbol}\n"
            f"Signal: {(signal_payload or {}).get('signal')}\n"
            f"Features: {json.dumps(features, ensure_ascii=False)}\n"
            f"Top factors: {json.dumps(explanations, ensure_ascii=False)}"
        )
        response = client.models.generate_content(model=model_name, contents=prompt)
        text = (getattr(response, 'text', None) or '').strip()
        return text or None
    except Exception:
        return None


def _gemini_dynamic_recommendation(symbol: str, action: str, context: dict[str, Any]) -> str | None:
    api_key = getattr(settings, 'GEMINI_AI_API_KEY', None)
    if not api_key:
        return None
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        model_name = getattr(settings, 'GEMINI_AI_MODEL', 'models/gemini-2.5-flash')
        prompt = (
            "Reformule en 1-2 phrases une recommandation de trading, claire et actionnable. "
            "Le verbe d'action doit être l'un des suivants: Vendre, Garder, Étendre.\n"
            f"Ticker: {symbol}\n"
            f"Action: {action}\n"
            f"Contexte: {json.dumps(context, ensure_ascii=False)}"
        )
        response = client.models.generate_content(model=model_name, contents=prompt)
        text = (getattr(response, 'text', None) or '').strip()
        return text or None
    except Exception:
        return None


def _compute_adx(frame: pd.DataFrame, length: int = 14) -> tuple[float | None, float | None]:
    if frame is None or frame.empty or not {'High', 'Low', 'Close'}.issubset(frame.columns):
        return None, None
    try:
        import pandas_ta as ta

        adx = ta.adx(frame['High'], frame['Low'], frame['Close'], length=length)
        if adx is None or adx.empty:
            return None, None
        col = [c for c in adx.columns if c.upper().startswith('ADX')]
        if not col:
            return None, None
        series = adx[col[0]].dropna()
        if len(series) < 2:
            return float(series.iloc[-1]) if len(series) else None, None
        return float(series.iloc[-1]), float(series.iloc[-1] - series.iloc[-2])
    except Exception:
        return None, None


def _atr_from_frame(frame: pd.DataFrame, length: int = 14) -> float | None:
    if frame is None or frame.empty or not {'High', 'Low', 'Close'}.issubset(frame.columns):
        return None
    try:
        high = frame['High']
        low = frame['Low']
        close = frame['Close']
        tr = pd.concat([
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(length).mean().iloc[-1]
        return float(atr) if atr is not None else None
    except Exception:
        return None


def _classify_tier(
    frame: pd.DataFrame,
    signal_payload: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if frame is None or frame.empty:
        return 'T2', ['fallback']
    last = frame.tail(1).iloc[0]
    volume_z = float(last.get('VolumeZ') or 0.0)
    rvol10 = float(last.get('RVOL10') or 0.0)
    sentiment = float(last.get('sentiment_score') or 0.0)
    news_count = float(last.get('news_count') or 0.0)
    close = float(last.get('Close') or 0.0)
    ma20 = float(last.get('MA20') or 0.0)
    ma200 = float(last.get('MA200') or 0.0)
    rsi14 = float(last.get('RSI14') or 0.0)
    volatility = float(last.get('Volatility') or 0.0)

    vol_spike = volume_z >= float(os.getenv('TIER1_VOLUMEZ_MIN', '2.5')) or rvol10 >= float(os.getenv('TIER1_RVOL_MIN', '5'))
    if vol_spike:
        reasons.append('volume_spike')
    news_impact = news_count >= float(os.getenv('TIER1_NEWS_MIN', '1')) and sentiment >= float(os.getenv('TIER1_SENTIMENT_MIN', '0.6'))
    if news_impact:
        reasons.append('news_impact')

    breakout = False
    try:
        if 'Close' in frame.columns and len(frame) >= 60:
            recent_high = float(frame['Close'].rolling(252, min_periods=60).max().iloc[-1])
            breakout = recent_high > 0 and close >= recent_high * float(os.getenv('TIER1_BREAKOUT_PCT', '0.995'))
    except Exception:
        breakout = False
    if breakout:
        reasons.append('breakout')

    if vol_spike and news_impact and breakout:
        return 'T1', reasons

    tier2_ok = close > ma20 and close > ma200 and 40 <= rsi14 <= 70
    if tier2_ok:
        reasons.append('swing_setup')
        return 'T2', reasons

    if volatility >= float(os.getenv('TIER3_VOLATILITY_MIN', '0.03')):
        reasons.append('intraday_volatility')
    return 'T3', reasons


def _parse_driver_map(raw: str | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    raw = (raw or '').strip()
    if not raw:
        return mapping
    for chunk in raw.split(','):
        if '=' not in chunk:
            continue
        symbol, driver = chunk.split('=', 1)
        symbol = symbol.strip().upper()
        driver = driver.strip().upper()
        if symbol and driver:
            mapping[symbol] = driver
    return mapping


def _default_driver_for_symbol(symbol: str) -> str:
    symbol_upper = (symbol or '').strip().upper()
    if symbol_upper.endswith(('.TO', '.V', '.CN')):
        return '^GSPTSE'
    return 'SPY'


def get_market_regime_context() -> dict[str, Any]:
    cache_key = 'market_regime_context'
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    try:
        engine = DataFusionEngine('SPY')
        df = engine.fuse_all()
        if df is None or df.empty:
            result = {'status': 'no_data'}
            cache.set(cache_key, result, timeout=60 * 5)
            return result
        last = df.tail(1).iloc[0]
        spy_close = float(last.get('Close') or 0.0)
        ema200 = float(last.get('MA200') or 0.0)
        vix = float(last.get('VIXCLS') or 0.0)
        risk_off = (vix >= float(os.getenv('MARKET_RISK_OFF_VIX', '25'))) or (ema200 > 0 and spy_close < ema200)
        result = {
            'status': 'ok',
            'spy_close': spy_close,
            'spy_ema200': ema200,
            'vix': vix,
            'risk_off': bool(risk_off),
        }
        cache.set(cache_key, result, timeout=60 * 5)
        return result
    except Exception as exc:
        result = {'status': 'error', 'error': str(exc)}
        cache.set(cache_key, result, timeout=60 * 2)
        return result


@shared_task
def market_guard_check() -> dict[str, Any]:
    """Audit market regime and high spread risks on open trades."""
    regime = get_market_regime_context()
    spread_limit = float(os.getenv('MARKET_GUARD_SPREAD_LIMIT', '0.02'))
    results: list[dict[str, Any]] = []
    open_trades = PaperTrade.objects.filter(status='OPEN')
    for trade in open_trades:
        symbol = (trade.ticker or '').strip().upper()
        if not symbol:
            continue
        spread_pct = get_latest_bid_ask_spread_pct(symbol)
        if spread_pct is None:
            bid, ask = _latest_bid_ask(symbol)
            if bid and ask and ask > 0:
                spread_pct = float((ask - bid) / ((ask + bid) / 2))
        if spread_pct is None:
            continue
        if spread_pct >= spread_limit:
            message = f"⚠️ Spread élevé sur {symbol}: {spread_pct * 100:.2f}%"
            AlertEvent.objects.create(category='MARKET_GUARD_SPREAD', message=message)
            results.append({'symbol': symbol, 'spread_pct': spread_pct})
    return {
        'regime': regime,
        'spread_limit': spread_limit,
        'alerts': results,
    }


@shared_task
def order_book_imbalance_alerts(symbols: list[str] | None = None) -> dict[str, Any]:
    """Emit alerts when bid/ask depth is weak (if data available)."""
    if symbols is None:
        symbols = []
        for sandbox in ['AI_PENNY', 'AI_BLUECHIP', 'WATCHLIST']:
            stored = SandboxWatchlist.objects.filter(sandbox=sandbox).first()
            if stored and stored.symbols:
                symbols.extend([str(s).strip().upper() for s in stored.symbols if str(s).strip()])
    symbols = sorted(set([s for s in symbols if s]))
    threshold = float(os.getenv('ORDER_BOOK_IMBALANCE_MIN', '1.0'))
    cooldown = int(os.getenv('ORDER_BOOK_IMBALANCE_COOLDOWN_MIN', '30'))
    results: dict[str, Any] = {}
    for symbol in symbols:
        imbalance = get_order_book_imbalance(symbol)
        if imbalance is None:
            results[symbol] = {'status': 'empty'}
            continue
        results[symbol] = {'status': 'ok', 'imbalance': imbalance}
        if imbalance < threshold:
            cache_key = f"order_book_imbalance:{symbol}"
            if not cache.get(cache_key):
                message = f"⚠️ Imbalance carnet {symbol}: bid/ask {imbalance:.2f}"
                stock = Stock.objects.filter(symbol__iexact=symbol).first()
                AlertEvent.objects.create(category='ORDER_BOOK_IMBALANCE', message=message, stock=stock)
                cache.set(cache_key, True, timeout=60 * cooldown)
    return {'threshold': threshold, 'results': results}


@shared_task
def scan_candlestick_patterns(symbols: list[str] | None = None) -> dict[str, Any]:
    """Scan last 5 candles and cache bullish hammer + volume support flags."""
    if symbols is None:
        symbols = []
        for sandbox in ['AI_PENNY', 'AI_BLUECHIP']:
            stored = SandboxWatchlist.objects.filter(sandbox=sandbox).first()
            if stored and stored.symbols:
                symbols.extend([str(s).strip().upper() for s in stored.symbols if str(s).strip()])
    symbols = sorted(set([s for s in symbols if s]))
    results: dict[str, Any] = {}
    for symbol in symbols:
        try:
            engine = DataFusionEngine(symbol)
            df = engine.fuse_all()
            if df is None or df.empty or not {'Open', 'High', 'Low', 'Close'}.issubset(df.columns):
                results[symbol] = {'status': 'no_data'}
                continue
            recent = df.tail(5)
            hammer_flag = False
            patterns_found: list[str] = []
            try:
                import talib

                groups = talib.get_function_groups().get('Pattern Recognition', [])
                for func_name in groups:
                    func = getattr(talib, func_name, None)
                    if func is None:
                        continue
                    out = func(recent['Open'], recent['High'], recent['Low'], recent['Close'])
                    if out is not None and len(out) and float(out.iloc[-1]) != 0:
                        patterns_found.append(func_name)
                hammer_flag = any('HAMMER' in name for name in patterns_found)
            except Exception:
                try:
                    import pandas_ta as ta

                    patterns = ta.cdl_pattern(
                        open_=recent['Open'],
                        high=recent['High'],
                        low=recent['Low'],
                        close=recent['Close'],
                        name=['hammer'],
                    )
                    if patterns is not None and not patterns.empty and 'CDL_HAMMER' in patterns:
                        hammer_flag = float(patterns['CDL_HAMMER'].iloc[-1]) != 0
                        if hammer_flag:
                            patterns_found.append('CDL_HAMMER')
                except Exception:
                    hammer_flag = False

            last = df.tail(1).iloc[0]
            volume_z = float(last.get('VolumeZ') or 0.0)
            ema200 = float(last.get('MA200') or 0.0)
            close = float(last.get('Close') or 0.0)
            near_support = ema200 > 0 and abs(close - ema200) / ema200 <= float(os.getenv('PATTERN_EMA200_BAND', '0.02'))
            hammer_support = bool(hammer_flag and volume_z >= float(os.getenv('PATTERN_VOLUMEZ_MIN', '2.0')) and near_support)

            payload = {
                'status': 'ok',
                'patterns': patterns_found,
                'hammer_support': hammer_support,
                'volume_z': volume_z,
                'ema200': ema200,
                'close': close,
            }
            cache.set(f"pattern_scan:{symbol}", payload, timeout=60 * 60 * 6)
            results[symbol] = payload
        except Exception as exc:
            results[symbol] = {'status': 'error', 'error': str(exc)}
    return {'count': len(results), 'results': results}


@shared_task
def calculate_cross_correlations() -> dict[str, Any]:
    """Compute cross-asset correlations and emit laggard alerts."""
    driver_map = {
        'BTO.TO': 'GC=F',
        'AEM.TO': 'GC=F',
        'K.TO': 'GC=F',
        'CNQ.TO': 'CL=F',
        'SU.TO': 'CL=F',
        'HIVE.V': 'BTC-USD',
        'BITF.TO': 'BTC-USD',
    }
    driver_map.update(_parse_driver_map(os.getenv('CROSS_CORR_DRIVER_MAP')))

    symbols: list[str] = []
    for sandbox in ['AI_PENNY', 'AI_BLUECHIP', 'WATCHLIST']:
        stored = SandboxWatchlist.objects.filter(sandbox=sandbox).first()
        if stored and stored.symbols:
            symbols.extend([str(s).strip().upper() for s in stored.symbols if str(s).strip()])
    symbols = sorted(set([s for s in symbols if s]))

    lookback_days = int(os.getenv('CROSS_CORR_LOOKBACK_DAYS', '90'))
    corr_window = int(os.getenv('CROSS_CORR_WINDOW', '60'))
    corr_threshold = float(os.getenv('CROSS_CORR_ALERT_THRESHOLD', '0.85'))
    driver_move = float(os.getenv('CROSS_CORR_DRIVER_MOVE', '0.01'))
    laggard_move = float(os.getenv('CROSS_CORR_LAGGARD_MOVE', '0.002'))

    results: dict[str, Any] = {}
    for symbol in symbols:
        driver = driver_map.get(symbol) or _default_driver_for_symbol(symbol)
        try:
            sym_df = yf.download(symbol, period=f"{lookback_days}d", interval='1d')
            drv_df = yf.download(driver, period=f"{lookback_days}d", interval='1d')
            if sym_df is None or sym_df.empty or drv_df is None or drv_df.empty:
                results[symbol] = {'status': 'no_data'}
                continue
            sym_close = sym_df['Close'].dropna()
            drv_close = drv_df['Close'].dropna()
            aligned = pd.concat([sym_close.pct_change(), drv_close.pct_change()], axis=1).dropna()
            if aligned.empty:
                results[symbol] = {'status': 'no_data'}
                continue
            corr = float(aligned.iloc[:, 0].rolling(corr_window).corr(aligned.iloc[:, 1]).iloc[-1])
            sym_last = float(sym_close.iloc[-1])
            sym_prev = float(sym_close.iloc[-2]) if len(sym_close) > 1 else sym_last
            drv_last = float(drv_close.iloc[-1])
            drv_prev = float(drv_close.iloc[-2]) if len(drv_close) > 1 else drv_last
            sym_ret = (sym_last - sym_prev) / sym_prev if sym_prev else 0.0
            drv_ret = (drv_last - drv_prev) / drv_prev if drv_prev else 0.0
            payload = {
                'status': 'ok',
                'driver': driver,
                'corr': corr,
                'symbol_return': sym_ret,
                'driver_return': drv_ret,
            }
            results[symbol] = payload
            cache.set(f"cross_corr:{symbol}", payload, timeout=60 * 60 * 12)

            if corr >= corr_threshold and drv_ret >= driver_move and abs(sym_ret) <= laggard_move:
                stock = Stock.objects.filter(symbol__iexact=symbol).first()
                message = (
                    f"{symbol} laggard: corr {corr:.2f} avec {driver}, {driver} +{drv_ret*100:.2f}% "
                    f"vs {symbol} {sym_ret*100:.2f}%"
                )
                AlertEvent.objects.create(category='LAGGARD_ALERT', message=message, stock=stock)
        except Exception as exc:
            results[symbol] = {'status': 'error', 'error': str(exc)}
    return {'count': len(results), 'results': results}


@shared_task
def audit_volume_news_lag(symbol: str) -> dict[str, Any]:
    """Audit whether volume spikes precede news for a symbol."""
    symbol = (symbol or '').strip().upper()
    if not symbol:
        return {'status': 'invalid_symbol'}
    try:
        engine = DataFusionEngine(symbol)
        df = engine.fuse_all()
        if df is None or df.empty or 'VolumeZ' not in df:
            return {'status': 'no_data'}
        window_days = int(os.getenv('VOLUME_NEWS_LAG_DAYS', '30'))
        recent = df.tail(window_days)
        if recent.empty:
            return {'status': 'no_data'}
        vol_idx = recent['VolumeZ'].idxmax()
        vol_time = pd.to_datetime(vol_idx)
        stock = Stock.objects.filter(symbol__iexact=symbol).first()
        if not stock:
            return {'status': 'no_stock'}
        news_qs = StockNews.objects.filter(stock=stock).exclude(published_at__isnull=True)
        if not news_qs.exists():
            return {'status': 'no_news'}
        news_first = news_qs.order_by('published_at').first()
        news_time = news_first.published_at
        if not news_time:
            return {'status': 'no_news_time'}
        lag_hours = (news_time - vol_time).total_seconds() / 3600.0
        payload = {
            'status': 'ok',
            'volume_peak_at': str(vol_time),
            'news_first_at': str(news_time),
            'lag_hours': lag_hours,
        }
        cache.set(f"volume_news_lag:{symbol}", payload, timeout=60 * 60 * 12)
        if lag_hours >= float(os.getenv('VOLUME_NEWS_LAG_ALERT_HOURS', '2')):
            message = f"{symbol} volume spike precedes news by {lag_hours:.1f}h"
            AlertEvent.objects.create(category='VOLUME_NEWS_LAG', message=message, stock=stock)
        return payload
    except Exception as exc:
        return {'status': 'error', 'error': str(exc)}


def _projection_price(symbol: str, entry_price: float) -> float | None:
    hist = yf.Ticker(symbol).history(period='60d', interval='1d')
    close = _extract_close_series(hist)
    if close is None or close.empty:
        return None
    recent_high = float(close.tail(20).max())
    base = float(entry_price) * 1.1 if entry_price else recent_high
    return max(recent_high, base)


def _skip_fundamentals_info(symbol: str) -> bool:
    symbol_upper = (symbol or '').upper()
    return symbol_upper in NON_FUNDAMENTAL_SYMBOLS or _is_crypto_symbol(symbol_upper)


def _send_alert(subject: str, message: str) -> None:
    alert_to = os.getenv('ALERT_EMAIL_TO')
    from_email = os.getenv('DEFAULT_FROM_EMAIL', 'alerts@localhost')
    if alert_to:
        try:
            send_mail(subject, message, from_email, [alert_to], fail_silently=True)
        except Exception:
            pass

    webhook = os.getenv('SLACK_WEBHOOK_URL')
    if webhook:
        try:
            requests.post(webhook, json={'text': f"{subject}\n{message}"}, timeout=10)
        except Exception:
            pass


def _send_telegram_alert(message: str, allow_during_blackout: bool = False, category: str = 'signal') -> None:
    if not allow_during_blackout:
        blocked, event = _is_high_impact_window()
        if blocked:
            cache.set(
                'telegram_signal_blocked',
                {
                    'category': category,
                    'message': message,
                    'event': event,
                    'blocked_at': timezone.now().isoformat(),
                },
                timeout=60 * 60,
            )
            return
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown',
    }
    try:
        requests.post(url, json=payload, timeout=8)
    except Exception:
        pass


def _send_telegram_message(message: str, chat_id: str | None = None) -> None:
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown',
    }
    try:
        requests.post(url, json=payload, timeout=8)
    except Exception:
        pass


def _send_telegram_inline_message(message: str, reply_markup: dict[str, Any], chat_id: str | None = None) -> None:
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'HTML',
        'disable_web_page_preview': True,
        'reply_markup': reply_markup,
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception:
        pass


def _queue_telegram_candidate(payload: dict[str, Any]) -> None:
    queue_key = 'telegram_signal_queue'
    queue = cache.get(queue_key) or []
    queue.append(payload)
    cache.set(queue_key, queue, timeout=60 * 60)


def send_telegram_signal(ticker: str, score: float, diagnostic: str, deepseek_score: float | None = None) -> bool:
    min_score = float(os.getenv('TELEGRAM_MIN_SCORE', '80'))
    if score < min_score:
        _queue_telegram_candidate({
            'ticker': ticker,
            'score': score,
            'diagnostic': diagnostic,
            'deepseek_score': deepseek_score,
            'ts': timezone.now().isoformat(),
        })
        return False

    emoji = "🚀" if score >= 90 else "📈"
    message = (
        f"{emoji} *ALERTE DANAS : {ticker}*\n"
        "━━━━━━━━━━━━━━━\n"
        f"🎯 *Score ML:* {score:.1f}%\n"
        f"🧠 *DeepSeek:* {deepseek_score if deepseek_score is not None else '—'}/10\n"
        f"📝 *Analyse:* {diagnostic}\n"
        "━━━━━━━━━━━━━━━\n"
        f"🔗 [Wealthsimple](https://my.wealthsimple.com/app/invest/search?query={ticker})"
    )
    _send_telegram_alert(message, allow_during_blackout=True, category='signal')
    return True


def should_send_to_telegram(score: float | None, ai_text: str) -> bool:
    min_score = float(os.getenv('TELEGRAM_MIN_SCORE', '80'))
    if score is not None and score < min_score:
        return False
    danger_words = [
        'éviter',
        'trop risqué',
        'pas recommandé',
        'no-go',
        'bearish',
        'prudence',
    ]
    text = (ai_text or '').lower()
    return not any(word in text for word in danger_words)


@shared_task
def telegram_summary_task(limit: int = 3) -> dict[str, Any]:
    queue_key = 'telegram_signal_queue'
    queue = cache.get(queue_key) or []
    if not queue:
        return {'status': 'empty', 'count': 0}

    ranked = sorted(queue, key=lambda item: float(item.get('score') or 0), reverse=True)
    top = ranked[:limit]
    lines = ["🧠 *RÉCAP DANAS (Top Opportunités)*", ""]
    for item in top:
        ticker = item.get('ticker') or '—'
        score = float(item.get('score') or 0)
        lines.append(f"• {ticker} — {score:.1f}%")
    _send_telegram_alert("\n".join(lines), allow_during_blackout=True, category='report')
    cache.delete(queue_key)
    return {'status': 'ok', 'count': len(top)}


def telegram_answer_callback(callback_id: str) -> None:
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token or not callback_id:
        return
    url = f"https://api.telegram.org/bot{token}/answerCallbackQuery"
    payload = {
        'callback_query_id': callback_id,
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception:
        pass


def _alpaca_approval_cache_key(token: str) -> str:
    return f"alpaca_approval:{token}"


def _alpaca_approval_pending_key(sandbox: str, symbol: str) -> str:
    return f"alpaca_approval_pending:{sandbox}:{symbol}"


def _format_pct(value: float | None) -> str:
    if value is None:
        return '—'
    try:
        return f"{float(value) * 100:.1f}%" if float(value) <= 1 else f"{float(value):.1f}%"
    except Exception:
        return '—'


def _queue_alpaca_trade_approval(
    *,
    symbol: str,
    sandbox: str,
    qty: int,
    price: float,
    limit_price: float | None,
    stop_loss: float,
    signal_payload: dict[str, Any] | None,
    tier: str,
    tier_reasons: list[str] | None,
    use_limit_orders: bool,
    notes: str | None,
) -> bool:
    if not symbol or qty <= 0 or price <= 0:
        return False
    pending_key = _alpaca_approval_pending_key(sandbox, symbol)
    if cache.get(pending_key):
        return True
    ttl_min = int(os.getenv('ALPACA_APPROVAL_TTL_MIN', '30'))
    token = uuid4().hex[:8]
    payload = {
        'symbol': symbol,
        'sandbox': sandbox,
        'qty': qty,
        'price': price,
        'limit_price': limit_price,
        'stop_loss': stop_loss,
        'signal_payload': signal_payload or {},
        'tier': tier,
        'tier_reasons': tier_reasons or [],
        'use_limit_orders': use_limit_orders,
        'notes': notes or '',
        'pending_key': pending_key,
    }
    cache.set(_alpaca_approval_cache_key(token), payload, timeout=ttl_min * 60)
    cache.set(pending_key, token, timeout=ttl_min * 60)
    confidence = None
    try:
        confidence = float((signal_payload or {}).get('signal') or 0.0)
    except Exception:
        confidence = None
    order_label = 'LIMIT' if use_limit_orders and limit_price else 'MARKET'
    message_lines = [
        "📝 Validation requise (Alpaca)",
        f"Ticker: {symbol} [{sandbox}]",
        f"Signal: {_format_pct(confidence)}",
        f"Prix: ${price:.2f}",
        f"Quantité: {qty}",
        f"Stop ATR: ${stop_loss:.2f}",
        f"Ordre: {order_label}{f' @ ${limit_price:.2f}' if limit_price else ''}",
        f"Tier: {tier}",
    ]
    reply_markup = {
        'inline_keyboard': [
            [
                {'text': '✅ Approuver', 'callback_data': f"approve:{token}"},
                {'text': '❌ Refuser', 'callback_data': f"reject:{token}"},
            ]
        ]
    }
    _send_telegram_inline_message("\n".join(message_lines), reply_markup)
    return True


def process_alpaca_trade_approval(token: str, approved: bool) -> dict[str, Any]:
    key = _alpaca_approval_cache_key(token)
    payload = cache.get(key)
    if not payload:
        return {'status': 'expired'}
    if payload.get('status') in {'approved', 'rejected'}:
        return {'status': payload.get('status')}
    symbol = payload.get('symbol')
    sandbox = payload.get('sandbox')
    pending_key = payload.get('pending_key')
    if not approved:
        payload['status'] = 'rejected'
        cache.set(key, payload, timeout=60 * 30)
        if pending_key:
            cache.delete(pending_key)
        _decision_log(symbol, sandbox, 'SKIP', 'manual_reject')
        return {'status': 'rejected', 'symbol': symbol}

    qty = int(payload.get('qty') or 0)
    price = float(payload.get('price') or 0)
    limit_price = payload.get('limit_price')
    use_limit_orders = bool(payload.get('use_limit_orders'))
    if qty <= 0 or price <= 0:
        if pending_key:
            cache.delete(pending_key)
        return {'status': 'invalid'}
    if use_limit_orders and limit_price:
        order = submit_limit_order(symbol, qty, 'buy', float(limit_price))
    else:
        order = submit_market_order(symbol, qty, 'buy')
    if order is None:
        if pending_key:
            cache.delete(pending_key)
        payload['status'] = 'order_failed'
        cache.set(key, payload, timeout=60 * 10)
        return {'status': 'order_failed', 'symbol': symbol}

    tier = payload.get('tier') or 'T2'
    tier_reasons = payload.get('tier_reasons') or []
    signal_payload = payload.get('signal_payload') or {}
    stop_loss = float(payload.get('stop_loss') or 0.0)
    notes = payload.get('notes') or ''
    PaperTrade.objects.create(
        ticker=symbol,
        sandbox=sandbox,
        entry_price=round(price, 2),
        quantity=qty,
        entry_signal=signal_payload.get('signal'),
        entry_features={
            **(signal_payload.get('features') or {}),
            'tier': tier,
            'tier_reasons': tier_reasons,
            'alpaca_high_water': price,
        },
        entry_explanations=signal_payload.get('explanations'),
        model_name=signal_payload.get('model_name', 'BLUECHIP'),
        model_version=signal_payload.get('model_version', ''),
        broker='ALPACA',
        broker_order_id=str(getattr(order, 'id', '') or ''),
        broker_status=str(getattr(order, 'status', '') or ''),
        broker_side='BUY',
        broker_updated_at=timezone.now(),
        stop_loss=round(stop_loss, 2) if stop_loss > 0 else None,
        status='OPEN',
        pnl=0,
        notes=f"manual approval | {notes}".strip(),
    )
    payload['status'] = 'approved'
    cache.set(key, payload, timeout=60 * 60)
    if pending_key:
        cache.delete(pending_key)
    _decision_log(symbol, sandbox, 'BUY', 'manual_approval')
    return {'status': 'approved', 'symbol': symbol, 'qty': qty}


def _market_sentiment_score() -> float | None:
    sentiment, meta = get_market_sentiment()
    spy_change = meta.get('spy_change')
    tsx_change = meta.get('tsx_change')
    values = [v for v in [spy_change, tsx_change] if v is not None]
    if values:
        return float(sum(values) / len(values))
    if sentiment == 'BEARISH':
        return -1.0
    if sentiment == 'BULLISH':
        return 1.0
    return None


_MONTHS_FR = {
    1: 'JANVIER',
    2: 'FÉVRIER',
    3: 'MARS',
    4: 'AVRIL',
    5: 'MAI',
    6: 'JUIN',
    7: 'JUILLET',
    8: 'AOÛT',
    9: 'SEPTEMBRE',
    10: 'OCTOBRE',
    11: 'NOVEMBRE',
    12: 'DÉCEMBRE',
}


def _format_french_date(date_value: date) -> str:
    month = _MONTHS_FR.get(date_value.month, str(date_value.month))
    return f"{date_value.day} {month} {date_value.year}"


def _journal_task_logs(as_of_date: date, limit: int = 6) -> list[str]:
    logs = list(
        TaskRunLog.objects.filter(started_at__date=as_of_date)
        .order_by('started_at')[:limit]
    )
    if not logs:
        return ["— Aucun log notable"]
    lines: list[str] = []
    for entry in logs:
        ts = timezone.localtime(entry.started_at).strftime('%H:%M')
        status_icon = '✅' if entry.status == 'SUCCESS' else '⚠️'
        name = entry.task_name.replace('_', ' ').replace(':', ' ').strip()
        lines.append(f"{ts} | {status_icon} {name}")
    return lines


def _journal_scanner_sections() -> tuple[str, list[str]]:
    results = cache.get('market_scanner_results') or []
    if not results:
        results = cache.get('market_scanner_afterhours_results') or []
    total = len(results)
    intro = (
        f"Le bot a scanné {total} tickers aujourd'hui via ses deux pipelines."
        if total else
        "Le bot n'a pas trouvé de candidats aujourd'hui."
    )
    penny_max = float(os.getenv('AI_PENNY_MAX_PRICE', '15'))
    penny = [r for r in results if float(r.get('price') or 0) <= penny_max]
    bluechip = [r for r in results if float(r.get('price') or 0) > penny_max]

    def _format_target(item: dict[str, Any]) -> list[str]:
        symbol = item.get('symbol') or ''
        score = float(item.get('score') or 0.0)
        rvol = float(item.get('rvol') or 0.0)
        patterns = ', '.join(item.get('patterns') or []) or 'Pattern technique'
        lines = [f"Target Détectée : ${symbol}"]
        lines.append(f"Score IA : {score:.2f} (RVOL {rvol:.2f} · {patterns})")
        lines.append("Action : Signal envoyé au Paper Trading.")
        return lines

    output: list[str] = []
    output.append("1. Pipeline Penny (Stocks $1-$15)")
    if penny:
        output.extend(_format_target(penny[0]))
    else:
        output.append("Aucune target détectée.")

    output.append("")
    output.append("2. Pipeline Bluechip (Grosses Caps)")
    if bluechip:
        output.extend(_format_target(bluechip[0]))
    else:
        output.append("Aucune target détectée.")

    return intro, output


def _journal_paper_trades(as_of_date: date) -> tuple[list[str], list[str], float]:
    closed = list(
        PaperTrade.objects.filter(
            broker='ALPACA',
            status='CLOSED',
            exit_date__date=as_of_date,
        ).order_by('exit_date')
    )
    open_trades = list(
        PaperTrade.objects.filter(
            broker='ALPACA',
            status='OPEN',
        ).order_by('entry_date')
    )

    closed_lines: list[str] = []
    closed_pnl = 0.0
    for trade in closed:
        entry_price = float(trade.entry_price or 0)
        exit_price = float(trade.exit_price or 0)
        pnl = float(trade.pnl or 0)
        closed_pnl += pnl
        closed_lines.append(
            f"{trade.ticker} : Entrée ${entry_price:.2f} ➔ Sortie ${exit_price:.2f} ({pnl:+.2f})"
        )

    open_lines: list[str] = []
    open_pnl = 0.0
    for trade in open_trades:
        entry_price = float(trade.entry_price or 0)
        current_price = get_latest_trade_price(trade.ticker)
        if current_price is not None:
            current_price = float(current_price)
        else:
            current_price = entry_price
        qty = float(trade.quantity or 0)
        pnl = (current_price - entry_price) * qty
        open_pnl += pnl
        pct = ((current_price - entry_price) / entry_price * 100) if entry_price else 0.0
        status_icon = '🟢' if pct > 0.5 else '🔴' if pct < -0.5 else '⚪'
        open_lines.append(
            f"{trade.ticker} : ${current_price:.2f} ({pct:+.1f}%) {status_icon}"
        )

    return closed_lines, open_lines, (closed_pnl + open_pnl)


def _journal_portfolio_summary() -> tuple[str, str, str]:
    total_capital = Portfolio.objects.aggregate(total=models.Sum('capital')).get('total')
    holdings_count = PortfolioHolding.objects.count()
    capital_text = f"{float(total_capital):.2f}$ CAD" if total_capital else "n/a"
    positions_text = f"{holdings_count} position(s)" if holdings_count else "Aucune"
    status_text = "Prêt pour l'ouverture de demain." if holdings_count == 0 else "Surveillance active."
    return capital_text, positions_text, status_text


def _afterhours_scanner_symbols() -> list[str]:
    scr_ids = [
        s.strip()
        for s in os.getenv('AFTERHOURS_SCREENERS', 'day_gainers,most_actives').split(',')
        if s.strip()
    ]
    limit = int(os.getenv('AFTERHOURS_SYMBOL_LIMIT', '120'))
    symbols: list[str] = []
    try:
        quotes = _fetch_yfinance_screeners(scr_ids, count=limit)
        symbols = [
            str(item.get('symbol') or '').strip().upper()
            for item in quotes
            if item.get('symbol')
        ]
    except Exception:
        symbols = []
    if not symbols:
        symbols = _default_scanner_symbols()
    swing = list(MasterWatchlistEntry.objects.filter(category='SWING').values_list('symbol', flat=True))
    if swing:
        symbols.extend([str(s).strip().upper() for s in swing if str(s).strip()])
    symbols = [s for s in dict.fromkeys(symbols) if s]
    return symbols[:limit]


def _afterhours_market_scan(symbols: list[str] | None = None) -> dict[str, Any]:
    min_price = float(os.getenv('SCANNER_MIN_PRICE', '0.5'))
    max_price = float(os.getenv('SCANNER_MAX_PRICE', '10'))
    min_volume = float(os.getenv('SCANNER_MIN_VOLUME', '500000'))
    min_change = float(os.getenv('SCANNER_MIN_CHANGE_PCT', '2'))
    min_rvol = float(os.getenv('AFTERHOURS_DAILY_RVOL_MIN', '1.5'))
    min_confidence = float(os.getenv('SCANNER_MIN_CONFIDENCE', '65'))
    update_watchlist = os.getenv('AI_SCANNER_UPDATE_WATCHLIST', 'true').lower() in {'1', 'true', 'yes', 'y'}
    notify = os.getenv('AI_SCANNER_TELEGRAM', 'false').lower() in {'1', 'true', 'yes', 'y'}

    if not symbols:
        symbols = _afterhours_scanner_symbols()
    if not symbols:
        return {'status': 'empty', 'count': 0, 'results': [], 'mode': 'afterhours'}

    results: list[dict[str, Any]] = []
    for symbol in symbols:
        try:
            hist = yf.Ticker(symbol).history(period='20d', interval='1d', timeout=10)
            if hist is None or hist.empty or 'Close' not in hist:
                continue
            close = hist['Close']
            volume = hist['Volume'] if 'Volume' in hist else None
            if close is None or close.empty or len(close) < 2:
                continue
            last_close = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])
            if last_close <= 0 or prev_close <= 0:
                continue
            if not (min_price <= last_close <= max_price):
                continue
            change_pct = ((last_close - prev_close) / prev_close) * 100
            if change_pct < min_change:
                continue
            if volume is None or volume.empty:
                continue
            last_vol = float(volume.iloc[-1])
            avg_vol = float(volume.tail(20).mean()) if len(volume) >= 5 else float(volume.mean())
            if last_vol < min_volume or avg_vol <= 0:
                continue
            rvol = last_vol / avg_vol
            if rvol < min_rvol:
                continue



            base_score = 0.5 + (min(10.0, abs(change_pct)) / 100) + (min(5.0, rvol) / 50)
            confidence_pct = max(0.0, min(95.0, base_score * 100))
            if confidence_pct < min_confidence:
                continue

            results.append({
                'symbol': symbol,
                'price': round(last_close, 4),
                'change_pct': round(change_pct, 2),
                'volume': int(last_vol),
                'rvol': round(rvol, 2),
                'score': round(confidence_pct / 100, 4),
                'confidence_pct': round(confidence_pct, 2),
            })
        except Exception:
            continue



    results = sorted(results, key=lambda x: x['score'], reverse=True)
    if results:
        cache.set('market_scanner_afterhours_results', results, timeout=60 * 60 * 24)

    if update_watchlist and results:
        SandboxWatchlist.objects.update_or_create(
            sandbox='AI_PENNY',
            defaults={'symbols': [entry['symbol'] for entry in results[:25]]},
        )
        if os.getenv('AI_SCANNER_UPDATE_WATCHLIST_MAIN', 'false').lower() in {'1', 'true', 'yes', 'y'}:
            main_limit = int(os.getenv('AI_SCANNER_MAIN_LIMIT', '15'))


            SandboxWatchlist.objects.update_or_create(
                sandbox='WATCHLIST',


                defaults={'symbols': [entry['symbol'] for entry in results[:max(1, main_limit)]]},
            )



    if notify and results:
        top = results[0]
        _send_telegram_alert(
            f"📌 Pré-sélection fermeture: {top['symbol']} score {top['score']:.2f} "
            f"RVOL {top['rvol']} Δ {top['change_pct']:+.2f}%",
            allow_during_blackout=True,


            category='report',
        )

    return {'status': 'ok', 'count': len(results), 'results': results, 'mode': 'afterhours'}


def _ny_time_now() -> datetime:
    try:
        return timezone.now().astimezone(ZoneInfo('America/New_York'))
    except Exception:
        return timezone.now()


def _entry_time_features(symbol: str, now: datetime | None = None) -> dict[str, Any]:
    now = now or _ny_time_now()
    hour = now.hour
    weekday = now.weekday()
    minutes_since_open = max(0, (now.hour * 60 + now.minute) - (9 * 60 + 30))
    minutes_to_close = max(0, (16 * 60) - (now.hour * 60 + now.minute))
    is_power_hour = 1 if now.hour == 15 else 0
    is_lunch = 1 if (now.hour == 11 and now.minute >= 30) or (now.hour == 12) or (now.hour == 13 and now.minute < 30) else 0
    earnings_in_days = None
    try:
        ticker = yf.Ticker(symbol)
        earnings_date = _get_earnings_date(ticker)
        if earnings_date:
            earnings_in_days = (earnings_date - now.date()).days
    except Exception:
        earnings_in_days = None
    return {
        'entry_hour': hour,
        'entry_weekday': weekday,
        'minutes_since_open': minutes_since_open,
        'minutes_to_close': minutes_to_close,
        'is_power_hour': is_power_hour,
        'is_lunch': is_lunch,
        'earnings_in_days': earnings_in_days if earnings_in_days is not None else 0,
    }


def _price_move_after_entry(symbol: str, entry_time: datetime, hours: int) -> tuple[float | None, float | None]:
    start = entry_time.astimezone(dt_timezone.utc)
    end = start + timedelta(hours=hours)
    df = get_intraday_bars_range(symbol, start=start, end=end)
    if df is None or df.empty or 'close' not in df.columns:
        try:
            hist = yf.download(symbol, period='5d', interval='1m')
            df = hist.rename(columns={'Close': 'close'}) if hist is not None else None
        except Exception:
            df = None
    if df is None or df.empty or 'close' not in df.columns:
        return None, None
    series = df['close'].dropna()
    if series.empty:
        return None, None
    entry_price = float(series.iloc[0]) if float(series.iloc[0]) else None
    if not entry_price:
        return None, None
    max_move = (float(series.max()) - entry_price) / entry_price
    min_move = (float(series.min()) - entry_price) / entry_price
    return max_move, min_move




def _risk_manager_allocation(
    confidence_score: float,
    sentiment_score: float | None,
    price: float | None,
    atr: float | None,
) -> float:
    if confidence_score < 60:
        allocation = float(os.getenv('RISK_ALLOC_MIN', '50'))
    elif confidence_score < 75:
        allocation = float(os.getenv('RISK_ALLOC_MIN', '50'))
    elif confidence_score < 90:
        allocation = float(os.getenv('RISK_ALLOC_STD', '150'))
    else:
        allocation = float(os.getenv('RISK_ALLOC_MAX', '250'))

    if sentiment_score is not None:
        sentiment_score = max(0.0, min(1.0, sentiment_score))


        allocation *= max(0.6, min(1.2, 1 + ((sentiment_score - 0.5) * 0.4)))

    if price and atr:
        atr_pct = atr / price if price else 0.0
        if atr_pct >= float(os.getenv('RISK_ALLOC_VOL_ATR_PCT', '0.06')):
            allocation *= 0.75

    return round(float(allocation), 2)


def _dynamic_position_target(equity: float) -> float:
    try:
        pct = float(os.getenv('DYNAMIC_POSITION_PCT', '0.015'))
        min_value = float(os.getenv('DYNAMIC_POSITION_MIN', '50'))
        max_value = float(os.getenv('DYNAMIC_POSITION_MAX', '3000'))
    except Exception:
        pct = 0.015
        min_value = 50.0
        max_value = 3000.0
    if equity <= 0:
        return 0.0
    target = equity * pct
    return max(min_value, min(max_value, float(target)))


def _time_hhmm(dt_value: datetime | None = None) -> str:
    dt_value = dt_value or _ny_time_now()


    return dt_value.strftime('%H:%M')


def _market_closed_now() -> bool:
    now = _ny_time_now()
    if now.weekday() >= 5:
        return True
    return not (dt_time(9, 30) <= now.time() <= dt_time(16, 0))


def _collect_training_symbols() -> dict[str, list[str]]:
    penny_symbols: set[str] = set()
    bluechip_symbols: set[str] = set()

    watch_penny = SandboxWatchlist.objects.filter(sandbox='AI_PENNY').first()
    if watch_penny and watch_penny.symbols:
        penny_symbols.update({str(s).strip().upper() for s in watch_penny.symbols if str(s).strip()})

    watch_blue = SandboxWatchlist.objects.filter(sandbox='AI_BLUECHIP').first()
    if watch_blue and watch_blue.symbols:
        bluechip_symbols.update({str(s).strip().upper() for s in watch_blue.symbols if str(s).strip()})

    holdings = PortfolioHolding.objects.select_related('stock').all()
    for holding in holdings:
        symbol = (holding.stock.symbol or '').strip().upper()
        if symbol:
            bluechip_symbols.add(symbol)

    tfsa_accounts = Account.objects.filter(account_type='TFSA')
    if tfsa_accounts.exists():
        txs = (
            AccountTransaction.objects.filter(account__in=tfsa_accounts)
            .select_related('stock')
            .order_by('date')
        )
        positions: dict[int, float] = {}
        for tx in txs:
            if tx.type == 'DIVIDEND':
                continue
            qty = float(tx.quantity or 0)
            if qty <= 0:
                continue
            sign = 1 if tx.type == 'BUY' else -1
            positions[tx.stock_id] = positions.get(tx.stock_id, 0.0) + (sign * qty)
        for stock_id, shares in positions.items():
            if shares <= 0:
                continue
            stock = Stock.objects.filter(id=stock_id).first()
            if not stock:
                continue
            symbol = (stock.symbol or '').strip().upper()
            if symbol:
                bluechip_symbols.add(symbol)

    return {
        'PENNY': sorted(penny_symbols),
        'BLUECHIP': sorted(bluechip_symbols),
    }


def _profit_wallet_path() -> Path:
    return Path(settings.BASE_DIR) / 'profit_wallet.json'


def _week_start_date(now: datetime) -> date:
    ny_now = now.astimezone(ZoneInfo('America/New_York'))
    start = ny_now.date() - timedelta(days=ny_now.weekday())
    return start


def _load_profit_wallet() -> dict[str, Any]:
    path = _profit_wallet_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_profit_wallet(payload: dict[str, Any]) -> None:
    path = _profit_wallet_path()
    try:
        path.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass


def _spy_premarket_snapshot() -> tuple[float | None, float | None]:
    try:
        hist = yf.Ticker('SPY').history(period='2d', interval='1m', prepost=True)
        if hist is None or hist.empty or 'Close' not in hist:
            return None, None
        last_price = float(hist['Close'].iloc[-1])
        daily = yf.Ticker('SPY').history(period='3d', interval='1d')
        if daily is None or daily.empty or 'Close' not in daily:
            return last_price, None
        prev_close = float(daily['Close'].iloc[-2]) if len(daily) >= 2 else float(daily['Close'].iloc[-1])
        if prev_close <= 0:
            return last_price, None
        change_pct = ((last_price - prev_close) / prev_close) * 100
        return last_price, change_pct
    except Exception:
        return None, None


def _top_watchlist_symbols(limit: int = 3) -> list[str]:
    research = cache.get('weekend_deep_research') or {}
    penny = (research.get('penny') or {}).get('watchlist') or []
    if penny:
        return [str(item.get('symbol') or '').strip().upper() for item in penny[:limit] if item.get('symbol')]
    watch = SandboxWatchlist.objects.filter(sandbox='AI_PENNY').first()
    if watch and watch.symbols:
        return [s for s in watch.symbols[:limit] if s]
    return []


ECONOMIC_CALENDAR_CACHE_KEY = 'economic_calendar_week'


def _parse_economic_event_datetime(event: dict[str, Any]) -> datetime | None:
    raw_date = str(event.get('date') or '').strip()
    raw_time = str(event.get('time') or '').strip()
    if raw_time.lower().startswith('all'):
        return None
    if raw_date and ':' in raw_date:
        dt_str = raw_date
    elif raw_date and raw_time:
        dt_str = f"{raw_date} {raw_time}"
    else:
        dt_str = raw_date
    if not dt_str:
        return None
    try:
        parsed = pd.to_datetime(dt_str, utc=True, errors='coerce')
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime()
    except Exception:
        return None


def _economic_calendar_events() -> list[dict[str, Any]]:
    return cache.get(ECONOMIC_CALENDAR_CACHE_KEY, []) or []


def _is_high_impact_event(event: dict[str, Any]) -> bool:
    impact = str(event.get('impact') or '').strip().lower()
    return impact in {'high', 'high impact', 'critique', 'critical', 'red'}


def _is_high_impact_window(window_minutes: int = 30) -> tuple[bool, dict[str, Any] | None]:
    now = _ny_time_now()
    events = _economic_calendar_events()
    if not events:
        return False, None
    window = timedelta(minutes=window_minutes)
    for event in events:
        if not _is_high_impact_event(event):
            continue
        event_time_str = event.get('datetime_ny') or event.get('datetime_utc')
        try:
            event_time = pd.to_datetime(event_time_str, utc=True, errors='coerce')
            if pd.isna(event_time):
                continue
            event_time = event_time.to_pydatetime().astimezone(ZoneInfo('America/New_York'))
        except Exception:
            continue
        if event_time - window <= now <= event_time + window:
            return True, event
    return False, None


def _economic_calendar_note_for_today() -> str | None:
    today = _ny_time_now().date()
    events = _economic_calendar_events()
    if not events:
        return None
    upcoming = []
    for event in events:
        if not _is_high_impact_event(event):
            continue
        try:
            event_time = pd.to_datetime(event.get('datetime_ny'), utc=True, errors='coerce')
            if pd.isna(event_time):
                continue
            event_time = event_time.to_pydatetime().astimezone(ZoneInfo('America/New_York'))
        except Exception:
            continue
        if event_time.date() == today:
            upcoming.append(f"{event_time.strftime('%H:%M')} {event.get('title')}")
    if not upcoming:
        return None
    return "⚠️ Attention : " + ", ".join(upcoming) + ". L'IA sera en mode observation autour de ces annonces."


def _bluechip_aggressive_multiplier() -> float:
    try:
        return float(cache.get('bluechip_aggressive_multiplier', 1.0) or 1.0)
    except (TypeError, ValueError):
        return 1.0


def _calc_stop_target_from_prev_candle(bars: pd.DataFrame, current_price: float) -> tuple[float, float, float]:
    if bars is None or bars.empty or len(bars) < 2:
        stop_loss = max(current_price * 0.98, current_price - 0.05)
        risk = max(current_price - stop_loss, 0.01)
        target_price = current_price + (2 * risk)
        return stop_loss, target_price, risk
    prev = bars.iloc[-2]
    prev_low = float(prev.get('low') or prev.get('l') or 0.0)
    buffer = max(0.01, prev_low * 0.001)
    stop_loss = max(prev_low - buffer, current_price * 0.5)
    risk = max(current_price - stop_loss, 0.01)
    target_price = current_price + (2 * risk)
    return stop_loss, target_price, risk


def _calc_spread_and_liquidity(symbol: str, bars: pd.DataFrame | None) -> tuple[float | None, float | None]:
    snapshots = get_stock_snapshots([symbol])
    snap = (snapshots or {}).get(symbol)
    bid = None
    ask = None
    if snap is not None:
        latest_quote = getattr(snap, 'latest_quote', None) or getattr(snap, 'latestQuote', None)
        if latest_quote is not None:
            bid = float(getattr(latest_quote, 'bid_price', 0) or 0)
            ask = float(getattr(latest_quote, 'ask_price', 0) or 0)
    spread_pct = None
    if bid and ask and bid > 0:
        spread_pct = ((ask - bid) / bid) * 100

    avg_vol = None
    if bars is not None and not bars.empty and 'volume' in bars:
        recent = bars.tail(3)
        avg_vol = float(recent['volume'].mean()) if not recent.empty else None
    return spread_pct, avg_vol


def _sector_etf_for_stock(symbol: str) -> str | None:
    stock = Stock.objects.filter(symbol__iexact=symbol).first()
    if not stock or not stock.sector:
        return None
    sector = stock.sector.lower()
    mapping = {
        'biotech': 'XBI',
        'health': 'XLV',
        'tech': 'XLK',
        'financial': 'XLF',
        'energy': 'XLE',
        'industrial': 'XLI',
        'consumer defensive': 'XLP',
        'consumer cyclical': 'XLY',
        'materials': 'XLB',
        'utilities': 'XLU',
        'real estate': 'XLRE',
        'communication': 'XLC',
    }
    for key, etf in mapping.items():
        if key in sector:
            return etf
    return None


def _sector_confirmation_ok(symbol: str) -> bool:
    etf = _sector_etf_for_stock(symbol)
    if not etf:
        return True
    bars = get_intraday_bars(etf, minutes=90)
    if bars is None or bars.empty:
        return True
    closes = bars['close'] if 'close' in bars else bars.get('c')
    if closes is None or closes.empty:
        return True
    ema9 = closes.ewm(span=9, adjust=False).mean().iloc[-1]
    latest = float(closes.iloc[-1])
    return latest >= float(ema9)


def _poc_from_bars(bars: pd.DataFrame) -> float | None:
    if bars is None or bars.empty:
        return None
    closes = bars['close'] if 'close' in bars else None
    volumes = bars['volume'] if 'volume' in bars else None
    if closes is None or volumes is None:
        return None
    last_price = float(closes.iloc[-1]) if not closes.empty else 0.0
    if last_price <= 0:
        return None
    if last_price < 1:
        step = 0.001
    elif last_price < 10:
        step = 0.01
    elif last_price < 100:
        step = 0.05
    else:
        step = 0.1
    bins = (closes / step).round() * step
    volume_by_price = volumes.groupby(bins).sum()
    if volume_by_price.empty:
        return None
    return float(volume_by_price.idxmax())


def _resample_ohlcv(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame()
    df = bars.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df.dropna(subset=['timestamp']).set_index('timestamp')
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    for col in agg.keys():
        if col not in df.columns:
            return pd.DataFrame()
    resampled = df.resample(f'{minutes}min').agg(agg).dropna()
    resampled = resampled.reset_index().rename(columns={'index': 'timestamp'})
    return resampled


def _intraday_context_for_timeframe(symbol: str, minutes: int, timeframe: int, rvol_window: int = 20) -> dict[str, Any] | None:
    bars = get_intraday_bars(symbol, minutes=minutes)
    if bars is None or bars.empty:
        return None
    resampled = _resample_ohlcv(bars, timeframe)
    if resampled.empty:
        return None
    enriched = enrich_bars_with_patterns(resampled, rvol_window=rvol_window)
    if enriched.empty:
        return None
    last = enriched.iloc[-1]
    return {
        'bars': enriched,
        'rvol': float(last.get('rvol') or 0.0),
        'pattern_signal': float(last.get('pattern_signal') or 0.0),
        'patterns': last.get('patterns') or [],
        'volatility': float(last.get('volatility') or 0.0),
        'last_close': float(last.get('close') or 0.0),
    }


def _build_swing_signal_message(
    ticker: str,
    confidence_pct: float,
    entry_price: float,
    buy_limit: float,
    target_price: float,
    stop_price: float,
    investment: float,
    news_title: str | None,
    news_source: str | None,
    news_note: str | None,
) -> str:
    quantity = investment / buy_limit if buy_limit else 0.0
    target_gain = investment * ((target_price - buy_limit) / buy_limit) if buy_limit else 0.0
    currency = _symbol_currency(ticker)
    lines = [
        f"🚀 OPPORTUNITÉ SWING : ${ticker}",
        f"🔥 Confiance : {confidence_pct:.1f}%",
        f"📥 Prix Entrée : {entry_price:.2f} {currency} (WealthSimple Limit Order)",
        "🎯 ACTION À PRENDRE (Wealthsimple)",
        f"🔹 Prix Limite : {buy_limit:.2f} {currency}",
        f"🔹 Quantité : {quantity:.2f} actions",
        f"💰 Objectif : {target_price:.2f} {currency} (+{target_gain:.2f}$ pour {investment:.0f}$ investi)",
        f"🛡️ Protection : {stop_price:.2f} {currency} (Stop-Loss)",
    ]
    if news_title:
        source_txt = f"Source : {news_source}" if news_source else "Source : Google News"
        lines.append(f"📰 NEWS RÉCENTE : {news_title}")
        lines.append(source_txt)
    if news_note:
        lines.append(f"{news_note}")
    return "\n".join(lines)


def _sentiment_label(score: float | None) -> str:
    if score is None:
        return 'Neutre'
    if score >= 0.5:
        return 'Très Positif'
    if score >= 0.2:
        return 'Positif'
    if score <= -0.5:
        return 'Très Négatif'
    if score <= -0.2:
        return 'Négatif'
    return 'Neutre'


def _rsi_label(rsi: float | None) -> str:
    if rsi is None:
        return 'n/a'
    if rsi <= 30:
        return 'Survendu'
    if rsi >= 70:
        return 'Suracheté'
    return 'Neutre'


def _tsx_driver_for_symbol(symbol: str) -> str:
    base_map = {
        'BTO.TO': 'GC=F',
        'AEM.TO': 'GC=F',
        'K.TO': 'GC=F',
        'CNQ.TO': 'CL=F',
        'SU.TO': 'CL=F',
    }
    base_map.update(_parse_driver_map(os.getenv('CROSS_CORR_DRIVER_MAP')))
    symbol_upper = (symbol or '').strip().upper()
    return base_map.get(symbol_upper, _default_driver_for_symbol(symbol_upper))


def _tsx_driver_correlation(symbol: str) -> tuple[float | None, str | None]:
    symbol_upper = (symbol or '').strip().upper()
    if not symbol_upper:
        return None, None
    cache_key = f"cross_corr:{symbol_upper}"
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and cached.get('status') == 'ok':
        return float(cached.get('corr') or 0.0), cached.get('driver')
    driver = _tsx_driver_for_symbol(symbol_upper)
    try:
        sym_df = yf.download(symbol_upper, period="90d", interval='1d')
        drv_df = yf.download(driver, period="90d", interval='1d')
        if sym_df is None or sym_df.empty or drv_df is None or drv_df.empty:
            return None, driver
        sym_close = sym_df['Close'].dropna()
        drv_close = drv_df['Close'].dropna()
        aligned = pd.concat([sym_close.pct_change(), drv_close.pct_change()], axis=1).dropna()
        if aligned.empty:
            return None, driver
        corr = float(aligned.iloc[:, 0].rolling(60).corr(aligned.iloc[:, 1]).iloc[-1])
        cache.set(cache_key, {'status': 'ok', 'driver': driver, 'corr': corr}, timeout=60 * 60 * 6)
        return corr, driver
    except Exception:
        return None, driver


def _tsx_move_type(rvol: float | None, volatility: float | None, sentiment: float | None) -> str:
    rvol_val = float(rvol or 0.0)
    vol_val = float(volatility or 0.0)
    sent_val = float(sentiment or 0.0)
    if rvol_val >= 4 or vol_val >= 0.05:
        return 'DAY TRADE'
    if sent_val >= 0.3:
        return '3-DAY HOLD'
    return 'BIG SWING'


def _tsx_strategy_text(patterns: list[str], rvol: float | None, rsi: float | None, sentiment: float | None) -> str:
    notes: list[str] = []
    if rsi is not None and rsi <= 35:
        notes.append('Gros Dip détecté avec RSI survendu.')
    if rvol is not None and float(rvol) >= 3:
        notes.append('Volume 3x supérieur à la moyenne (RVOL).')
    if patterns:
        nice = ', '.join(patterns[:2])
        notes.append(f"Pattern haussier: {nice}.")
    if sentiment is not None and float(sentiment) >= 0.3:
        notes.append('Sentiment news positif.')
    return ' '.join(notes).strip() or 'Setup technique confirmé sur support.'


def _build_tsx_signal_message(
    ticker: str,
    move_type: str,
    strategy: str,
    buy_limit: float,
    target_price: float,
    stop_price: float,
    rsi: float | None,
    sentiment: float | None,
    corr: float | None,
    driver: str | None,
) -> str:
    currency = _symbol_currency(ticker)
    profit_pct = ((target_price - buy_limit) / buy_limit) * 100 if buy_limit else 0.0
    corr_txt = f"{corr:.2f}" if corr is not None else 'n/a'
    driver_txt = driver or 'n/a'
    rsi_txt = f"{rsi:.0f}" if rsi is not None else 'n/a'
    duration = 'session (intraday)' if move_type == 'DAY TRADE' else '3 jours' if move_type == '3-DAY HOLD' else '5-7 jours'
    lines = [
        f"🚀 SIGNAL DE TRADING : [{ticker}]",
        f"💡 TYPE DE MOVE : {move_type}",
        f"🎯 STRATÉGIE : {strategy}",
        "",
        "📊 ENTRÉE & SORTIE :",
        f"Achat (In) : < {buy_limit:.2f} {currency}",
        f"Cible (Out) : {target_price:.2f} {currency} (Profit +{profit_pct:.1f}%)",
        f"Stop Loss : {stop_price:.2f} {currency}",
        "",
        "🧠 ANALYSE IA :",
        f"RSI : {rsi_txt} ({_rsi_label(rsi)})",
        f"Sentiment : {_sentiment_label(sentiment)}",
        f"Corrélation {driver_txt} : {corr_txt}",
        f"⏱️ DURÉE PRÉVUE : {duration} ou clôture si cible atteinte.",
    ]
    return "\n".join(lines)


def _build_tsx_manual_trade_alert(
    ticker: str,
    action_label: str,
    confidence: float | None,
    confidence_threshold: float,
    sentiment: float | None,
    sentiment_threshold: float,
    corr: float | None,
    driver: str | None,
    price: float | None,
    entry_low: float | None,
    entry_high: float | None,
    target_price: float | None,
    stop_price: float | None,
) -> str:
    conf_val = float(confidence or 0.0)
    sent_val = float(sentiment or 0.0)
    conf_mark = '✅' if conf_val >= confidence_threshold else '❌'
    sent_mark = '✅' if sent_val >= sentiment_threshold else '❌'
    corr_txt = f"{corr:+.2f}" if corr is not None else 'n/a'
    driver_txt = driver or 'n/a'
    price_txt = f"{price:.2f}$" if price is not None else 'n/a'
    entry_txt = (
        f"{entry_low:.2f}$ - {entry_high:.2f}$" if entry_low is not None and entry_high is not None else 'n/a'
    )
    target_txt = f"{target_price:.2f}$" if target_price is not None else 'n/a'
    stop_txt = f"{stop_price:.2f}$" if stop_price is not None else 'n/a'
    lines = [
        "🇨🇦 ALERTE TRADE MANUEL (TSX)",
        f"Symbole : {ticker}",
        f"Action suggérée : {action_label}",
        "",
        "🧠 Validation IA :",
        f"Confiance ML : {conf_val:.2f} (Seuil {confidence_threshold:.2f} {conf_mark})",
        f"Sentiment Gemini : {sent_val:.2f} ({_sentiment_label(sentiment)} {sent_mark})",
        f"Corrélation {driver_txt} : {corr_txt}",
        "",
        "📈 Niveaux d'exécution :",
        f"Prix actuel : ~{price_txt}",
        f"Entrée idéale (Dip) : {entry_txt}",
        f"Cible de sortie : {target_txt}",
        f"Stop-Loss : {stop_txt}",
        "",
        "🔍 Observation Live :",
        "Le bot surveille les bougies 5min. Je t'enverrai une mise à jour si le RSI dépasse 70 "
        "ou si le carnet d'ordres montre une sortie massive des acheteurs.",
    ]
    return "\n".join(lines)


def _watchlist_symbols(sandboxes: list[str] | None = None, limit: int | None = None) -> list[str]:
    sandboxes = sandboxes or ['WATCHLIST']
    symbols: list[str] = []
    for sandbox in sandboxes:
        stored = SandboxWatchlist.objects.filter(sandbox=sandbox).first()
        if stored and stored.symbols:
            symbols.extend([str(s).strip().upper() for s in stored.symbols if str(s).strip()])
    symbols = sorted(set([s for s in symbols if s]))
    if limit:
        return symbols[:limit]
    return symbols


def _percent_change(symbol: str) -> float | None:
    try:
        hist = yf.Ticker(symbol).history(period='2d', interval='1d', timeout=10)
        if hist is None or hist.empty or 'Close' not in hist or len(hist) < 2:
            return None
        prev = _safe_float(hist['Close'].iloc[-2])
        last = _safe_float(hist['Close'].iloc[-1])
        if not prev or not last:
            return None
        return ((last - prev) / prev) * 100
    except Exception:
        return None


def _latest_price_with_status(symbol: str) -> tuple[float | None, str]:
    price = _latest_price_snapshot(symbol)
    status = 'offline'
    try:
        hist = yf.Ticker(symbol).history(period='5d', interval='1h', timeout=10)
        if hist is not None and not hist.empty and 'Close' in hist:
            last_ts = hist.index[-1]
            if last_ts is not None:
                if hasattr(last_ts, 'tzinfo') and last_ts.tzinfo:
                    last_seen = last_ts.tz_convert('UTC')
                else:
                    last_seen = timezone.make_aware(last_ts, dt_timezone.utc)
                now = timezone.now()
                if now - last_seen <= timedelta(hours=2):
                    status = 'live'
            if price is None:
                price = _safe_float(hist['Close'].iloc[-1])
    except Exception:
        pass
    return (float(price) if _is_valid_price(price) else None), status


def _overnight_window(now_ny: datetime) -> tuple[datetime, datetime]:
    morning = now_ny.replace(hour=8, minute=0, second=0, microsecond=0)
    evening = now_ny.replace(hour=20, minute=0, second=0, microsecond=0)
    if now_ny.time() < dt_time(8, 0):
        end = morning
        start = (morning - timedelta(days=1)).replace(hour=20, minute=0, second=0, microsecond=0)
    else:
        end = morning
        start = (morning - timedelta(days=1)).replace(hour=20, minute=0, second=0, microsecond=0)
        if now_ny >= evening:
            end = now_ny
    return start, end


def _overnight_positive_news(limit: int = 3) -> list[dict[str, Any]]:
    now_ny = _ny_time_now()
    start, end = _overnight_window(now_ny)
    qs = StockNews.objects.filter(published_at__gte=start, published_at__lt=end)
    if not qs.exists():
        return []
    grouped = (
        qs.values('stock__symbol')
        .annotate(avg_sent=models.Avg('sentiment'), count=models.Count('id'))
        .order_by('-avg_sent', '-count')
    )
    results: list[dict[str, Any]] = []
    for row in grouped[:limit]:
        symbol = row.get('stock__symbol')
        if not symbol:
            continue
        headline = qs.filter(stock__symbol=symbol).exclude(headline__isnull=True).order_by('-published_at').first()
        results.append({
            'symbol': symbol,
            'avg_sent': float(row.get('avg_sent') or 0.0),
            'count': int(row.get('count') or 0),
            'headline': headline.headline if headline else None,
        })
    return results


def _guardian_score(
    pattern_signal: float | None,
    rvol: float | None,
    sentiment: float | None,
    imbalance: float | None,
) -> float:
    score = 0.5
    if pattern_signal is not None:
        score += max(-1.0, min(1.0, float(pattern_signal))) * 0.2
    if rvol is not None:
        score += min(1.0, float(rvol) / 3.0) * 0.15
    if sentiment is not None:
        score += max(-1.0, min(1.0, float(sentiment))) * 0.1
    if imbalance is not None:
        score += max(-1.0, min(1.0, (float(imbalance) - 1.0))) * 0.1
    return max(0.0, min(1.0, score))


def _latest_price_snapshot(symbol: str) -> float | None:
    price = get_latest_trade_price(symbol)
    if _is_valid_price(price):
        return price
    try:
        info = yfin.Ticker(symbol).fast_info or {}
        for key in ('last_price', 'lastPrice', 'last', 'regularMarketPrice'):
            val = info.get(key)
            if _is_valid_price(val):
                return float(val)
    except Exception:
        return None
    return None


def _bid_ask_spread_pct(symbol: str) -> float | None:
    spread = get_latest_bid_ask_spread_pct(symbol)
    if spread is not None:
        return float(spread)
    bid, ask = _latest_bid_ask(symbol)
    if bid is None or ask is None or ask <= 0:
        return None
    mid = (bid + ask) / 2
    if mid <= 0:
        return None
    return float((ask - bid) / mid)


def _leader_drop_pct(symbol: str, minutes: int = 240) -> float | None:
    bars = get_intraday_bars(symbol, minutes=minutes)
    if bars is None or bars.empty or 'close' not in bars.columns:
        return None
    series = bars['close'].dropna()
    if series.empty:
        return None
    first = float(series.iloc[0])
    last = float(series.iloc[-1])
    if first <= 0:
        return None
    return (last - first) / first


def _compute_rsi(series: pd.Series, period: int = 14) -> float | None:
    if series is None or series.empty or len(series) < period:
        return None
    delta = series.diff().fillna(0)
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    last_gain = float(gain.iloc[-1]) if pd.notna(gain.iloc[-1]) else None
    last_loss = float(loss.iloc[-1]) if pd.notna(loss.iloc[-1]) else None
    if last_gain is None or last_loss is None:
        return None
    if last_loss == 0:
        return 100.0 if last_gain > 0 else 0.0
    rs = last_gain / last_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)


def _extract_close_series(hist: pd.DataFrame) -> pd.Series | None:
    if hist is None or hist.empty:
        return None
    if 'Close' in hist.columns:
        return hist['Close']
    if isinstance(hist.columns, pd.MultiIndex):
        try:
            close = hist.xs('Close', axis=1, level=-1)
        except Exception:
            return None
        if isinstance(close, pd.DataFrame):
            return close.iloc[:, 0] if not close.empty else None
        return close
    return None


def _compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float | None, float | None, float | None]:
    if series is None or series.empty or len(series) < slow:
        return None, None, None
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(hist.iloc[-1])


def _compute_bollinger(series: pd.Series, window: int = 20, width: float = 2.0) -> tuple[float | None, float | None, float | None]:
    if series is None or series.empty or len(series) < window:
        return None, None, None
    ma = series.rolling(window).mean().iloc[-1]
    std = series.rolling(window).std().iloc[-1]
    if pd.isna(ma) or pd.isna(std):
        return None, None, None
    upper = float(ma + width * std)
    lower = float(ma - width * std)
    return float(ma), upper, lower


def _macd_weakening(hist_series: pd.Series) -> bool:
    if hist_series is None or hist_series.empty or len(hist_series) < 3:
        return False
    tail = hist_series.tail(3)
    return float(tail.iloc[-1]) < float(tail.iloc[-2]) < float(tail.iloc[-3])


def _build_signal_message(
    ticker: str,
    confidence: float,
    entry_price: float,
    target_price: float,
    stop_loss: float,
    pattern: str,
    rvol: float,
    liquidity_note: str,
) -> str:
    target_pct = ((target_price - entry_price) / entry_price) * 100 if entry_price else 0
    currency = _symbol_currency(ticker)
    return (
        f"🚀 SIGNAL IA : ${ticker}\n"
        f"🔥 Confiance : {confidence:.1f}%\n"
        f"📥 ENTRÉE : {entry_price:.2f} {currency}\n"
        f"💰 VENDRE À : {target_price:.2f} {currency} (Objectif +{target_pct:.2f}%)\n"
        f"🛡️ STOP-LOSS : {stop_loss:.2f} {currency}\n"
        f"📉 RAISON : {pattern} + Volume Relatif {rvol:.2f}\n"
        f"{liquidity_note}\n"
        f"🕒 HEURE : {_time_hhmm()}"
    )


def _is_lunch_time_strict(confidence: float) -> bool:
    strict_bonus = float(os.getenv('LUNCH_CONFIDENCE_BONUS', '7'))
    now_ny = _ny_time_now()
    if dt_time(12, 0) <= now_ny.time() <= dt_time(13, 0):
        return confidence >= (float(os.getenv('MIN_CONFIDENCE', '70')) + strict_bonus)
    return confidence >= float(os.getenv('MIN_CONFIDENCE', '70'))


def _active_signal_exists(ticker: str, minutes: int = 30) -> bool:
    cutoff = timezone.now() - timedelta(minutes=minutes)
    return ActiveSignal.objects.filter(ticker__iexact=ticker, status='OPEN', opened_at__gte=cutoff).exists()


def _task_log_start(task_name: str) -> TaskRunLog:
    return TaskRunLog.objects.create(task_name=task_name, status='SUCCESS')


def _task_log_finish(log: TaskRunLog, status: str, payload: dict[str, Any] | None = None, error: str = '') -> None:
    finished = timezone.now()
    duration_ms = int((finished - log.started_at).total_seconds() * 1000)
    log.status = status
    log.finished_at = finished
    log.duration_ms = duration_ms
    if error:
        log.error = str(error)[:2000]
    if payload is not None:
        log.payload = payload
    log.save(update_fields=['status', 'finished_at', 'duration_ms', 'error', 'payload'])


@shared_task
def cleanup_task_run_logs() -> dict[str, Any]:
    log = _task_log_start('cleanup_task_run_logs')
    retention_days = int(os.getenv('TASKRUNLOG_RETENTION_DAYS', '30'))
    cutoff = timezone.now() - timedelta(days=retention_days)
    deleted, _ = TaskRunLog.objects.filter(started_at__lt=cutoff).delete()
    result = {'deleted': deleted, 'retention_days': retention_days}
    _task_log_finish(log, 'SUCCESS', result)
    return result


@shared_task
def send_morning_log_email(limit: int = 200) -> dict[str, Any]:
    log = _task_log_start('send_morning_log_email')
    try:
        email_to = settings.ALERT_EMAIL_TO
        if not email_to:
            result = {'status': 'skipped', 'reason': 'no email configured'}
            _task_log_finish(log, 'SUCCESS', result)
            return result

        limit = max(10, min(500, int(limit)))
        logs = list(TaskRunLog.objects.order_by('-started_at')[:limit])
        lines = []
        failed_recent = TaskRunLog.objects.filter(
            started_at__gte=timezone.now() - timedelta(hours=24),
            status='FAILED',
        ).exists()
        confirmation = 'OK' if not failed_recent else 'ATTENTION'

        for entry in logs:
            started = entry.started_at.isoformat() if entry.started_at else 'n/a'
            duration = f"{entry.duration_ms}ms" if entry.duration_ms is not None else 'n/a'
            error = (entry.error or '').strip().replace('\n', ' ')
            if len(error) > 200:
                error = error[:200] + '...'
            lines.append(
                f"{started} | {entry.task_name} | {entry.status} | {duration} | {error}"
            )

        subject = f"Morning Logs ({confirmation}) - last {len(logs)} lines"
        header = f"Confirmation: {confirmation}\nLast 24h failed tasks: {'none' if not failed_recent else 'present'}\n\n"
        message = header + "\n".join(lines)

        send_mail(
            subject=subject,
            message=message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[email_to],
            fail_silently=True,
        )
        payload = {'status': 'sent', 'count': len(logs), 'confirmation': confirmation}
        _task_log_finish(log, 'SUCCESS', payload)
        return payload
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


def _is_valid_price(value: float | None) -> bool:
    return value is not None and isfinite(value) and value > 0


def _usd_cad_rate() -> float:
    fallback = 1.36
    try:
        fallback = float(getattr(settings, 'USD_CAD_RATE', os.getenv('USD_CAD_RATE', '1.36')))
    except (TypeError, ValueError):
        fallback = 1.36

    auto_rate = os.getenv('AUTO_USD_CAD_RATE', 'true').lower() in {'1', 'true', 'yes', 'on'}
    if not auto_rate:
        return fallback

    cached = cache.get('usd_cad_rate')
    if cached:
        try:
            return float(cached)
        except (TypeError, ValueError):
            pass

    symbol = os.getenv('USD_CAD_YF_SYMBOL', 'USDCAD=X')
    try:
        data = yf.download(
            tickers=symbol,
            period='5d',
            interval='1d',
            group_by='ticker',
            threads=False,
            auto_adjust=False,
        )
        data = _normalize_price_frame(data)
        if data is None or data.empty:
            return fallback
        close_col = 'Close' if 'Close' in data.columns else 'Adj Close' if 'Adj Close' in data.columns else None
        if not close_col:
            return fallback
        last = data[close_col].dropna()
        if last.empty:
            return fallback
        rate = float(last.iloc[-1])
        cache.set('usd_cad_rate', rate, timeout=60 * 60 * 6)
        return rate
    except Exception:
        return fallback


def _to_cad_price(symbol: str, price: float | None, info: dict[str, Any]) -> float | None:
    if price is None:
        return None
    symbol_upper = (symbol or '').upper()
    force_list = {
        s.strip().upper()
        for s in str(os.getenv('FORCE_CAD_TICKERS', '')).split(',')
        if s.strip()
    }
    if symbol_upper in force_list:
        return float(price) * _usd_cad_rate()
    if not symbol_upper.endswith('.TO'):
        return price
    currency = (info.get('currency') or info.get('financialCurrency') or '').upper()
    if currency == 'USD':
        return float(price) * _usd_cad_rate()
    return price


def _backfill_latest_price(stock: Stock) -> bool:
    last = PriceHistory.objects.filter(stock=stock).order_by('-date').first()
    if last and _is_valid_price(float(last.close_price or 0)):
        stock.latest_price = float(last.close_price)
        stock.latest_price_updated_at = timezone.now()
        stock.save(update_fields=['latest_price', 'latest_price_updated_at'])
        return True
    return False


def _normalize_weights(stocks: list[Stock]) -> list[float]:
    weights = [float(s.target_weight or 0) for s in stocks]
    total = sum(weights)
    if total > 1.5:
        return [w / 100 for w in weights]
    return weights


def _calc_yield(stocks: list[Stock]) -> float:
    if not stocks:
        return DEFAULT_YIELD
    weights = _normalize_weights(stocks)
    total = sum(weights) or 1
    return sum((weights[i] / total) * float(stocks[i].dividend_yield or 0) for i in range(len(stocks)))


def _normalize_price_frame(data: pd.DataFrame | None) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    frame = data.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        level0 = frame.columns.get_level_values(0)
        level1 = frame.columns.get_level_values(1)
        if 'Close' in level0 or 'Adj Close' in level0:
            frame.columns = level0
        elif 'Close' in level1 or 'Adj Close' in level1:
            frame.columns = level1
        else:
            frame.columns = [col[0] for col in frame.columns]
    rename_map = {
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Adj Close': 'close',
        'Volume': 'volume',
    }
    frame = frame.rename(columns=rename_map)
    if 'timestamp' not in frame.columns and isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.reset_index()
        for col in ('Date', 'Datetime', 'index'):
            if col in frame.columns:
                frame = frame.rename(columns={col: 'timestamp'})
                break
    return frame


@shared_task
def fetch_prices_hourly() -> dict[str, float]:
    log = _task_log_start('fetch_prices_hourly')
    prices: dict[str, float] = {}
    try:
        for stock in Stock.objects.all().order_by('symbol'):
            if not _is_valid_symbol(stock.symbol):
                continue
            data = None
            try:
                data = get_daily_bars(stock.symbol, days=30)
            except Exception:
                data = None
            latest_price = get_latest_trade_price(stock.symbol)
            if data is None or data.empty:
                try:
                    data = yf.download(stock.symbol, period='1mo', interval='1d', progress=False)
                except Exception:
                    data = None
            data = _normalize_price_frame(data)
            if data.empty:
                if _backfill_latest_price(stock):
                    prices[stock.symbol] = float(stock.latest_price or 0)
                continue
            data = data.sort_values('timestamp') if 'timestamp' in data.columns else data
            last_row = data.iloc[-1]
            close_value = _safe_float(last_row.get('close'))
            price = latest_price if latest_price is not None else float(close_value or 0)
            if not _is_valid_price(price):
                if _backfill_latest_price(stock):
                    prices[stock.symbol] = float(stock.latest_price or 0)
                continue
            day_low = _safe_float(last_row.get('low')) if 'low' in data.columns else None
            day_high = _safe_float(last_row.get('high')) if 'high' in data.columns else None

            price = _to_cad_price(stock.symbol, price, {})
            day_low = _to_cad_price(stock.symbol, day_low, {})
            day_high = _to_cad_price(stock.symbol, day_high, {})

            stock.latest_price = price
            stock.day_low = day_low
            stock.day_high = day_high
            stock.latest_price_updated_at = timezone.now()
            stock.save(update_fields=['latest_price', 'day_low', 'day_high', 'latest_price_updated_at'])

            for _, row in data.iterrows():
                dt = row.get('timestamp')
                if dt is None:
                    continue
                close_price = float(_safe_float(row.get('close')) or 0)
                close_price = _to_cad_price(stock.symbol, close_price, {})
                PriceHistory.objects.update_or_create(
                    stock=stock,
                    date=dt.date(),
                    defaults={'close_price': close_price},
                )

            prices[stock.symbol] = price
        _task_log_finish(log, 'SUCCESS', {'count': len(prices)})
        return prices
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        _send_alert('Task failed: fetch_prices_hourly', str(exc))
        raise


def _is_generic_news_item(headline: str | None, source: str | None) -> bool:
    text = f"{headline or ''} {source or ''}".strip().lower()
    if not text:
        return True
    patterns = (
        'stock traders daily',
        'strategic equity report',
        'investment strategy',
        'equity report',
        'edition - stock traders daily',
    )
    return any(pattern in text for pattern in patterns)


def _is_duplicate_news_item(stock: Stock, headline: str | None, published_at: datetime | None) -> bool:
    if not headline:
        return False
    if published_at is None:
        published_at = timezone.now()
    window_start = published_at - timedelta(days=2)
    return StockNews.objects.filter(
        stock=stock,
        headline__iexact=headline,
        published_at__gte=window_start,
    ).exists()


@shared_task
def fetch_fundamentals_daily() -> dict[str, int]:
    log = _task_log_start('fetch_fundamentals_daily')
    updated = 0
    skipped = 0
    try:
        for stock in Stock.objects.all().order_by('symbol'):
            if not _is_valid_symbol(stock.symbol) or _skip_fundamentals_info(stock.symbol):
                skipped += 1
                continue
            ticker = yf.Ticker(stock.symbol)
            try:
                info = ticker.info or {}
            except Exception:
                info = {}
            sector = (info.get('sector') or '').strip()
            div_yield = info.get('dividendYield')
            if div_yield is None:
                div_yield = info.get('trailingAnnualDividendYield')
            div_yield = float(div_yield) if div_yield is not None else None
            price_hint = info.get('regularMarketPrice') or info.get('currentPrice') or stock.latest_price
            if stock.symbol.upper() == 'AVGO':
                dividend_rate = info.get('dividendRate') or info.get('trailingAnnualDividendRate')
                if dividend_rate is not None:
                    try:
                        div_yield = float(dividend_rate) / 332.54
                    except Exception:
                        pass
            if stock.symbol.upper() == 'TEC.TO':
                dividend_rate = info.get('dividendRate') or info.get('trailingAnnualDividendRate')
                if dividend_rate is not None and price_hint:
                    try:
                        div_yield = float(dividend_rate) / float(price_hint)
                    except Exception:
                        pass
                if div_yield is not None:
                    div_yield = min(float(div_yield), 0.02)

            changed = False
            if sector and ((not stock.sector) or stock.sector.lower() == 'unknown'):
                stock.sector = sector
                changed = True
            if div_yield is not None:
                if stock.symbol.upper() in {'AVGO', 'TEC.TO'}:
                    stock.dividend_yield = div_yield
                    changed = True
                elif not stock.dividend_yield or float(stock.dividend_yield or 0) == 0:
                    stock.dividend_yield = div_yield
                    changed = True

            if changed:
                stock.save(update_fields=['sector', 'dividend_yield'])
                updated += 1
        _task_log_finish(log, 'SUCCESS', {'updated': updated, 'skipped': skipped})
        return {'updated': updated, 'skipped': skipped}
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        _send_alert('Task failed: fetch_fundamentals_daily', str(exc))
        raise


@shared_task
def refresh_dividend_yield(symbols: list[str] | None = None) -> dict[str, Any]:
    log = _task_log_start('refresh_dividend_yield')
    updated = 0
    skipped = 0
    payload_symbols = symbols or []
    if not payload_symbols:
        env_list = os.getenv('DIVIDEND_REFRESH_SYMBOLS', '')
        payload_symbols = [s.strip().upper() for s in env_list.split(',') if s.strip()]
    try:
        for symbol in payload_symbols:
            stock = Stock.objects.filter(symbol__iexact=symbol).first()
            if not stock:
                skipped += 1
                continue
            try:
                info = yf.Ticker(stock.symbol).info or {}
            except Exception:
                info = {}
            div_yield = info.get('dividendYield')
            if div_yield is None:
                div_yield = info.get('trailingAnnualDividendYield')
            div_yield = float(div_yield) if div_yield is not None else None
            price_hint = info.get('regularMarketPrice') or info.get('currentPrice') or stock.latest_price
            if div_yield is None:
                dividend_rate = info.get('dividendRate') or info.get('trailingAnnualDividendRate')
                if dividend_rate is not None and price_hint:
                    try:
                        div_yield = float(dividend_rate) / float(price_hint)
                    except Exception:
                        div_yield = None
            if div_yield is None:
                try:
                    dividends = yf.Ticker(stock.symbol).dividends
                except Exception:
                    dividends = None
                if dividends is not None and not dividends.empty:
                    try:
                        recent = dividends.tail(12)
                        annualized = float(recent.sum())
                        if price_hint:
                            div_yield = annualized / float(price_hint)
                    except Exception:
                        div_yield = None
            if div_yield is None:
                skipped += 1
                continue
            stock.dividend_yield = div_yield
            stock.save(update_fields=['dividend_yield'])
            updated += 1
        _task_log_finish(log, 'SUCCESS', {'updated': updated, 'skipped': skipped})
        return {'updated': updated, 'skipped': skipped}
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        _send_alert('Task failed: refresh_dividend_yield', str(exc))
        raise


@shared_task
def fetch_news_daily(days: int = 1, page_size: int = 10, language: str = 'en') -> dict[str, int]:
    log = _task_log_start('fetch_news_daily')
    api_key = os.getenv('NEWSAPI_KEY')
    if not api_key:
        _task_log_finish(log, 'SUCCESS', {'created': 0, 'seen': 0, 'skipped': 'missing_api_key'})
        return {'created': 0, 'seen': 0}

    newsapi = NewsApiClient(api_key=api_key)
    analyzer = SentimentIntensityAnalyzer()
    from_dt = timezone.now() - timedelta(days=days)
    from_iso = from_dt.strftime('%Y-%m-%d')

    created = 0
    seen = 0
    skipped = 0
    skipped = 0
    updated = 0
    skipped = 0

    try:
        for stock in Stock.objects.all().order_by('symbol'):
            result: dict[str, Any] = newsapi.get_everything(
                q=stock.symbol,
                language=language,
                sort_by='publishedAt',
                from_param=from_iso,
                page_size=page_size,
            )
            articles = result.get('articles') or []

            for a in articles:
                seen += 1
                url = (a.get('url') or '').strip()
                if not url:
                    continue
                url = url[:500]

                published_at = None
                published_raw = a.get('publishedAt')
                if published_raw:
                    try:
                        published_at = datetime.fromisoformat(published_raw.replace('Z', '+00:00'))
                    except ValueError:
                        published_at = None

                headline = (a.get('title') or '').strip()[:300] or url
                description = (a.get('description') or '').strip()
                text_for_sentiment = f"{headline}. {description}".strip()
                sentiment = analyzer.polarity_scores(text_for_sentiment).get('compound')
                source = ((a.get('source') or {}).get('name') or '').strip()[:100]
                if _is_generic_news_item(headline, source):
                    skipped += 1
                    continue
                if _is_duplicate_news_item(stock, headline, published_at):
                    skipped += 1
                    continue

                _, was_created = StockNews.objects.get_or_create(
                    url=url,
                    defaults={
                        'stock': stock,
                        'headline': headline,
                        'source': source,
                        'published_at': published_at,
                        'sentiment': sentiment,
                        'raw': a,
                    },
                )
                if was_created:
                    created += 1
                else:
                    updated += 1

        _task_log_finish(log, 'SUCCESS', {'created': created, 'updated': updated, 'skipped': skipped})
        return {'created': created, 'updated': updated, 'skipped': skipped}
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        _send_alert('Task failed: fetch_news_daily', str(exc))
        raise


@shared_task
def fetch_finnhub_news_daily(days: int = 1) -> dict[str, int]:
    log = _task_log_start('fetch_finnhub_news_daily')
    news_enabled = os.getenv('FINNHUB_NEWS_ENABLED', 'false').strip().lower() in {'1', 'true', 'yes', 'on'}
    if not news_enabled:
        result = {'created': 0, 'seen': 0, 'skipped': 'disabled'}
        _task_log_finish(log, 'SUCCESS', result)
        return result
    api_key = os.getenv('FINNHUB_API_KEY')
    if not api_key:
        _task_log_finish(log, 'SUCCESS', {'created': 0, 'seen': 0, 'skipped': 'missing_api_key'})
        return {'created': 0, 'seen': 0}

    client = finnhub.Client(api_key=api_key)
    analyzer = SentimentIntensityAnalyzer()

    to_dt = timezone.now().date()
    from_dt = to_dt - timedelta(days=days)

    created = 0
    seen = 0

    try:
        for stock in Stock.objects.all().order_by('symbol'):
            news = client.company_news(stock.symbol, _from=str(from_dt), to=str(to_dt))
            for item in news:
                seen += 1
                url = (item.get('url') or '').strip()
                if not url:
                    continue
                url = url[:500]

                headline = (item.get('headline') or '').strip()[:300] or url
                summary = (item.get('summary') or '').strip()
                text_for_sentiment = f"{headline}. {summary}".strip()
                sentiment = analyzer.polarity_scores(text_for_sentiment).get('compound')

                published_at = None
                if item.get('datetime'):
                    published_at = datetime.fromtimestamp(item['datetime'], tz=timezone.UTC)

                if _is_generic_news_item(headline, 'Finnhub'):
                    skipped += 1
                    continue
                if _is_duplicate_news_item(stock, headline, published_at):
                    skipped += 1
                    continue

                _, was_created = StockNews.objects.get_or_create(
                    url=url,
                    defaults={
                        'stock': stock,
                        'headline': headline,
                        'source': 'Finnhub',
                        'published_at': published_at,
                        'sentiment': sentiment,
                        'raw': item,
                    },
                )
                if was_created:
                    created += 1

        _task_log_finish(log, 'SUCCESS', {'created': created, 'seen': seen, 'skipped': skipped})
        return {'created': created, 'seen': seen, 'skipped': skipped}
    except Exception as exc:
        error_text = str(exc)
        if 'status_code: 403' in error_text or '403' in error_text:
            result = {'created': created, 'seen': seen, 'skipped': 'finnhub_403'}
            _task_log_finish(log, 'SUCCESS', result)
            return result
        _task_log_finish(log, 'FAILED', error=error_text)
        _send_alert('Task failed: fetch_finnhub_news_daily', error_text)
        raise


@shared_task
def fetch_google_news_daily(days: int = 1) -> dict[str, int]:
    log = _task_log_start('fetch_google_news_daily')
    analyzer = SentimentIntensityAnalyzer()
    cutoff = timezone.now() - timedelta(days=days)

    created = 0
    seen = 0
    skipped = 0
    skipped = 0

    try:
        for stock in Stock.objects.all().order_by('symbol'):
            query = quote_plus(f"{stock.symbol} stock")
            url = (
                "https://news.google.com/rss/search"
                f"?q={query}&hl=en-CA&gl=CA&ceid=CA:en"
            )
            feed = feedparser.parse(url)

            for entry in feed.entries:
                seen += 1
                link = (entry.get('link') or '').strip()
                if not link:
                    continue
                link = link[:500]

                headline = (entry.get('title') or '').strip()[:300] or link
                summary = (entry.get('summary') or '').strip()
                text_for_sentiment = f"{headline}. {summary}".strip()
                sentiment = analyzer.polarity_scores(text_for_sentiment).get('compound')

                published_at = None
                if entry.get('published'):
                    try:
                        published_at = parsedate_to_datetime(entry['published'])
                    except Exception:
                        published_at = None

                if published_at and published_at < cutoff:
                    continue
                if _is_generic_news_item(headline, 'Google News'):
                    skipped += 1
                    continue
                if _is_duplicate_news_item(stock, headline, published_at):
                    skipped += 1
                    continue

                _, was_created = StockNews.objects.get_or_create(
                    url=link,
                    defaults={
                        'stock': stock,
                        'headline': headline,
                        'source': 'Google News',
                        'published_at': published_at,
                        'sentiment': sentiment,
                        'raw': entry,
                    },
                )
                if was_created:
                    created += 1

        _task_log_finish(log, 'SUCCESS', {'created': created, 'seen': seen, 'skipped': skipped})
        return {'created': created, 'seen': seen, 'skipped': skipped}
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        _send_alert('Task failed: fetch_google_news_daily', str(exc))
        raise


@shared_task
def fetch_macro_daily(start: str = '2025-01-01') -> dict[str, int]:
    log = _task_log_start('fetch_macro_daily')
    try:
        call_command('fetch_macro', start=start)
    except Exception:
        _task_log_finish(log, 'FAILED', error='fetch_macro failed')
        _send_alert('Task failed: fetch_macro_daily', 'fetch_macro failed')
        return {'ok': 0}
    _task_log_finish(log, 'SUCCESS', {'ok': 1})
    return {'ok': 1}


@shared_task
def ensure_data_pipeline_daily() -> dict[str, Any]:
    """Quality checks and backfills for market, macro, and news data."""
    log = _task_log_start('ensure_data_pipeline_daily')
    results: dict[str, Any] = {}
    cutoff = timezone.now() - timedelta(days=2)
    stale = Stock.objects.filter(
        models.Q(latest_price_updated_at__lt=cutoff) | models.Q(latest_price_updated_at__isnull=True)
    )
    backfilled_prices = 0
    for stock in stale:
        if _backfill_latest_price(stock):
            backfilled_prices += 1
    results['price_backfilled'] = backfilled_prices
    results['stale_prices'] = stale.count()

    macro_recent = MacroIndicator.objects.filter(
        date__gte=timezone.now().date() - timedelta(days=1)
    ).exists()
    if not macro_recent:
        try:
            last = MacroIndicator.objects.order_by('-date').first()
            start_date = (last.date + timedelta(days=1)) if last else (timezone.now().date() - timedelta(days=30))
            call_command('fetch_macro', start=start_date.isoformat())
            results['macro_backfill'] = True
        except Exception as exc:
            results['macro_backfill'] = False
            results['macro_error'] = str(exc)
    else:
        results['macro_backfill'] = False

    news_cutoff = timezone.now() - timedelta(hours=24)
    news_count = StockNews.objects.filter(fetched_at__gte=news_cutoff).count()
    results['news_count_24h'] = news_count
    min_daily_news = int(os.getenv('MIN_DAILY_NEWS', '25'))
    if news_count < min_daily_news:
        backfill_days = int(os.getenv('NEWS_BACKFILL_DAYS', '3'))
        results['news_backfill'] = {
            'newsapi': fetch_news_daily(days=backfill_days, page_size=20),
            'finnhub': fetch_finnhub_news_daily(days=backfill_days),
            'google': fetch_google_news_daily(days=backfill_days),
            'press_releases': fetch_press_releases_hourly(hours=72),
        }
        _send_alert(
            'Data gap: low news volume',
            f"News count 24h={news_count}, backfilled {backfill_days} days",
        )
    else:
        results['news_backfill'] = None

    if results['stale_prices']:
        _send_alert('Data gap: stale prices', f"Stale prices={results['stale_prices']}")
    if results.get('macro_backfill') is True:
        _send_alert('Data gap: macro backfill', 'Macro data was backfilled')

    _task_log_finish(log, 'SUCCESS', results)
    return results


@shared_task
def fetch_press_releases_hourly(hours: int = 24) -> dict[str, int]:
    log = _task_log_start('fetch_press_releases_hourly')
    cutoff = timezone.now() - timedelta(hours=hours)
    finbert = _finbert_pipeline()

    created = 0
    seen = 0

    try:
        for stock in Stock.objects.all().order_by('symbol'):
            query = quote_plus(f"{stock.symbol} press release")
            url = (
                "https://news.google.com/rss/search"
                f"?q={query}&hl=en-CA&gl=CA&ceid=CA:en"
            )
            feed = feedparser.parse(url)

            for entry in feed.entries:
                seen += 1
                link = (entry.get('link') or '').strip()
                if not link:
                    continue
                link = link[:500]

                headline = (entry.get('title') or '').strip()[:300] or link
                summary = (entry.get('summary') or '').strip()

                published_at = None
                if entry.get('published'):
                    try:
                        published_at = parsedate_to_datetime(entry['published'])
                    except Exception:
                        published_at = None

                if published_at and published_at < cutoff:
                    continue

                text = f"{headline}. {summary}".strip()
                finbert_result = finbert(text[:1000])[0]
                label = finbert_result.get('label', '').lower()
                score = float(finbert_result.get('score') or 0)
                if label == 'positive':
                    sentiment = score
                elif label == 'negative':
                    sentiment = -score
                else:
                    sentiment = 0.0

                _, was_created = StockNews.objects.get_or_create(
                    url=link,
                    defaults={
                        'stock': stock,
                        'headline': headline,
                        'source': 'Press Release',
                        'published_at': published_at,
                        'sentiment': sentiment,
                        'raw': _sanitize_json({
                            'summary': summary,
                            'finbert': finbert_result,
                        }),
                    },
                )
                if was_created:
                    created += 1

        _task_log_finish(log, 'SUCCESS', {'created': created, 'seen': seen})
        return {'created': created, 'seen': seen}
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        _send_alert('Task failed: fetch_press_releases_hourly', str(exc))
        raise


@shared_task
def calculate_drip_weekly() -> dict[str, float]:
    projections: dict[str, float] = {}
    for portfolio in Portfolio.objects.all().prefetch_related('stocks'):
        stocks = list(portfolio.stocks.all())
        yield_rate = _calc_yield(stocks)
        projections[portfolio.name] = float(portfolio.capital or 0) * (1 + yield_rate)
    return projections


@shared_task
def calculate_drip_monthly() -> dict[str, float]:
    results: dict[str, float] = {}
    today = timezone.now().date()

    for portfolio in Portfolio.objects.all().prefetch_related('stocks'):
        transactions = portfolio.transaction_set.select_related('stock')
        total_invested = 0.0
        per_symbol = {}

        for tx in transactions:
            sign = -1 if tx.transaction_type == 'SELL' else 1
            shares = float(tx.shares or 0) * sign
            invested = shares * float(tx.price_per_share or 0)
            symbol = tx.stock.symbol
            dy = float(tx.stock.dividend_yield or 0)

            total_invested += invested
            if symbol not in per_symbol:
                per_symbol[symbol] = {'invested': 0.0, 'dividend_yield': dy}
            per_symbol[symbol]['invested'] += invested
            if dy:
                per_symbol[symbol]['dividend_yield'] = dy

        if total_invested > 0:
            weighted_yield = 0.0
            for entry in per_symbol.values():
                if entry['invested'] > 0:
                    weighted_yield += (entry['invested'] / total_invested) * entry['dividend_yield']
            capital = total_invested
            yield_rate = weighted_yield
        else:
            stocks = list(portfolio.stocks.all())
            yield_rate = _calc_yield(stocks)
            capital = float(portfolio.capital or 0)

        monthly_yield = yield_rate / 12
        dividend_income = capital * monthly_yield
        new_capital = capital + dividend_income

        DripSnapshot.objects.update_or_create(
            portfolio=portfolio,
            as_of=today,
            defaults={
                'capital': new_capital,
                'dividend_income': dividend_income,
                'yield_rate': yield_rate,
            },
        )

        results[portfolio.name] = new_capital

    return results


@shared_task
def check_alerts() -> dict[str, int]:
    price_threshold = settings.ALERT_PRICE_THRESHOLD
    drop_pct_threshold = settings.ALERT_DROP_PCT
    capital_threshold = settings.ALERT_CAPITAL_THRESHOLD
    email_to = settings.ALERT_EMAIL_TO
    sms_to = settings.ALERT_SMS_TO
    cooldown_hours = getattr(settings, 'ALERT_COOLDOWN_HOURS', 12)

    hits = 0

    def send_alert(subject: str, message: str, category: str, stock: Stock | None = None,
                   portfolio: Portfolio | None = None):
        nonlocal hits
        if cooldown_hours and cooldown_hours > 0:
            cutoff = timezone.now() - timedelta(hours=cooldown_hours)
            recent = AlertEvent.objects.filter(
                category=category,
                stock=stock,
                portfolio=portfolio,
                created_at__gte=cutoff,
            ).exists()
            if recent:
                return
        hits += 1

        AlertEvent.objects.create(
            category=category,
            message=message,
            stock=stock,
            portfolio=portfolio,
        )

        if email_to:
            send_mail(
                subject=subject,
                message=message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[email_to],
                fail_silently=True,
            )

        if sms_to and settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN and settings.TWILIO_FROM_NUMBER:
            try:
                from twilio.rest import Client

                client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
                client.messages.create(
                    body=f"{subject}: {message}",
                    from_=settings.TWILIO_FROM_NUMBER,
                    to=sms_to,
                )
            except Exception:
                pass

    today = timezone.now().date()

    def _is_crypto_symbol(symbol: str) -> bool:
        symbol = (symbol or '').upper()
        return symbol.endswith('-CAD') or symbol.endswith('-USD')

    for stock in Stock.objects.all().order_by('symbol'):
        # Price drop alert (uses last two stored closes)
        recent_prices = list(
            PriceHistory.objects.filter(stock=stock).order_by('-date')[:2]
        )
        if len(recent_prices) == 2:
            latest = recent_prices[0].close_price
            previous = recent_prices[1].close_price
            if previous > 0:
                drop_pct = ((previous - latest) / previous) * 100
                if drop_pct >= drop_pct_threshold:
                    send_alert(
                        subject=f"Price drop: {stock.symbol}",
                        message=f"Dropped {drop_pct:.2f}% ({previous:.2f} → {latest:.2f}).",
                        category='PRICE_DROP',
                        stock=stock,
                    )

        # Absolute price threshold alert
        if (
            stock.latest_price is not None
            and stock.latest_price >= price_threshold
            and not _is_crypto_symbol(stock.symbol)
        ):
            send_alert(
                subject=f"Price alert: {stock.symbol}",
                message=f"{stock.symbol} hit ${stock.latest_price:.2f} (threshold ${price_threshold:.2f}).",
                category='PRICE_THRESHOLD',
                stock=stock,
            )

        # Dividend change alert
        last_dividend = (
            DividendHistory.objects.filter(stock=stock).order_by('-date').first()
        )
        current_dividend = float(stock.dividend_yield or 0)

        if last_dividend and last_dividend.dividend_yield != current_dividend:
            send_alert(
                subject=f"Dividend change: {stock.symbol}",
                message=f"Yield changed from {last_dividend.dividend_yield:.4f} to {current_dividend:.4f}.",
                category='DIVIDEND_CHANGE',
                stock=stock,
            )

        DividendHistory.objects.update_or_create(
            stock=stock,
            date=today,
            defaults={'dividend_yield': current_dividend},
        )

    # Capital threshold alert
    for portfolio in Portfolio.objects.all():
        capital = float(portfolio.capital or 0)
        if capital < capital_threshold:
            send_alert(
                subject=f"Capital alert: {portfolio.name}",
                message=f"Capital ${capital:.2f} is below ${capital_threshold:.2f}.",
                category='CAPITAL_THRESHOLD',
                portfolio=portfolio,
            )

    return {'hits': hits}


@shared_task
def check_event_risks(days: int = 1) -> dict[str, int]:
    keywords = [k.lower() for k in settings.EVENT_RISK_KEYWORDS]
    cutoff = timezone.now() - timedelta(days=days)
    hits = 0

    recent_news = StockNews.objects.filter(fetched_at__gte=cutoff)
    for news in recent_news.select_related('stock'):
        text = f"{news.headline} {news.raw.get('summary', '') if isinstance(news.raw, dict) else ''}".lower()
        matched = [k for k in keywords if k in text]
        if not matched:
            continue

        message = (
            f"Potential event risk for {news.stock.symbol}: "
            f"{news.headline} (keywords: {', '.join(matched)})"
        )

        already = AlertEvent.objects.filter(
            category='EVENT_RISK',
            message=message,
            created_at__date=timezone.now().date(),
        ).exists()

        if not already:
            AlertEvent.objects.create(
                category='EVENT_RISK',
                message=message,
                stock=news.stock,
            )
            hits += 1

    return {'hits': hits}


@shared_task
def calculate_drip(account_id: int) -> list[dict[str, float | str]]:
    account = Account.objects.get(id=account_id)
    stocks = list(Stock.objects.all())
    transactions = AccountTransaction.objects.filter(account=account)

    capital = 0.0
    for t in transactions:
        sign = -1 if t.type == 'SELL' else 1
        capital += sign * float(t.quantity) * float(t.price)

    drip_data: list[dict[str, float | str]] = []
    for stock in stocks:
        dividend_amount = float(stock.dividend_yield or 0) * float(stock.target_weight or 0) * capital
        drip_data.append({
            'stock': stock.symbol,
            'dividend': dividend_amount,
            'reinvested': dividend_amount,
        })

    return drip_data


@shared_task
def weekly_ai_forecast() -> dict[str, str | float]:
    results: dict[str, str | float] = {}
    for stock in Stock.objects.all().order_by('symbol'):
        predicted_price, recommendation = run_predictions(stock.symbol)
        Prediction.objects.create(
            stock=stock,
            predicted_price=predicted_price,
            recommendation=recommendation,
            date=timezone.now().date(),
        )
        results[stock.symbol] = predicted_price

    return results


@shared_task
def generate_portfolio_digest(days: int = 7) -> dict[str, str]:
    end_date = timezone.now().date()
    start_date = end_date - timedelta(days=days)
    results: dict[str, str] = {}

    for portfolio in Portfolio.objects.all():
        holdings = PortfolioHolding.objects.select_related('stock').filter(portfolio=portfolio)
        total_value = 0.0
        start_value = 0.0
        movers = []

        for h in holdings:
            stock = h.stock
            latest = PriceHistory.objects.filter(stock=stock).order_by('-date').first()
            past = PriceHistory.objects.filter(stock=stock, date__lte=start_date).order_by('-date').first()

            if latest and past:
                value_now = float(h.shares or 0) * float(latest.close_price)
                value_then = float(h.shares or 0) * float(past.close_price)
                total_value += value_now
                start_value += value_then

                change_pct = ((latest.close_price - past.close_price) / past.close_price) * 100
                movers.append((stock.symbol, change_pct))

        growth_pct = ((total_value - start_value) / start_value) * 100 if start_value else 0
        movers.sort(key=lambda x: x[1], reverse=True)
        top = movers[0] if movers else None

        sentiment = (
            StockNews.objects.filter(stock__in=[h.stock for h in holdings])
            .aggregate(avg=models.Avg('sentiment'))
            .get('avg')
        ) or 0

        summary = (
            f"Your {portfolio.name} portfolio changed {growth_pct:.2f}% in the last {days} days. "
        )
        if top:
            summary += f"{top[0]} moved {top[1]:.2f}%. "
        summary += f"Average news sentiment: {sentiment:.2f}."

        PortfolioDigest.objects.create(
            portfolio=portfolio,
            period_start=start_date,
            period_end=end_date,
            summary=summary,
        )

        results[portfolio.name] = summary

    return results


@shared_task
def update_user_preferences() -> dict[str, int]:
    results: dict[str, int] = {}
    for account in Account.objects.select_related('user').all():
        user = account.user
        pref, _ = UserPreference.objects.get_or_create(user=user)

        sector_counts: dict[str, float] = dict(pref.preferred_sectors or {})
        txs = AccountTransaction.objects.filter(account=account)
        for tx in txs:
            sector = tx.stock.sector or 'Unknown'
            delta = float(tx.quantity or 0) if tx.type != 'SELL' else -float(tx.quantity or 0)
            sector_counts[sector] = sector_counts.get(sector, 0) + delta

        pref.preferred_sectors = sector_counts
        pref.save(update_fields=['preferred_sectors', 'updated_at'])
        results[user.id] = len(sector_counts)

    return results


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, pd.Series):
            if value.empty:
                return None
            value = value.iloc[0]
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _is_valid_symbol(symbol: str) -> bool:
    if not symbol:
        return False
    symbol = symbol.strip().upper()
    if symbol in {'.', '-', '_'}:
        return False
    return bool(re.fullmatch(r"[A-Z0-9\.\-]{1,10}", symbol))


def _fetch_json(url: str) -> Any:
    try:
        resp = requests.get(url, timeout=12)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def _sanitize_json(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _finbert_pipeline():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")


def _latest_indicator_value(series: dict[str, dict[str, str]], key: str) -> float | None:
    if not series:
        return None
    latest_date = sorted(series.keys(), reverse=True)[0]
    return _safe_float(series.get(latest_date, {}).get(key))


def _fetch_yahoo_screener(scr_id: str, count: int = 250) -> list[dict[str, Any]]:
    url = (
        "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
        f"?formatted=false&scrIds={scr_id}&count={count}&start=0"
    )
    try:
        resp = requests.get(
            url,
            timeout=12,
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
            },
        )
        if resp.status_code != 200:
            return []
        if 'Too Many Requests' in resp.text:
            return []
        payload = resp.json() or {}
    except Exception:
        return []
    try:
        result = (payload.get('finance') or {}).get('result') or []
        if not result:
            return []
    except Exception:
        return []

    quotes = result[0].get('quotes') or []
    return quotes


def _fetch_sec_tickers(limit: int = 5000) -> list[str]:
    url = "https://www.sec.gov/files/company_tickers_exchange.json"
    try:
        resp = requests.get(
            url,
            timeout=12,
            headers={
                'User-Agent': 'PersonalStock/1.0 (contact: local)',
                'Accept-Language': 'en-US,en;q=0.9',
            },
        )
        if resp.status_code != 200:
            return []
        payload = resp.json() or {}
    except Exception:
        return []

    data = payload.get('data') or []
    symbols: list[str] = []
    for row in data:
        if not isinstance(row, list) or len(row) < 3:
            continue
        symbol = (row[2] or '').strip().upper()
        if symbol and symbol.isalpha():
            symbols.append(symbol)
        if len(symbols) >= limit:
            break
    return symbols


def _fetch_yfinance_screeners(scr_ids: list[str], count: int = 200) -> list[dict[str, Any]]:
    quotes: list[dict[str, Any]] = []
    for scr_id in scr_ids:
        try:
            payload = yf.screen(scr_id, count=count) or {}
        except Exception:
            continue
        quotes.extend(payload.get('quotes') or [])

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in quotes:
        symbol = (item.get('symbol') or '').strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        unique.append(item)
    return unique


@shared_task
def run_penny_ai_scout() -> dict[str, int]:
    """Nightly FMP + Alpha Vantage pipeline for penny stock AI scores."""
    fmp_key = os.getenv('FMP_API_KEY')
    av_key = os.getenv('ALPHAVANTAGE_API_KEY')
    if not fmp_key:
        return {'created': 0, 'updated': 0}

    min_price = float(os.getenv('PENNY_MIN_PRICE', '0.00001'))
    max_price = float(os.getenv('PENNY_MAX_PRICE', '1.0'))
    min_volume = int(os.getenv('PENNY_MIN_VOLUME', '100000'))
    max_market_cap = int(os.getenv('PENNY_MAX_MARKET_CAP', '50000000'))
    max_symbols = int(os.getenv('PENNY_MAX_SYMBOLS', '50'))

    screener_url = (
        "https://financialmodelingprep.com/stable/company-screener"
        f"?priceLowerThan={max_price}&volumeMoreThan={min_volume}"
        f"&marketCapLowerThan={max_market_cap}&limit={max_symbols}&apikey={fmp_key}"
    )
    universe = _fetch_json(screener_url) or []
    restricted = False
    if isinstance(universe, dict):
        error_msg = universe.get('Error Message') or universe.get('error') or universe.get('message')
        if error_msg and 'Restricted Endpoint' in str(error_msg):
            restricted = True
            universe = []
        elif error_msg:
            return {'created': 0, 'updated': 0}
    if not isinstance(universe, list):
        return {'created': 0, 'updated': 0}

    if not universe:
        scr_id = os.getenv('PENNY_SCREENER_ID', 'penny_stocks')
        yahoo_quotes = _fetch_yahoo_screener(scr_id, count=max_symbols)
        for quote in yahoo_quotes:
            symbol = (quote.get('symbol') or '').strip().upper()
            if not symbol:
                continue
            price = _safe_float(quote.get('regularMarketPrice'))
            volume = _safe_float(quote.get('regularMarketVolume') or quote.get('averageDailyVolume3Month'))
            market_cap = _safe_float(quote.get('marketCap'))

            if price is not None and price < min_price:
                continue
            if price is not None and price > max_price:
                continue
            if volume is not None and volume < min_volume:
                continue
            if market_cap is not None and market_cap > max_market_cap:
                continue

            universe.append({
                'symbol': symbol,
                'companyName': quote.get('shortName') or quote.get('longName') or symbol,
                'exchangeShortName': quote.get('fullExchangeName') or quote.get('exchange') or '',
                'sector': quote.get('sector') or '',
                'industry': quote.get('industry') or '',
                'price': price,
                'marketCap': market_cap,
                'volume': volume,
            })

    if not universe:
        scr_ids = [
            s.strip()
            for s in os.getenv('PENNY_YF_SCREENERS', 'aggressive_small_caps,small_cap_gainers,day_gainers,most_actives').split(',')
            if s.strip()
        ]
        ignore_mcap = os.getenv('PENNY_YF_IGNORE_MARKET_CAP', 'true').lower() == 'true'
        yf_count = int(os.getenv('PENNY_YF_COUNT', '200'))
        yf_quotes = _fetch_yfinance_screeners(scr_ids, count=yf_count)
        for quote in yf_quotes:
            symbol = (quote.get('symbol') or '').strip().upper()
            if not symbol:
                continue
            price = _safe_float(quote.get('regularMarketPrice'))
            volume = _safe_float(quote.get('regularMarketVolume') or quote.get('averageDailyVolume3Month'))
            market_cap = _safe_float(quote.get('marketCap'))

            if price is not None and price < min_price:
                continue
            if price is not None and price > max_price:
                continue
            if volume is not None and volume < min_volume:
                continue
            if not ignore_mcap and market_cap is not None and market_cap > max_market_cap:
                continue

            universe.append({
                'symbol': symbol,
                'companyName': quote.get('shortName') or quote.get('longName') or symbol,
                'exchangeShortName': quote.get('fullExchangeName') or quote.get('exchange') or '',
                'sector': quote.get('sector') or '',
                'industry': quote.get('industry') or '',
                'price': price,
                'marketCap': market_cap,
                'volume': volume,
            })

    if not universe:
        sample_size = int(os.getenv('PENNY_SAMPLE_SIZE', '400'))
        seed_symbols = [s.strip().upper() for s in os.getenv('PENNY_SEED_SYMBOLS', '').split(',') if s.strip()]
        sec_symbols = _fetch_sec_tickers(limit=max(sample_size, 500))
        if sec_symbols:
            random.shuffle(sec_symbols)
            sec_symbols = sec_symbols[:sample_size]
        candidates = list(dict.fromkeys(seed_symbols + sec_symbols))
        if candidates:
            try:
                data = yf.download(
                    tickers=" ".join(candidates),
                    period='5d',
                    interval='1d',
                    group_by='ticker',
                    threads=True,
                )
            except Exception:
                data = None

            if data is not None:
                for symbol in candidates:
                    if symbol not in data:
                        continue
                    frame = data[symbol]
                    if frame is None or frame.empty:
                        continue
                    price = _safe_float(frame['Close'].iloc[-1]) if 'Close' in frame else None
                    volume = _safe_float(frame['Volume'].iloc[-1]) if 'Volume' in frame else None
                    if price is not None and price < min_price:
                        continue
                    if price is not None and price > max_price:
                        continue
                    if volume is not None and volume < min_volume:
                        continue
                    universe.append({
                        'symbol': symbol,
                        'companyName': symbol,
                        'exchangeShortName': '',
                        'sector': '',
                        'industry': '',
                        'price': price,
                        'marketCap': None,
                        'volume': volume,
                    })

    if not universe:
        signal_qs = list(PennySignal.objects.order_by('-as_of', '-combined_score')[:max_symbols])
        signal_map = {s.symbol: s for s in signal_qs}
        fallback_symbols = list(signal_map.keys())
        if not fallback_symbols:
            fallback_symbols = list(Stock.objects.order_by('symbol').values_list('symbol', flat=True)[:max_symbols])

        for symbol in fallback_symbols:
            ticker = yf.Ticker(symbol)
            info: dict[str, Any]
            try:
                info = ticker.info or {}
            except Exception:
                info = {}

            price = _safe_float(info.get('regularMarketPrice'))
            volume = _safe_float(info.get('regularMarketVolume') or info.get('averageVolume'))
            market_cap = _safe_float(info.get('marketCap'))

            if price is None and symbol in signal_map:
                price = _safe_float(signal_map[symbol].last_price)
            if volume is None and symbol in signal_map:
                volume = _safe_float(signal_map[symbol].avg_volume)

            if price is None:
                try:
                    data = ticker.history(period='5d', interval='1d', timeout=10)
                    if data is not None and not data.empty and 'Close' in data:
                        price = float(data['Close'].iloc[-1])
                        if volume is None and 'Volume' in data:
                            volume = float(data['Volume'].iloc[-1])
                except Exception:
                    pass

            if price is not None and price > max_price:
                continue

            if volume is not None and volume < min_volume:
                continue

            if market_cap is not None and market_cap > max_market_cap:
                continue

            universe.append({
                'symbol': symbol,
                'companyName': info.get('longName') or info.get('shortName') or symbol,
                'exchangeShortName': info.get('exchange') or info.get('fullExchangeName') or '',
                'sector': info.get('sector') or '',
                'industry': info.get('industry') or '',
                'price': price,
                'marketCap': market_cap,
                'volume': volume,
            })

        if not universe and signal_map:
            for symbol, signal in signal_map.items():
                universe.append({
                    'symbol': symbol,
                    'companyName': symbol,
                    'exchangeShortName': '',
                    'sector': '',
                    'industry': '',
                    'price': _safe_float(signal.last_price),
                    'marketCap': None,
                    'volume': _safe_float(signal.avg_volume),
                })

    created = 0
    updated = 0
    today = timezone.now().date()
    seen = len(universe)

    for item in universe[:max_symbols]:
        symbol = (item.get('symbol') or '').strip().upper()
        if not symbol:
            continue

        stock_obj, _ = PennyStockUniverse.objects.update_or_create(
            symbol=symbol,
            defaults={
                'name': (item.get('companyName') or item.get('name') or '').strip(),
                'exchange': (item.get('exchangeShortName') or item.get('exchange') or '').strip(),
                'sector': (item.get('sector') or '').strip(),
                'industry': (item.get('industry') or '').strip(),
                'price': _safe_float(item.get('price')),
                'market_cap': _safe_float(item.get('marketCap')),
                'volume': _safe_float(item.get('volume')),
                'raw': _sanitize_json(item),
            },
        )

        income = _fetch_json(
            f"https://financialmodelingprep.com/stable/income-statement?symbol={symbol}&limit=1&apikey={fmp_key}"
        ) or []
        balance = _fetch_json(
            f"https://financialmodelingprep.com/stable/balance-sheet-statement?symbol={symbol}&limit=1&apikey={fmp_key}"
        ) or []
        cashflow = _fetch_json(
            f"https://financialmodelingprep.com/stable/cash-flow-statement?symbol={symbol}&limit=1&apikey={fmp_key}"
        ) or []

        income_row = income[0] if isinstance(income, list) and income else {}
        balance_row = balance[0] if isinstance(balance, list) and balance else {}
        cashflow_row = cashflow[0] if isinstance(cashflow, list) and cashflow else {}

        revenue = _safe_float(income_row.get('revenue'))
        total_debt = _safe_float(balance_row.get('totalDebt'))
        if total_debt is None:
            total_debt = _safe_float(balance_row.get('shortTermDebt') or 0) + _safe_float(balance_row.get('longTermDebt') or 0)
        cash = _safe_float(
            balance_row.get('cashAndCashEquivalents')
            or balance_row.get('cashAndCashEquivalentsAtCarryingValue')
        )
        operating_cash = _safe_float(cashflow_row.get('operatingCashFlow'))
        burn_rate = None
        if operating_cash is not None and operating_cash < 0:
            burn_rate = abs(operating_cash) / 12

        share_issuance = _safe_float(cashflow_row.get('commonStockIssued')) or 0

        shares_outstanding = None
        try:
            shares_outstanding = _safe_float(yf.Ticker(symbol).info.get('sharesOutstanding'))
        except Exception:
            shares_outstanding = None

        sentiment_data = _fetch_json(
            f"https://financialmodelingprep.com/stable/news/stock?symbols={symbol}&limit=50&apikey={fmp_key}"
        ) or []
        sentiment_score = None
        if isinstance(sentiment_data, list) and sentiment_data:
            scores = [
                _safe_float(item.get('sentimentScore') or item.get('sentiment') or item.get('sentiment_score'))
                for item in sentiment_data
                if _safe_float(item.get('sentimentScore') or item.get('sentiment')) is not None
            ]
            if scores:
                sentiment_score = sum(scores) / len(scores)

        if sentiment_score is None:
            cutoff = timezone.now() - timedelta(days=30)
            news_avg = (
                StockNews.objects.filter(stock__symbol=symbol, published_at__gte=cutoff)
                .aggregate(avg=models.Avg('sentiment'))
                .get('avg')
            )
            if news_avg is not None:
                sentiment_score = float(news_avg)

        social_data = _fetch_json(
            f"https://financialmodelingprep.com/stable/social-sentiment?symbol={symbol}&apikey={fmp_key}"
        ) or []
        social_mentions = None
        if isinstance(social_data, list) and social_data:
            counts = [
                _safe_float(item.get('mention') or item.get('mentions') or item.get('totalMentions'))
                for item in social_data
                if _safe_float(item.get('mention') or item.get('mentions') or item.get('totalMentions')) is not None
            ]
            if counts:
                social_mentions = int(sum(counts))
        if social_mentions is None:
            social_mentions = int(
                StockNews.objects.filter(stock__symbol=symbol).count()
            )

        rsi = None
        macd_hist = None
        if av_key:
            rsi_payload = _fetch_json(
                "https://www.alphavantage.co/query"
                f"?function=RSI&symbol={symbol}&interval=daily&time_period=14&series_type=close&apikey={av_key}"
            ) or {}
            rsi_series = rsi_payload.get('Technical Analysis: RSI') or {}
            rsi = _latest_indicator_value(rsi_series, 'RSI')

            macd_payload = _fetch_json(
                "https://www.alphavantage.co/query"
                f"?function=MACD&symbol={symbol}&interval=daily&series_type=close&apikey={av_key}"
            ) or {}
            macd_series = macd_payload.get('Technical Analysis: MACD') or {}
            macd_hist = _latest_indicator_value(macd_series, 'MACD_Hist')

        liquidity_score = _clamp((_safe_float(item.get('volume')) or 0) / 1_000_000)
        sentiment_norm = 0.5
        if sentiment_score is not None:
            sentiment_norm = _clamp((sentiment_score + 1) / 2)

        rsi_score = 0.5
        if rsi is not None:
            rsi_score = 1 - _clamp((rsi - 30) / 40)

        macd_score = 0.5
        if macd_hist is not None:
            macd_score = 1.0 if macd_hist > 0 else 0.3

        technical_score = _clamp((rsi_score + macd_score) / 2)

        fundamental_score = 0.5
        months_of_cash = None
        if cash is not None and burn_rate:
            months_of_cash = cash / burn_rate
            fundamental_score = _clamp(months_of_cash / 6)

        dilution_penalty = 0.0
        market_cap = _safe_float(item.get('marketCap')) or 0
        if share_issuance and market_cap:
            dilution_penalty = _clamp(share_issuance / market_cap, 0.0, 0.5)

        previous_snapshot = (
            PennyStockSnapshot.objects.filter(stock=stock_obj).order_by('-as_of').first()
        )
        if shares_outstanding and previous_snapshot and previous_snapshot.shares_outstanding:
            change = (shares_outstanding - previous_snapshot.shares_outstanding) / previous_snapshot.shares_outstanding
            if change > 0:
                dilution_penalty = max(dilution_penalty, _clamp(change, 0.0, 0.5))

        ai_score = 100 * _clamp(
            (0.25 * fundamental_score)
            + (0.25 * technical_score)
            + (0.25 * sentiment_norm)
            + (0.25 * liquidity_score)
            - dilution_penalty
        )

        _, was_created = PennyStockSnapshot.objects.update_or_create(
            stock=stock_obj,
            as_of=today,
            defaults={
                'price': _safe_float(item.get('price')),
                'market_cap': market_cap or None,
                'volume': _safe_float(item.get('volume')),
                'revenue': revenue,
                'debt': total_debt,
                'cash': cash,
                'burn_rate': burn_rate,
                'shares_outstanding': shares_outstanding,
                'rsi': rsi,
                'macd_hist': macd_hist,
                'sentiment_score': sentiment_score,
                'social_mentions': social_mentions,
                'dilution_score': dilution_penalty,
                'ai_score': ai_score,
                'flags': {
                    'months_of_cash': months_of_cash,
                    'fundamental_score': fundamental_score,
                    'technical_score': technical_score,
                    'sentiment_norm': sentiment_norm,
                    'liquidity_score': liquidity_score,
                },
                'raw': _sanitize_json({
                    'income': income_row,
                    'balance': balance_row,
                    'cashflow': cashflow_row,
                }),
            },
        )

        if was_created:
            created += 1
        else:
            updated += 1

    return {'created': created, 'updated': updated, 'seen': seen}


@shared_task
def backtest_retrain_guard() -> dict[str, Any]:
    """Run AI backtest guardrail and retrain if win rate drops."""
    symbol = os.getenv('BACKTEST_SYMBOL', 'SPY').strip().upper()
    lookback_days = int(os.getenv('BACKTEST_LOOKBACK_DAYS', '60'))
    min_win_rate = float(os.getenv('BACKTEST_MIN_WIN_RATE', '52'))

    engine = DataFusionEngine(symbol)
    data = engine.fuse_all()
    if data is None or data.empty:
        return {'status': 'no_data', 'symbol': symbol}

    model_path = get_model_path('BLUECHIP')
    model = load_or_train_model(data, model_path=model_path)
    backtester = AIBacktester(data, model, symbol=symbol)
    result = backtester.run_simulation(lookback_days=lookback_days)

    retrained = False
    if result.win_rate < min_win_rate:
        model = train_fusion_model(data, model_path=model_path)
        retrained = model is not None

    return {
        'status': 'ok',
        'symbol': symbol,
        'lookback_days': lookback_days,
        'win_rate': result.win_rate,
        'min_win_rate': min_win_rate,
        'retrained': retrained,
    }


def _build_training_frame(symbol: str, lookback_days: int, news_days: int) -> pd.DataFrame:
    prev_news_days = os.environ.get('NEWS_SENTIMENT_DAYS')
    prev_use_alpaca = os.environ.get('DATAFUSION_USE_ALPACA')
    os.environ['NEWS_SENTIMENT_DAYS'] = str(news_days)
    os.environ['DATAFUSION_USE_ALPACA'] = 'true'
    try:
        engine = DataFusionEngine(symbol)
        data = engine.fuse_all()
        if data is None or data.empty:
            os.environ['DATAFUSION_USE_ALPACA'] = 'false'
            engine = DataFusionEngine(symbol)
            data = engine.fuse_all()
    finally:
        if prev_use_alpaca is None:
            os.environ.pop('DATAFUSION_USE_ALPACA', None)
        else:
            os.environ['DATAFUSION_USE_ALPACA'] = prev_use_alpaca
        if prev_news_days is None:
            os.environ.pop('NEWS_SENTIMENT_DAYS', None)
        else:
            os.environ['NEWS_SENTIMENT_DAYS'] = prev_news_days
    if data is None or data.empty:
        return pd.DataFrame()
    if len(data) > lookback_days:
        data = data.tail(lookback_days)
    return data


def _symbol_training_report(symbol: str, df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {}
    first = df.iloc[0]
    last = df.iloc[-1]
    close_first = float(first.get('Close') or 0)
    close_last = float(last.get('Close') or 0)
    change_pct = ((close_last - close_first) / close_first) * 100 if close_first else 0.0
    volume_change_pct = None
    if 'Volume' in df.columns and len(df) >= 40:
        vol_recent = float(df['Volume'].tail(20).mean() or 0)
        vol_past = float(df['Volume'].head(20).mean() or 0)
        if vol_past:
            volume_change_pct = ((vol_recent - vol_past) / vol_past) * 100

    reasons: list[str] = []
    rsi = float(last.get('RSI14') or 0)
    macd_hist = float(last.get('MACD_HIST') or 0)
    volume_z = float(last.get('VolumeZ') or 0)
    if change_pct >= 5:
        reasons.append('trend_haussier_6m')
    if change_pct <= -5:
        reasons.append('trend_baissier_6m')
    if rsi >= 70:
        reasons.append('rsi_surachat')
    elif rsi <= 30:
        reasons.append('rsi_survente')
    if macd_hist > 0:
        reasons.append('macd_positif')
    if volume_z >= 1.5:
        reasons.append('volume_en_hausse')
    candle_body_pct = float(last.get('CandleBodyPct') or 0)
    if candle_body_pct >= 0.6:
        reasons.append('candle_impulsion')

    return {
        'symbol': symbol,
        'close': close_last,
        'change_6m_pct': round(change_pct, 2),
        'volume_change_pct': round(volume_change_pct, 2) if volume_change_pct is not None else None,
        'rsi14': round(rsi, 2),
        'macd_hist': round(macd_hist, 4),
        'sentiment_score': float(last.get('sentiment_score') or 0),
        'news_count': int(last.get('news_count') or 0),
        'candle_body_pct': round(candle_body_pct, 4),
        'candle_range_pct': round(float(last.get('CandleRangePct') or 0), 4),
        'reasons': reasons,
    }


@shared_task
def nightly_closed_market_retrain() -> dict[str, Any]:
    log = _task_log_start('nightly_closed_market_retrain')
    if not _market_closed_now():
        result = {'skipped': 'market_open'}
        _task_log_finish(log, 'SUCCESS', result)
        return result

    lookback_days = int(os.getenv('NIGHTLY_TRAIN_LOOKBACK_DAYS', '60'))
    news_days = int(os.getenv('NIGHTLY_TRAIN_NEWS_DAYS', '60'))
    max_symbols = int(os.getenv('NIGHTLY_TRAIN_MAX_SYMBOLS', '1000'))

    symbols_by_universe = _collect_training_symbols()
    penny_symbols = symbols_by_universe.get('PENNY') or []
    blue_symbols = symbols_by_universe.get('BLUECHIP') or []

    if max_symbols > 0:
        penny_symbols = penny_symbols[:max_symbols]
        blue_symbols = blue_symbols[:max_symbols]

    datasets: dict[str, list[pd.DataFrame]] = {'PENNY': [], 'BLUECHIP': []}
    reports: dict[str, Any] = {}
    skipped: list[str] = []

    for symbol in penny_symbols:
        df = _build_training_frame(symbol, lookback_days, news_days)
        if df is None or df.empty:
            skipped.append(symbol)
            continue
        datasets['PENNY'].append(df)
        reports[symbol] = _symbol_training_report(symbol, df)

    for symbol in blue_symbols:
        df = _build_training_frame(symbol, lookback_days, news_days)
        if df is None or df.empty:
            skipped.append(symbol)
            continue
        datasets['BLUECHIP'].append(df)
        reports[symbol] = _symbol_training_report(symbol, df)

    model_updates: dict[str, Any] = {}
    for universe, frames in datasets.items():
        if not frames:
            continue
        merged = pd.concat(frames, ignore_index=True)
        payload = train_fusion_model(merged, model_path=get_model_path(universe))
        if payload:
            model_updates[universe] = {
                'model_version': payload.get('model_version'),
                'samples': int(len(merged)),
                'cv_mean': payload.get('cv_mean'),
                'features': payload.get('features') or [],
            }

    report_payload = {
        'as_of': timezone.now().isoformat(),
        'lookback_days': lookback_days,
        'news_days': news_days,
        'symbol_counts': {
            'penny': len(penny_symbols),
            'bluechip': len(blue_symbols),
            'skipped': len(skipped),
        },
        'model_updates': model_updates,
        'reports': reports,
    }

    try:
        report_dir = Path(_journal_output_dir())
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"nightly_retrain_{timezone.now().date().isoformat()}.json"
        report_path.write_text(json.dumps(report_payload, indent=2, default=str))
        report_payload['report_path'] = str(report_path)
    except Exception:
        pass

    _task_log_finish(log, 'SUCCESS', report_payload)
    return report_payload


def _env_float(prefix: str, name: str, default: str) -> float:
    return float(os.getenv(f'{prefix}_{name}', os.getenv(f'PAPER_{name}', default)))


def _sandbox_env_float(sandbox: str, name: str, default: str) -> float:
    key = f'{sandbox}_{name}'
    return float(os.getenv(key, os.getenv(name, default)))


def _sandbox_env_int(sandbox: str, name: str, default: str) -> int:
    key = f'{sandbox}_{name}'
    return int(os.getenv(key, os.getenv(name, default)))


def _env_list(prefix: str, name: str, default: str) -> list[str]:
    raw = os.getenv(f'{prefix}_{name}', os.getenv(f'PAPER_{name}', default))
    items = [s.strip().upper() for s in str(raw).split(',') if s.strip()]
    return items


def _get_watchlist(sandbox: str, prefix: str, default: str) -> list[str]:
    stored = SandboxWatchlist.objects.filter(sandbox=sandbox).first()
    if stored and stored.symbols:
        return [str(s).strip().upper() for s in stored.symbols if str(s).strip()]
    return _env_list(prefix, 'WATCHLIST', default)


def _coerce_date(value: Any) -> date | None:
    if value is None:
        return None
    try:
        if isinstance(value, pd.Series):
            if value.empty:
                return None
            value = value.iloc[0]
        elif isinstance(value, (list, tuple)):
            if not value:
                return None
            value = value[0]
        parsed = pd.to_datetime(value, errors='coerce')
        if parsed is None or pd.isna(parsed):
            return None
        if hasattr(parsed, 'to_pydatetime'):
            parsed = parsed.to_pydatetime()
        if isinstance(parsed, datetime):
            return parsed.date()
        if isinstance(parsed, date):
            return parsed
    except Exception:
        return None
    return None


def _get_earnings_date(ticker: yf.Ticker) -> date | None:
    try:
        calendar = ticker.calendar
        if calendar is None:
            return None
        if isinstance(calendar, pd.DataFrame):
            if 'Earnings Date' in calendar.index:
                return _coerce_date(calendar.loc['Earnings Date'][0])
            if 'Earnings Date' in calendar.columns:
                return _coerce_date(calendar['Earnings Date'].iloc[0])
        if isinstance(calendar, dict):
            return _coerce_date(calendar.get('Earnings Date'))
    except Exception:
        return None
    return None


def _is_blacklisted(symbol: str) -> tuple[bool, str]:
    try:
        ticker = yf.Ticker(symbol)
    except Exception:
        return False, ''

    try:
        earnings_date = _get_earnings_date(ticker)
        if earnings_date:
            cutoff = (timezone.now() + timedelta(days=2)).date()
            if earnings_date <= cutoff:
                return True, 'Earnings upcoming'
    except Exception:
        pass

    try:
        hist = ticker.history(period='2d', interval='1d', timeout=10)
        if hist is not None and len(hist) >= 2 and 'Close' in hist:
            prev = _safe_float(hist['Close'].iloc[-2])
            last = _safe_float(hist['Close'].iloc[-1])
            if prev and last:
                change = abs((last - prev) / prev)
                if change > 0.15:
                    return True, 'High Volatility Spike'
    except Exception:
        pass

    return False, ''


def _earnings_blackout(symbol: str, days: int = 2) -> tuple[bool, date | None]:
    try:
        ticker = yf.Ticker(symbol)
    except Exception:
        return False, None
    try:
        earnings_date = _get_earnings_date(ticker)
        if not earnings_date:
            return False, None
        cutoff = (timezone.now() + timedelta(days=days)).date()
        return earnings_date <= cutoff, earnings_date
    except Exception:
        return False, None


def _daily_trend_ok(symbol: str, use_alpaca: bool = False) -> bool:
    try:
        if use_alpaca:
            hist = get_daily_bars(symbol, days=200)
            if hist is None or hist.empty or 'close' not in hist:
                return True
            close = hist['close']
        else:
            hist = yf.Ticker(symbol).history(period='1y', interval='1d', timeout=10)
            if hist is None or hist.empty or 'Close' not in hist:
                return True
            close = hist['Close']
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        last = float(close.iloc[-1]) if len(close) else 0.0
        e20 = float(ema20.iloc[-1]) if len(ema20) else 0.0
        e50 = float(ema50.iloc[-1]) if len(ema50) else 0.0
        e200 = float(ema200.iloc[-1]) if len(ema200) else 0.0
        e200_prev = float(ema200.iloc[-6]) if len(ema200) >= 6 else e200
        if last <= 0 or e20 <= 0 or e50 <= 0:
            return True
        slope_ok = e200_prev <= 0 or e200 >= e200_prev
        return last >= e20 and e20 >= e50 and (e200 <= 0 or e50 >= e200) and slope_ok
    except Exception:
        return True


def _weak_list_health() -> dict[str, Any]:
    cache_key = 'weak_list_health'
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    symbols = list(MasterWatchlistEntry.objects.filter(category='WEAK_SHORT').values_list('symbol', flat=True))
    symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
    if not symbols:
        payload = {'status': 'empty', 'ratio_down': 0.0, 'count': 0, 'defensive': False}
        cache.set(cache_key, payload, timeout=60 * 30)
        return payload
    down = 0
    seen = 0
    for symbol in symbols[:200]:
        try:
            hist = yf.Ticker(symbol).history(period='2d', interval='1d', timeout=10)
            if hist is None or hist.empty or 'Close' not in hist or len(hist) < 2:
                continue
            prev = _safe_float(hist['Close'].iloc[-2])
            last = _safe_float(hist['Close'].iloc[-1])
            if not prev or not last:
                continue
            seen += 1
            if last < prev:
                down += 1
        except Exception:
            continue
    ratio_down = (down / seen) if seen else 0.0
    threshold = float(os.getenv('WEAK_LIST_DEFENSIVE_THRESHOLD', '0.8'))
    payload = {
        'status': 'ok',
        'ratio_down': round(ratio_down, 3),
        'count': seen,
        'defensive': bool(seen >= 10 and ratio_down >= threshold),
    }
    cache.set(cache_key, payload, timeout=60 * 30)
    return payload


def _has_hammer_pattern(ctx: dict[str, Any] | None) -> bool:
    if not ctx:
        return False
    patterns = ctx.get('patterns') or []
    if isinstance(patterns, list):
        return any('hammer' in str(p).lower() for p in patterns)
    return False


def _rsi_divergence(ctx: dict[str, Any] | None, lookback: int = 6) -> bool:
    if not ctx:
        return False
    bars = ctx.get('bars')
    if bars is None or bars.empty or 'close' not in bars:
        return False
    closes = pd.to_numeric(bars['close'], errors='coerce').dropna()
    if len(closes) < lookback + 1:
        return False
    series = closes.tail(lookback + 1)
    low_recent = float(series.iloc[-1])
    low_prev = float(series.iloc[:-1].min())
    if low_recent >= low_prev:
        return False
    rsi_series = _compute_rsi(series)
    if rsi_series is None or rsi_series.empty:
        return False
    rsi_recent = float(rsi_series.iloc[-1])
    rsi_prev = float(rsi_series.iloc[:-1].min())
    return rsi_recent > rsi_prev


@lru_cache(maxsize=512)
def _correlation_group(symbol: str) -> str | None:
    overrides = _parse_driver_map(os.getenv('CORRELATION_GROUP_OVERRIDES'))
    sym = (symbol or '').strip().upper()
    if sym in overrides:
        return overrides[sym]
    try:
        info = yf.Ticker(sym).info or {}
    except Exception:
        info = {}
    haystack = ' '.join([
        str(info.get('sector') or ''),
        str(info.get('industry') or ''),
        str(info.get('longName') or ''),
        str(info.get('shortName') or ''),
    ]).lower()
    if any(k in haystack for k in ['gold', 'silver', 'mining']):
        return 'GOLD'
    if any(k in haystack for k in ['oil', 'gas', 'energy']):
        return 'ENERGY'
    if any(k in haystack for k in ['bank', 'financial', 'insurance']):
        return 'FINANCIAL'
    if any(k in haystack for k in ['software', 'semiconductor', 'tech']):
        return 'TECH'
    return None


def _correlation_blocked(symbol: str, open_symbols: list[str]) -> bool:
    group = _correlation_group(symbol)
    if not group:
        return False
    max_per_group = int(os.getenv('CORRELATION_MAX_PER_GROUP', '2'))
    count = 0
    for sym in open_symbols:
        if _correlation_group(sym) == group:
            count += 1
    return count >= max_per_group


def _spread_too_wide(symbol: str, max_pct: float) -> bool:
    spread_pct = get_latest_bid_ask_spread_pct(symbol)
    if spread_pct is None:
        bid, ask = _latest_bid_ask(symbol)
        if bid and ask and ask > 0:
            spread_pct = float((ask - bid) / ((ask + bid) / 2))
    if spread_pct is None:
        return False
    return spread_pct > max_pct


def _pump_dump_risk(intraday_ctx: dict[str, Any] | None, sentiment: float | None) -> bool:
    if not intraday_ctx:
        return False
    rvol = float(intraday_ctx.get('rvol') or 0.0)
    if rvol <= 0:
        return False
    pump_rvol = float(os.getenv('PENNY_PUMP_DUMP_RVOL', '10'))
    euphoric = float(os.getenv('PENNY_PUMP_DUMP_SENTIMENT', '0.7'))
    if rvol < pump_rvol:
        return False
    if sentiment is None:
        return False
    return sentiment >= euphoric


def _vwap_filter_block(intraday_ctx: dict[str, Any] | None) -> bool:
    if not intraday_ctx:
        return False
    if os.getenv('VWAP_FILTER_ENABLED', 'true').lower() not in {'1', 'true', 'yes', 'y'}:
        return False
    return float(intraday_ctx.get('price_to_vwap') or 0.0) < 0.0


def _time_of_day_penalty(now: datetime | None = None) -> float:
    now = now or _ny_time_now()
    if os.getenv('MIDDAY_SIGNAL_PENALTY_ENABLED', 'true').lower() not in {'1', 'true', 'yes', 'y'}:
        return 1.0
    if (now.hour == 11 and now.minute >= 30) or (now.hour == 12) or (now.hour == 13 and now.minute < 30):
        return float(os.getenv('MIDDAY_SIGNAL_PENALTY', '0.9'))
    return 1.0


def _atr_spike(symbol: str, use_alpaca: bool = False) -> bool:
    spike_mult = float(os.getenv('ATR_SPIKE_MULT', '2.0'))
    try:
        if use_alpaca:
            hist = get_daily_bars(symbol, days=40)
            if hist is None or hist.empty or not {'high', 'low', 'close'}.issubset(hist.columns):
                return False
            high = hist['high']
            low = hist['low']
            close = hist['close']
        else:
            hist = yf.Ticker(symbol).history(period='3mo', interval='1d', timeout=10)
            if hist is None or hist.empty or not {'High', 'Low', 'Close'}.issubset(hist.columns):
                return False
            high = hist['High']
            low = hist['Low']
            close = hist['Close']
        tr = pd.concat([
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        latest = float(atr.iloc[-1]) if len(atr) else 0.0
        avg = float(atr.tail(10).mean()) if len(atr) >= 10 else float(atr.mean())
        return avg > 0 and latest >= avg * spike_mult
    except Exception:
        return False


def _is_halted(intraday_ctx: dict[str, Any] | None) -> bool:
    if not intraday_ctx:
        return False
    bars = intraday_ctx.get('bars')
    if bars is None or bars.empty or 'volume' not in bars:
        return False
    tail = bars.tail(2)
    if tail.empty:
        return False
    return float(tail['volume'].iloc[-1]) == 0.0 and float(tail['volume'].iloc[-2]) == 0.0


def _flash_crash(intraday_ctx: dict[str, Any] | None) -> bool:
    if not intraday_ctx:
        return False
    bars = intraday_ctx.get('bars')
    if bars is None or bars.empty or 'close' not in bars:
        return False
    tail = bars.tail(3)
    if len(tail) < 3:
        return False
    first = float(tail['close'].iloc[0] or 0.0)
    last = float(tail['close'].iloc[-1] or 0.0)
    if first <= 0:
        return False
    drop = (last - first) / first
    return drop <= float(os.getenv('FLASH_CRASH_PCT', '-0.10'))


def _btc_trend_ok(symbol: str) -> bool:
    symbols = [s.strip().upper() for s in os.getenv('BTC_CORR_SYMBOLS', 'HIVE,HUT,RIOT,MARA,BITF,CLSK,MSTR,HIVE.V,BITF.TO').split(',') if s.strip()]
    sym = (symbol or '').strip().upper()
    if not symbols or sym not in symbols:
        return True
    try:
        minutes = int(os.getenv('BTC_TREND_MINUTES', '60'))
        btc = get_intraday_bars('BTC-USD', minutes=minutes)
        if btc is None or btc.empty or 'close' not in btc:
            return True
        close = btc['close']
        ema_fast = int(os.getenv('BTC_TREND_EMA_FAST', '20'))
        ema_slow = int(os.getenv('BTC_TREND_EMA_SLOW', '50'))
        ema20 = close.ewm(span=ema_fast, adjust=False).mean()
        ema50 = close.ewm(span=ema_slow, adjust=False).mean()
        return float(ema20.iloc[-1]) >= float(ema50.iloc[-1])
    except Exception:
        return True


def _value_hunter_watchlist() -> list[str]:
    raw = os.getenv(
        'VALUE_HUNTER_WATCHLIST',
        'ATD.TO,DOL.TO,L.TO,RY.TO,TD.TO,BNS.TO,BMO.TO,CNR.TO,CP.TO,TFII.TO,ENB.TO,TRP.TO,FTS.TO,CSU.TO,BN.TO,GIB-A.TO,T.TO,BCE.TO,SAP.TO,MRU.TO',
    )
    return [s.strip().upper() for s in raw.split(',') if s.strip()]


def _check_bluechip_health(symbol: str) -> tuple[bool, dict[str, Any]]:
    try:
        info = yfin.Ticker(symbol).info or {}
    except Exception:
        info = {}
    fcf = info.get('freeCashflow') or 0
    debt_to_equity = (info.get('debtToEquity') or 0) / 100
    profit_margins = info.get('profitMargins') or 0
    min_fcf = float(os.getenv('VALUE_HUNTER_MIN_FCF', '0'))
    max_de = float(os.getenv('VALUE_HUNTER_MAX_DEBT_TO_EQUITY', '1.5'))
    min_margin = float(os.getenv('VALUE_HUNTER_MIN_PROFIT_MARGINS', '0.05'))
    ok = bool(fcf and fcf > min_fcf and debt_to_equity < max_de and profit_margins > min_margin)
    return ok, {
        'free_cashflow': float(fcf or 0),
        'debt_to_equity': float(debt_to_equity or 0),
        'profit_margins': float(profit_margins or 0),
    }


def _value_hunter_candidate(symbol: str) -> dict[str, Any] | None:
    symbol = (symbol or '').strip().upper()
    if not symbol:
        return None
    try:
        info = yfin.Ticker(symbol).info or {}
    except Exception:
        info = {}
    market_cap = info.get('marketCap') or 0
    try:
        market_cap = float(market_cap)
    except Exception:
        market_cap = 0
    if market_cap < float(os.getenv('VALUE_HUNTER_MIN_MARKET_CAP', '10000000000')):
        return None

    try:
        hist = yfin.Ticker(symbol).history(period='1y', interval='1d')
    except Exception:
        hist = None
    close = _extract_close_series(hist) if hist is not None else None
    if close is None or close.empty:
        return None
    price = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) > 1 else price
    drop_pct = ((price - prev_close) / prev_close) * 100 if prev_close else 0.0
    rsi = _compute_rsi(close, 14)
    if rsi is None:
        return None
    ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
    if not ma200 or ma200 <= 0:
        return None
    distance = ((ma200 - price) / ma200) * 100
    if distance < float(os.getenv('VALUE_HUNTER_MIN_DISTANCE_MA200_PCT', '10')):
        return None
    rsi_trigger = float(os.getenv('VALUE_HUNTER_MAX_RSI', '35'))
    alert_drop = float(os.getenv('VALUE_HUNTER_DROP_ALERT_PCT', '-3'))
    if drop_pct > alert_drop and rsi > float(os.getenv('VALUE_HUNTER_RSI_ALERT', '30')):
        return None
    if rsi > rsi_trigger:
        return None
    sentiment, _ = _news_sentiment_score(symbol, days=3)
    if sentiment < float(os.getenv('VALUE_HUNTER_MIN_SENTIMENT', '-0.2')):
        return None
    health_ok, health_meta = _check_bluechip_health(symbol)
    if not health_ok:
        return None

    return {
        'symbol': symbol,
        'price': round(price, 4),
        'rsi': round(float(rsi), 2),
        'distance_ma200': round(float(distance), 2),
        'market_cap': market_cap,
        'sentiment': round(float(sentiment), 3),
        'drop_pct': round(float(drop_pct), 2),
        **health_meta,
    }


@shared_task
def value_hunter_scan(send_alerts: bool = True) -> dict[str, Any]:
    watchlist = _value_hunter_watchlist()
    if not watchlist:
        return {'status': 'empty', 'count': 0}
    results: list[dict[str, Any]] = []
    for symbol in watchlist:
        candidate = _value_hunter_candidate(symbol)
        if candidate:
            results.append(candidate)

    results.sort(key=lambda x: (x.get('rsi', 100), -x.get('distance_ma200', 0)))
    cache.set('value_hunter_candidates', results, timeout=60 * 60)

    if send_alerts and results:
        lines = ["💎 MODE VALUE — AUBAINES BLUECHIP"]
        for item in results[:5]:
            lines.append(
                f"🚨 {item['symbol']} RSI {item['rsi']} | -{item['distance_ma200']}% MA200"
            )
        _send_telegram_alert("\n".join(lines), allow_during_blackout=True, category='report')

    return {'status': 'ok', 'count': len(results), 'results': results[:10]}


@shared_task
def send_value_hunter_report() -> dict[str, Any]:
    aggressive = cache.get('market_scanner_results') or []
    value_candidates = cache.get('value_hunter_candidates') or []

    lines = ["📊 *Rapport Opportunités*", "", "🔥 MODE AGRESSIF"]
    if aggressive:
        for item in aggressive[:5]:
            lines.append(f"• {item.get('symbol')}: score {float(item.get('score') or 0):.2f} RVOL {item.get('rvol')}")
    else:
        lines.append('— Aucun signal momentum')

    lines.append("")
    lines.append("💎 MODE VALUE")
    if value_candidates:
        for item in value_candidates[:5]:
            lines.append(f"• {item.get('symbol')}: RSI {item.get('rsi')} | -{item.get('distance_ma200')}% MA200")
    else:
        lines.append('— Aucun rabais solide')

    _send_telegram_alert("\n".join(lines), allow_during_blackout=True, category='report')
    return {'status': 'sent', 'aggressive': len(aggressive), 'value': len(value_candidates)}


def _midday_blackout() -> bool:
    if os.getenv('MIDDAY_NO_TRADE_ENABLED', 'true').lower() not in {'1', 'true', 'yes', 'y'}:
        return False
    now_ny = _ny_time_now()
    start_raw = os.getenv('MIDDAY_START', '11:45').strip()
    end_raw = os.getenv('MIDDAY_END', '13:15').strip()
    try:
        start_h, start_m = [int(x) for x in start_raw.split(':', 1)]
        end_h, end_m = [int(x) for x in end_raw.split(':', 1)]
        start = dt_time(start_h, start_m)
        end = dt_time(end_h, end_m)
    except Exception:
        start = dt_time(11, 45)
        end = dt_time(13, 15)
    return start <= now_ny.time() <= end


def _price_velocity_drop(intraday_ctx: dict[str, Any] | None, bars_back: int = 5) -> bool:
    if not intraday_ctx:
        return False
    bars = intraday_ctx.get('bars')
    if bars is None or bars.empty or 'close' not in bars:
        return False
    if len(bars) <= bars_back:
        return False
    start = float(bars['close'].iloc[-1 - bars_back] or 0.0)
    last = float(bars['close'].iloc[-1] or 0.0)
    if start <= 0:
        return False
    drop = (last - start) / start
    threshold = float(os.getenv('PRICE_VELOCITY_DROP_PCT', '-0.03'))
    return drop <= threshold


def _daily_equity_circuit_breaker(sandbox: str, equity_now: float) -> dict[str, Any]:
    threshold = float(os.getenv('DAILY_EQUITY_DRAWDOWN_PCT', '0.03'))
    result = {
        'triggered': False,
        'first_trigger': False,
        'baseline': None,
        'drawdown': 0.0,
        'threshold': threshold,
    }
    if equity_now <= 0:
        return result
    key = f"daily_equity_base:{sandbox}:{_ny_time_now().strftime('%Y%m%d')}"
    trigger_key = f"daily_equity_trip:{sandbox}:{_ny_time_now().strftime('%Y%m%d')}"
    baseline = cache.get(key)
    if baseline is None:
        cache.set(key, float(equity_now), timeout=60 * 60 * 24)
        result['baseline'] = float(equity_now)
        return result
    baseline = float(baseline)
    result['baseline'] = baseline
    if baseline <= 0:
        return result
    if cache.get(trigger_key):
        result['triggered'] = True
        return result
    if equity_now >= baseline:
        cache.set(key, float(equity_now), timeout=60 * 60 * 24)
        result['baseline'] = float(equity_now)
        result['drawdown'] = 0.0
        return result
    drawdown = (equity_now - baseline) / baseline
    result['drawdown'] = drawdown
    if drawdown <= -abs(threshold):
        cache.set(trigger_key, True, timeout=60 * 60 * 24)
        result['triggered'] = True
        result['first_trigger'] = True
    return result


def reset_daily_equity_breaker(sandbox: str | None = None, day: date | None = None) -> dict[str, Any]:
    day_key = (day or _ny_time_now().date()).strftime('%Y%m%d')
    sandboxes = [sandbox] if sandbox else ['AI_PENNY', 'AI_BLUECHIP', 'WATCHLIST']
    cleared = []
    for box in sandboxes:
        if not box:
            continue
        trigger_key = f"daily_equity_trip:{box}:{day_key}"
        baseline_key = f"daily_equity_base:{box}:{day_key}"
        if cache.get(trigger_key):
            cache.delete(trigger_key)
            cleared.append(box)
        if cache.get(baseline_key):
            cache.delete(baseline_key)
    return {'cleared': cleared, 'date': day_key}


def _reentry_cache_key(symbol: str, sandbox: str) -> str:
    day_key = _ny_time_now().strftime('%Y%m%d')
    return f"reentry_watch:{sandbox}:{symbol}:{day_key}"


def _mark_reentry(symbol: str, sandbox: str) -> None:
    cache.set(_reentry_cache_key(symbol, sandbox), True, timeout=60 * 60 * 6)


def _is_reentry_candidate(symbol: str, sandbox: str) -> bool:
    return bool(cache.get(_reentry_cache_key(symbol, sandbox)))


def _reentry_confirmed(intraday_ctx: dict[str, Any] | None) -> bool:
    if not intraday_ctx:
        return False
    confirm_pct = float(os.getenv('REENTRY_CONFIRMATION_PCT', '0.005'))
    price_to_vwap = float(intraday_ctx.get('price_to_vwap') or 0.0)
    return price_to_vwap >= confirm_pct


def _decision_log(symbol: str, sandbox: str, status: str, reason: str, signal: float | None = None) -> None:
    if os.getenv('DECISION_LOG_ENABLED', 'false').lower() not in {'1', 'true', 'yes', 'y'}:
        return
    path = os.getenv('DECISION_LOG_PATH', 'logs/decision_log.csv')
    try:
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        line = f"{timezone.now().isoformat()},{sandbox},{symbol},{status},{signal if signal is not None else ''},{reason}\n"
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write(line)
    except Exception:
        return


def _decision_journal_tail(limit: int = 5) -> list[dict[str, Any]]:
    path = os.getenv('DECISION_LOG_PATH', 'logs/decision_log.csv')
    try:
        if not os.path.exists(path):
            return []
        with open(path, 'r', encoding='utf-8') as handle:
            lines = handle.readlines()
        lines = [line.strip() for line in lines if line.strip()]
        tail = lines[-limit:]
        entries = []
        for line in tail:
            parts = line.split(',', 5)
            if len(parts) < 6:
                continue
            entries.append({
                'ts': parts[0],
                'sandbox': parts[1],
                'symbol': parts[2],
                'status': parts[3],
                'signal': parts[4],
                'reason': parts[5],
            })
        return entries
    except Exception:
        return []


def _ai_scan_cache_key(day: date | None = None) -> str:
    day_key = (day or _ny_time_now().date()).strftime('%Y%m%d')
    return f"ai_scan_summary:{day_key}"


def _record_ai_scan(symbol: str, confidence: float | None, source: str) -> None:
    symbol = (symbol or '').strip().upper()
    if not symbol:
        return
    normalized = None
    try:
        if confidence is not None:
            normalized = float(confidence)
            if normalized <= 1:
                normalized *= 100
    except Exception:
        normalized = None
    entry = {
        'symbol': symbol,
        'confidence': normalized,
        'source': (source or '').strip() or 'unknown',
        'ts': timezone.now().isoformat(),
    }
    key = _ai_scan_cache_key()
    existing = cache.get(key)
    if not isinstance(existing, list):
        existing = []
    existing.append(entry)
    cache.set(key, existing[-250:], timeout=60 * 60 * 24)


def _get_ai_scan_summary(day: date | None = None) -> list[dict[str, Any]]:
    key = _ai_scan_cache_key(day)
    entries = cache.get(key)
    if not isinstance(entries, list):
        return []
    cleaned: dict[str, dict[str, Any]] = {}
    for entry in entries:
        symbol = (entry.get('symbol') or '').strip().upper()
        if not symbol:
            continue
        confidence = entry.get('confidence')
        try:
            confidence = float(confidence) if confidence is not None else None
        except Exception:
            confidence = None
        current = cleaned.get(symbol)
        if current is None or (confidence is not None and (current.get('confidence') is None or confidence > current.get('confidence'))):
            cleaned[symbol] = {
                'symbol': symbol,
                'confidence': confidence,
                'source': entry.get('source') or 'unknown',
            }
    return list(cleaned.values())


def _system_log(
    category: str,
    level: str,
    message: str,
    symbol: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    try:
        SystemLog.objects.create(
            category=category,
            level=level,
            symbol=(symbol or None),
            message=message,
            metadata=metadata or {},
        )
    except Exception:
        return


def _canadian_boost(score: float, symbol: str) -> float:
    if os.getenv('PREFER_CANADIAN_SYMBOLS', 'false').lower() not in {'1', 'true', 'yes', 'y'}:
        return score
    boost = float(os.getenv('CANADIAN_SCORE_BOOST', '0.03'))
    symbol = (symbol or '').strip().upper()
    if symbol.endswith(('.TO', '.V', '.CN')):
        return min(1.0, score + boost)
    return score


def _prediction_log(symbol: str, universe: str, score: float) -> None:
    path = os.getenv('PREDICTIONS_LOG_PATH', 'logs/predictions.csv')
    try:
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        line = f"{timezone.now().isoformat()},{universe},{symbol},{score:.4f}\n"
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write(line)
    except Exception:
        return


def _kelly_fraction(sandbox: str, lookback_days: int = 120) -> float:
    if os.getenv('KELLY_ENABLED', 'false').lower() not in {'1', 'true', 'yes', 'y'}:
        return 0.0
    min_trades = int(os.getenv('KELLY_MIN_TRADES', '20'))
    max_frac = float(os.getenv('KELLY_MAX_FRACTION', '0.1'))
    cutoff = timezone.now() - timedelta(days=lookback_days)
    trades = PaperTrade.objects.filter(sandbox=sandbox, status='CLOSED', exit_date__gte=cutoff)
    if trades.count() < min_trades:
        return 0.0
    wins = trades.filter(outcome='WIN')
    losses = trades.filter(outcome='LOSS')
    win_rate = wins.count() / trades.count() if trades.count() else 0.0
    avg_win = float(wins.aggregate(avg=models.Avg('pnl')).get('avg') or 0.0)
    avg_loss = abs(float(losses.aggregate(avg=models.Avg('pnl')).get('avg') or 0.0))
    if avg_loss <= 0:
        return 0.0
    b = avg_win / avg_loss if avg_loss else 0.0
    f = win_rate - ((1 - win_rate) / b) if b > 0 else 0.0
    return max(0.0, min(max_frac, f))


def _loss_blacklist(symbol: str, sandbox: str) -> bool:
    if os.getenv('LOSS_BLACKLIST_ENABLED', 'true').lower() not in {'1', 'true', 'yes', 'y'}:
        return False
    days = int(os.getenv('LOSS_BLACKLIST_DAYS', '7'))
    max_losses = int(os.getenv('LOSS_BLACKLIST_TRADES', '3'))
    cutoff = timezone.now() - timedelta(days=days)
    losses = PaperTrade.objects.filter(
        ticker__iexact=symbol,
        sandbox=sandbox,
        status='CLOSED',
        outcome='LOSS',
        exit_date__gte=cutoff,
    ).count()
    return losses >= max_losses


def _sector_trend_ok(symbol: str) -> bool:
    if os.getenv('SECTOR_TREND_FILTER_ENABLED', 'false').lower() not in {'1', 'true', 'yes', 'y'}:
        return True
    mapping = {
        'TECH': 'XLK',
        'ENERGY': 'XLE',
        'FINANCIAL': 'XLF',
        'GOLD': 'GLD',
    }
    group = _correlation_group(symbol)
    etf = mapping.get(group)
    if not etf:
        return True
    try:
        hist = yfin.Ticker(etf).history(period='2d', interval='1d', timeout=10)
        if hist is None or hist.empty or 'Close' not in hist or len(hist) < 2:
            return True
        prev = float(hist['Close'].iloc[-2])
        last = float(hist['Close'].iloc[-1])
        if prev <= 0:
            return True
        change_pct = ((last - prev) / prev) * 100
        min_pct = float(os.getenv('SECTOR_TREND_MIN_PCT', '-1.0'))
        return change_pct >= min_pct
    except Exception:
        return True


def _multi_model_boost(symbol: str, current_signal: float, universe: str, use_alpaca: bool) -> float:
    if os.getenv('MULTI_MODEL_BOOST_ENABLED', 'false').lower() not in {'1', 'true', 'yes', 'y'}:
        return 1.0
    other_universe = 'BLUECHIP' if universe == 'PENNY' else 'PENNY'
    try:
        other_payload = _model_signal(symbol, other_universe, get_model_path(other_universe), use_alpaca=use_alpaca)
        other_signal = float(other_payload.get('signal') or 0.0) if other_payload else 0.0
        threshold = float(os.getenv('MULTI_MODEL_BOOST_MIN', '0.8'))
        if current_signal >= threshold and other_signal >= threshold:
            return float(os.getenv('MULTI_MODEL_BOOST_FACTOR', '1.2'))
    except Exception:
        return 1.0
    return 1.0


def _reddit_hype_risk(symbol: str) -> bool:
    if os.getenv('PENNY_REDDIT_HYPE_FILTER', 'false').lower() not in {'1', 'true', 'yes', 'y'}:
        return False
    latest = PennySignal.objects.filter(symbol__iexact=symbol).order_by('-as_of').first()
    if not latest:
        return False
    hype = float(latest.hype_score or 0.0)
    max_hype = float(os.getenv('PENNY_REDDIT_HYPE_MAX', '0.9'))
    return hype >= max_hype


def _get_vix_level() -> float | None:
    try:
        hist = yf.Ticker('^VIX').history(period='5d', interval='1d', timeout=10)
        if hist is not None and not hist.empty and 'Close' in hist:
            return _safe_float(hist['Close'].iloc[-1])
    except Exception:
        return None
    return None


def get_market_sentiment() -> tuple[str, dict[str, float | None]]:
    def _index_change(symbol: str) -> float | None:
        try:
            hist = yf.Ticker(symbol).history(period='2d', interval='1d', timeout=10)
            if hist is None or hist.empty or 'Close' not in hist or len(hist) < 2:
                return None
            prev = _safe_float(hist['Close'].iloc[-2])
            last = _safe_float(hist['Close'].iloc[-1])
            if not prev or not last:
                return None
            return ((last - prev) / prev) * 100
        except Exception:
            return None

    spy_change = _index_change('SPY')
    tsx_change = _index_change('^GSPTSE')
    vix_level = _get_vix_level()

    bear_threshold = float(os.getenv('MARKET_BEAR_THRESHOLD', '-1.5'))
    bull_threshold = float(os.getenv('MARKET_BULL_THRESHOLD', '1.5'))
    vix_caution = float(os.getenv('MARKET_VIX_CAUTION', '25'))

    if (spy_change is not None and spy_change <= bear_threshold) or (
        tsx_change is not None and tsx_change <= bear_threshold
    ):
        sentiment = 'BEARISH'
    elif vix_level is not None and vix_level >= vix_caution:
        sentiment = 'CAUTION'
    elif (spy_change is not None and spy_change >= bull_threshold) or (
        tsx_change is not None and tsx_change >= bull_threshold
    ):
        sentiment = 'BULLISH'
    else:
        sentiment = 'NEUTRAL'

    return sentiment, {
        'spy_change': None if spy_change is None else round(float(spy_change), 2),
        'tsx_change': None if tsx_change is None else round(float(tsx_change), 2),
        'vix': None if vix_level is None else round(float(vix_level), 2),
    }


def _model_signal(
    symbol: str,
    universe: str,
    model_path: str | Path,
    use_alpaca: bool = False,
) -> dict[str, Any] | None:
    try:
        fusion = DataFusionEngine(symbol, use_alpaca=use_alpaca)
        fusion_df = fusion.fuse_all()
        if fusion_df is None or fusion_df.empty:
            return None
        payload = load_or_train_model(fusion_df, model_path=model_path)
        if not payload or not payload.get('model'):
            return None
        last_row = fusion_df.tail(1).copy()
        feature_list = payload.get('features') or []
        for col in feature_list:
            if col not in last_row.columns:
                last_row[col] = 0.0
        features = last_row[feature_list].fillna(0).values
        signal = float(payload['model'].predict_proba(features)[0][1])
        signal = apply_feature_weighting_to_signal(signal, last_row.iloc[0], symbol)
        feature_snapshot = {col: float(last_row.iloc[0].get(col, 0.0)) for col in FEATURE_COLUMNS}
        explanations = []
        try:
            importances = getattr(payload['model'], 'feature_importances_', None)
            if importances is not None and len(importances) == len(feature_list):
                weights = []
                for i, col in enumerate(feature_list):
                    value = float(last_row.iloc[0].get(col, 0.0))
                    weight = abs(value) * float(importances[i])
                    weights.append((col, value, weight))
                total = sum([w[2] for w in weights]) or 1.0
                top_n = int(os.getenv('TRADE_EXPLANATION_TOP_N', '5'))
                explanations = [
                    {
                        'feature': col,
                        'value': value,
                        'contribution': round((weight / total) * 100, 2),
                    }
                    for col, value, weight in sorted(weights, key=lambda item: item[2], reverse=True)[:top_n]
                ]
        except Exception:
            explanations = []
        return {
            'signal': signal,
            'features': feature_snapshot,
            'explanations': explanations,
            'model_version': get_model_version(payload, model_path),
            'model_name': universe,
        }
    except Exception:
        return None


def _stop_loss_multiplier(explanations: list[dict[str, Any]] | None) -> float:
    if not explanations:
        return 1.0
    try:
        contributions = [float(item.get('contribution') or 0.0) for item in explanations if isinstance(item, dict)]
        total = sum(contributions) if contributions else 0.0
        if total >= 80:
            return 1.15
        if total >= 60:
            return 1.05
        if total <= 30:
            return 0.85
        if total <= 45:
            return 0.95
        return 1.0
    except Exception:
        return 1.0


def _add_candlestick_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    df = frame.copy()
    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            return pd.DataFrame()
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    if df.empty:
        return pd.DataFrame()

    df['pattern_doji'] = False
    df['pattern_hammer'] = False
    df['pattern_engulfing'] = False
    df['pattern_morning_star'] = False

    try:
        import pandas_ta as ta
        patterns = ta.cdl_pattern(
            open_=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=["doji", "hammer", "engulfing", "morningstar"],
        )
        if patterns is not None and not patterns.empty:
            if 'CDL_DOJI' in patterns:
                df['pattern_doji'] = patterns['CDL_DOJI'] != 0
            if 'CDL_HAMMER' in patterns:
                df['pattern_hammer'] = patterns['CDL_HAMMER'] != 0
            if 'CDL_ENGULFING' in patterns:
                df['pattern_engulfing'] = patterns['CDL_ENGULFING'] > 0
            if 'CDL_MORNINGSTAR' in patterns:
                df['pattern_morning_star'] = patterns['CDL_MORNINGSTAR'] != 0
            return df
    except Exception:
        pass

    for idx in range(len(df)):
        row = df.iloc[idx]
        prev = df.iloc[idx - 1] if idx > 0 else None
        prev2 = df.iloc[idx - 2] if idx > 1 else None

        body = abs(float(row['close']) - float(row['open']))
        candle_range = max(float(row['high']) - float(row['low']), 0)
        lower_shadow = min(float(row['open']), float(row['close'])) - float(row['low'])
        upper_shadow = float(row['high']) - max(float(row['open']), float(row['close']))
        if candle_range > 0 and body <= candle_range * 0.1:
            df.at[df.index[idx], 'pattern_doji'] = True
        if body > 0 and lower_shadow >= 2 * body and upper_shadow <= body * 0.5:
            df.at[df.index[idx], 'pattern_hammer'] = True
        if prev is not None:
            prev_red = float(prev['close']) < float(prev['open'])
            curr_green = float(row['close']) > float(row['open'])
            engulfs = float(row['close']) >= float(prev['open']) and float(row['open']) <= float(prev['close'])
            if prev_red and curr_green and engulfs:
                df.at[df.index[idx], 'pattern_engulfing'] = True
        if prev is not None and prev2 is not None:
            prev2_red = float(prev2['close']) < float(prev2['open'])
            prev_small = abs(float(prev['close']) - float(prev['open'])) <= (
                max(float(prev['high']) - float(prev['low']), 0.0) * 0.3
            )
            curr_green = float(row['close']) > float(row['open'])
            gap_down = float(prev['close']) < float(prev2['close'])
            recover = float(row['close']) >= (float(prev2['open']) + float(prev2['close'])) / 2
            if prev2_red and prev_small and curr_green and gap_down and recover:
                df.at[df.index[idx], 'pattern_morning_star'] = True

    return df


def _pattern_success_3d(frame: pd.DataFrame, target_pct: float = 0.05) -> pd.Series:
    if frame is None or frame.empty or 'close' not in frame.columns:
        return pd.Series(dtype=bool)
    closes = frame['close'].values
    success = []
    for idx in range(len(frame)):
        base = float(closes[idx]) if closes[idx] else 0.0
        if base <= 0 or idx + 3 >= len(frame):
            success.append(False)
            continue
        future_max = float(max(closes[idx + 1: idx + 4]))
        success.append(future_max >= base * (1 + target_pct))
    return pd.Series(success, index=frame.index)


def _normalize_score(value: float | None, low: float, high: float) -> float:
    if value is None:
        return 0.0
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (float(value) - low) / (high - low)))


def _bluechip_fundamental_score(symbol: str) -> tuple[float, dict[str, Any]]:
    fundamentals = _yahoo_fundamentals(symbol) or {}
    if not fundamentals:
        return 0.0, {}
    current_ratio = fundamentals.get('current_ratio')
    revenue_growth = fundamentals.get('revenue_growth')
    profit_margins = fundamentals.get('profit_margins')
    debt_to_equity = fundamentals.get('debt_to_equity')
    trailing_pe = fundamentals.get('trailing_pe')

    current_score = _normalize_score(current_ratio, 0.5, 3.0)
    growth_score = _normalize_score(revenue_growth, -0.05, 0.30)
    margin_score = _normalize_score(profit_margins, 0.0, 0.30)
    debt_score = 1.0 - _normalize_score(debt_to_equity, 0.0, 2.5)
    pe_score = 1.0 - _normalize_score(trailing_pe, 8.0, 35.0)
    scores = [current_score, growth_score, margin_score, debt_score, pe_score]
    valid = [s for s in scores if s is not None]
    return (float(sum(valid) / len(valid)) if valid else 0.0), fundamentals


def _index_correlation_score(symbol: str, days: int = 90) -> float:
    try:
        symbols = [symbol, 'SPY', 'QQQ']
        data = yf.download(
            tickers=" ".join(symbols),
            period=f"{days}d",
            interval='1d',
            group_by='ticker',
            threads=True,
            auto_adjust=False,
        )
        if data is None or data.empty:
            return 0.0
        returns: dict[str, pd.Series] = {}
        for sym in symbols:
            frame = data[sym] if isinstance(data.columns, pd.MultiIndex) and sym in data else data.copy()
            close = _extract_close_series(frame)
            if close is None or close.empty:
                continue
            returns[sym] = close.pct_change().dropna()
        base = returns.get(symbol)
        if base is None or base.empty:
            return 0.0
        corr_scores = []
        for idx_sym in ['SPY', 'QQQ']:
            idx_series = returns.get(idx_sym)
            if idx_series is None or idx_series.empty:
                continue
            aligned = pd.concat([base, idx_series], axis=1).dropna()
            if aligned.shape[0] < 10:
                continue
            corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]) or 0.0)
            corr_scores.append(max(0.0, corr))
        return max(corr_scores) if corr_scores else 0.0
    except Exception:
        return 0.0


def _mean_reversion_score(symbol: str) -> tuple[float, float | None, float | None]:
    try:
        hist = yf.Ticker(symbol).history(period='120d', interval='1d', timeout=10)
    except Exception:
        hist = None
    close = _extract_close_series(hist) if hist is not None else None
    if close is None or close.empty:
        return 0.0, None, None
    rsi = _compute_rsi(close, 14)
    if rsi is None:
        return 0.0, None, None
    rsi_score = max(0.0, min(1.0, (30.0 - rsi) / 30.0))
    _, _, lower = _compute_bollinger(close, window=20, width=2.0)
    last_close = float(close.iloc[-1]) if not close.empty else None
    bollinger_bonus = 0.15 if lower is not None and last_close is not None and last_close <= lower else 0.0
    score = max(0.0, min(1.0, rsi_score + bollinger_bonus))
    return score, rsi, lower


def _volume_zscore_20(volume: pd.Series) -> float | None:
    if volume is None or volume.empty:
        return None
    tail = volume.tail(20)
    if len(tail) < 5:
        return None
    mean = float(tail.mean()) if float(tail.mean() or 0) else 0.0
    std = float(tail.std()) if float(tail.std() or 0) else 0.0
    if std <= 0:
        return None
    return float((tail.iloc[-1] - mean) / std)


def _distance_from_ma200(close: pd.Series) -> float | None:
    if close is None or close.empty:
        return None
    ma200 = close.rolling(200, min_periods=50).mean().iloc[-1]
    if ma200 is None or not np.isfinite(ma200) or ma200 == 0:
        return None
    last = float(close.iloc[-1])
    return float((last - ma200) / ma200)


def _rsi_edge(rsi: float | None) -> float | None:
    if rsi is None:
        return None
    return float(max(0.0, (30.0 - rsi) / 30.0))


def _keyword_news_sentiment(symbol: str, keywords: list[str], days: int = 7) -> tuple[float, list[str]]:
    titles = _google_news_titles(symbol, days=days)
    if not titles:
        return 0.0, []
    analyzer = SentimentIntensityAnalyzer()
    scored_titles: list[str] = []
    scores: list[float] = []
    for title in titles:
        text = str(title or '').strip()
        if not text:
            continue
        if keywords and not any(k.lower() in text.lower() for k in keywords):
            continue
        compound = analyzer.polarity_scores(text).get('compound', 0.0)
        scores.append(float(compound))
        scored_titles.append(text)
    if not scores:
        return 0.0, []
    return float(np.mean(scores)), scored_titles[:5]


def _finnhub_price_target(symbol: str) -> float | None:
    key = os.getenv('FINNHUB_API_KEY') or os.getenv('FINNHUB_KEY')
    if not key:
        return None
    try:
        client = finnhub.Client(api_key=key)
        payload = client.price_target(symbol)
        target = payload.get('targetMean') if isinstance(payload, dict) else None
        return float(target) if target is not None else None
    except Exception:
        return None


def _penny_mean_reversion_snapshot(symbol: str) -> dict[str, Any] | None:
    try:
        daily = yfin.Ticker(symbol).history(period='1y', interval='1d', timeout=10)
    except Exception:
        daily = None
    if daily is None or daily.empty:
        return None
    close_col = 'Adj Close' if 'Adj Close' in daily.columns else 'Close'
    if close_col not in daily.columns:
        return None
    close = daily[close_col].dropna()
    if close.empty:
        return None
    price = float(close.iloc[-1])
    high_52w = float(close.max()) if not close.empty else price
    drawdown = ((price - high_52w) / high_52w) if high_52w else 0.0
    volume = daily['Volume'] if 'Volume' in daily.columns else pd.Series(index=daily.index, dtype='float64')
    volume_z = _volume_zscore_20(volume)
    rsi = _compute_rsi(close, 14)
    dist_ma200 = _distance_from_ma200(close)

    try:
        hourly = yfin.Ticker(symbol).history(period='60d', interval='1h', timeout=10)
    except Exception:
        hourly = None
    if hourly is not None and not hourly.empty:
        h_close_col = 'Adj Close' if 'Adj Close' in hourly.columns else 'Close'
        if h_close_col in hourly.columns:
            h_close = hourly[h_close_col].dropna()
            if not h_close.empty:
                rsi = _compute_rsi(h_close, 14) or rsi
                dist_ma200 = _distance_from_ma200(h_close) or dist_ma200

    rsi_edge = _rsi_edge(rsi)
    news_sentiment, news_titles = _keyword_news_sentiment(symbol, ['oil', 'contract', 'fda'])
    target_price = _finnhub_price_target(symbol)

    return {
        'symbol': symbol,
        'price': price,
        'drawdown': drawdown,
        'volume_z': volume_z,
        'rsi': rsi,
        'rsi_edge': rsi_edge,
        'distance_from_ma200': dist_ma200,
        'news_sentiment': news_sentiment,
        'news_titles': news_titles,
        'target_price': target_price,
    }


def _finbert_recent_sentiment(symbol: str, days: int = 2) -> tuple[float, list[str]]:
    titles = _google_news_titles(symbol, days=days)
    score = _finbert_score_from_titles(titles)
    return score, titles


def _penny_breakout_score(bars: pd.DataFrame | None) -> tuple[bool, float]:
    if bars is None or bars.empty:
        return False, 0.0
    frame = bars.copy()
    for col in ['high', 'close', 'volume']:
        if col not in frame.columns:
            return False, 0.0
    frame = frame.dropna(subset=['high', 'close', 'volume'])
    if len(frame) < 25:
        return False, 0.0
    recent = frame.tail(21)
    prev = recent.iloc[:-1]
    last = recent.iloc[-1]
    prev_high = float(prev['high'].max()) if not prev.empty else None
    if prev_high is None or prev_high <= 0:
        return False, 0.0
    last_close = float(last.get('close') or 0)
    last_volume = float(last.get('volume') or 0)
    avg_volume = float(prev['volume'].mean()) if not prev.empty else 0.0
    breakout = last_close > (prev_high * 1.002) and avg_volume > 0 and last_volume >= avg_volume * 1.5
    breakout_strength = max(0.0, (last_close / prev_high) - 1.0)
    volume_strength = (last_volume / avg_volume) if avg_volume else 0.0
    score = min(1.0, breakout_strength * 40) + min(1.0, volume_strength / 3)
    return breakout, max(0.0, min(1.0, score))


def _execute_paper_trades_for_sandbox(sandbox: str, prefix: str) -> dict[str, Any]:
    """Execute live paper trades for a specific sandbox using model signals and risk rules."""
    use_alpaca = sandbox in {'AI_BLUECHIP', 'AI_PENNY'}
    watchlist = _get_watchlist(sandbox, prefix, 'SPY,AAPL,MSFT,NVDA,AMZN')
    universe = 'PENNY' if sandbox == 'AI_PENNY' else 'BLUECHIP'
    model_path = get_model_path(universe)
    buy_threshold = _env_float(prefix, 'BUY_THRESHOLD', '0.82')
    sell_threshold = _env_float(prefix, 'SELL_THRESHOLD', '0.4')
    trail_pct = _env_float(prefix, 'TRAIL_PCT', '0.04')
    atr_mult = _env_float(prefix, 'ATR_MULT', '1.5')
    trail_profit_trigger = _env_float(prefix, 'TRAIL_PROFIT_TRIGGER_PCT', '0.15')
    trail_atr_mult = _env_float(prefix, 'TRAIL_ATR_MULT', '1.5')
    risk_pct = _env_float(prefix, 'RISK_PCT', '0.015')
    position_cap_pct = _env_float(prefix, 'POSITION_CAP_PCT', '0.10')
    initial_capital = _env_float(prefix, 'CAPITAL', '10000')
    min_volume_z = _env_float(prefix, 'VOLUME_ZSCORE_MIN', '0.5')
    break_even_pct = _env_float(prefix, 'BREAK_EVEN_PCT', '0.015')
    break_even_fee_pct = _env_float(prefix, 'BREAK_EVEN_FEE_PCT', '0.0')
    lock_profit_trigger_pct = _env_float(prefix, 'LOCK_PROFIT_TRIGGER_PCT', '0.05')
    lock_profit_stop_pct = _env_float(prefix, 'LOCK_PROFIT_STOP_PCT', '0.03')
    max_spread_pct = _env_float(prefix, 'MAX_SPREAD_PCT', '0.01')
    hard_position_cap_pct = _env_float(prefix, 'HARD_POSITION_CAP_PCT', '0.10')
    reinforce_min_score = _env_float(prefix, 'REINFORCE_MIN_SCORE', '0.85')
    min_altman_z = _env_float(prefix, 'MIN_ALTMAN_Z', '2.0')
    min_win_rate = _env_float(prefix, 'MIN_WIN_RATE', '0.40')
    monthly_contribution = _env_float(prefix, 'MONTHLY_CONTRIBUTION', '2300')
    max_position_cap_pct = 0.02 if sandbox == 'AI_PENNY' else None
    take_profit_volume_days = int(os.getenv(
        f'{prefix}_TAKE_PROFIT_VOLUME_Z_DAYS',
        os.getenv('PAPER_TAKE_PROFIT_VOLUME_Z_DAYS', '3'),
    ))

    closed_trades = PaperTrade.objects.filter(status='CLOSED', sandbox=sandbox)
    closed_pnl = float(sum([float(t.pnl or 0) for t in closed_trades]))
    open_trades_list = list(PaperTrade.objects.filter(status='OPEN', sandbox=sandbox))
    open_trades_by_symbol: dict[str, list[PaperTrade]] = {}
    for trade in open_trades_list:
        open_trades_by_symbol.setdefault(trade.ticker, []).append(trade)
    open_value = sum([float(t.entry_price) * float(t.quantity) for t in open_trades_list])

    if sandbox == 'AI_BLUECHIP':
        first_trade = (
            PaperTrade.objects.filter(sandbox=sandbox)
            .exclude(entry_date__isnull=True)
            .order_by('entry_date')
            .first()
        )
        if first_trade:
            today = timezone.localdate()
            start = first_trade.entry_date.date()
            months = max(0, (today.year - start.year) * 12 + (today.month - start.month))
            initial_capital += months * monthly_contribution

    if max_position_cap_pct is not None:
        position_cap_pct = min(position_cap_pct, max_position_cap_pct)
    position_cap_pct = min(position_cap_pct, hard_position_cap_pct)

    if sandbox == 'AI_BLUECHIP':
        buy_threshold = 0.85
        sell_threshold = 0.65
        trail_pct = 0.05
        min_volume_z = max(min_volume_z, 0.5)
    if sandbox == 'WATCHLIST':
        buy_threshold = 0.60
        sell_threshold = 0.25
    capital = initial_capital + closed_pnl
    available = max(0.0, capital - open_value)
    min_available_capital = _env_float(prefix, 'MIN_AVAILABLE_CAPITAL', '0.0')
    block_new_entries = sandbox == 'AI_PENNY' and min_available_capital > 0 and available < min_available_capital
    regime = get_market_regime_context()
    breaker = _daily_equity_circuit_breaker(sandbox, capital)
    if breaker.get('triggered'):
        if breaker.get('first_trigger'):
            for trade in open_trades_list:
                price = _latest_price(trade.ticker) or float(trade.entry_price)
                trade.status = 'CLOSED'
                trade.exit_price = price
                trade.exit_date = timezone.now()
                trade.pnl = float(price - float(trade.entry_price)) * float(trade.quantity)
                trade.outcome = 'WIN' if float(trade.pnl or 0) > 0 else 'LOSS'
                trade.notes = (trade.notes or '') + ' | Circuit breaker daily equity.'
                trade.save(update_fields=['status', 'exit_price', 'exit_date', 'pnl', 'outcome', 'notes'])
            baseline = float(breaker.get('baseline') or capital)
            loss_amount = max(0.0, baseline - capital)
            threshold_amount = abs(float(breaker.get('threshold') or 0)) * baseline
            drawdown_pct = abs(float(breaker.get('drawdown') or 0)) * 100
            now_str = _ny_time_now().strftime('%H:%M')
            _system_log('SYSTEM', 'ERROR', f'Circuit breaker triggered for {sandbox}', metadata={'capital': capital})
            _send_telegram_alert(
                "\n".join([
                    f"⛔ Circuit breaker {sandbox} (arrêt journée)",
                    f"📉 Perte: ${loss_amount:.2f} / seuil ${threshold_amount:.2f} ({drawdown_pct:.2f}%)",
                    f"🕒 Heure: {now_str}",
                ]),
                allow_during_blackout=True,
                category='risk',
            )
        return {
            'created': 0,
            'closed': len(open_trades_list),
            'available': round(available, 2),
            'entries_blocked_low_capital': True,
            'market_regime': regime,
            'circuit_breaker': True,
        }
    weak_health = _weak_list_health()
    if weak_health.get('defensive'):
        block_new_entries = True
    risk_off = bool(regime.get('risk_off')) if isinstance(regime, dict) else False
    regime_risk_factor = float(os.getenv('MARKET_RISK_OFF_POSITION_FACTOR', '0.5')) if risk_off else 1.0
    pattern_boost = float(os.getenv('PATTERN_SIGNAL_BOOST', '2.0'))

    def _penny_blocked() -> bool:
        if sandbox != 'AI_PENNY':
            return False
        trades = closed_trades.count()
        if trades <= 0:
            return False
        wins = closed_trades.filter(outcome='WIN').count()
        win_rate = (wins / trades) if trades else 0.0
        return win_rate < min_win_rate

    def _latest_price(symbol: str) -> float | None:
        if use_alpaca:
            price = get_latest_trade_price(symbol)
            if price is not None:
                return float(price)
        try:
            hist = yf.Ticker(symbol).history(period='5d', interval='1d', timeout=10)
            if hist is not None and not hist.empty and 'Close' in hist:
                return float(hist['Close'].iloc[-1])
        except Exception:
            return None
        return None

    def _atr(symbol: str) -> float:
        try:
            if use_alpaca:
                hist = get_daily_bars(symbol, days=30)
                if hist is None or hist.empty:
                    return 0.0
                high = hist['high']
                low = hist['low']
                close = hist['close']
            else:
                hist = yf.Ticker(symbol).history(period='20d', interval='1d', timeout=10)
                if hist is None or hist.empty or not {'High', 'Low', 'Close'}.issubset(hist.columns):
                    return 0.0
                high = hist['High']
                low = hist['Low']
                close = hist['Close']
            tr = pd.concat([
                (high - low).abs(),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            return float(atr) if atr is not None else 0.0
        except Exception:
            return 0.0

    def _volume_z_negative_streak(symbol: str, days: int) -> tuple[bool, float | None]:
        if days <= 0:
            return False, None
        try:
            fusion = DataFusionEngine(symbol, use_alpaca=use_alpaca)
            frame = fusion.fuse_all()
            if frame is None or frame.empty or 'VolumeZ' not in frame:
                return False, None
            series = frame['VolumeZ'].tail(days)
            if series.empty or len(series) < days:
                last_value = _safe_float(series.iloc[-1]) if len(series) else None
                return False, last_value
            values = [_safe_float(v) for v in series]
            if any(v is None for v in values):
                return False, _safe_float(values[-1]) if values else None
            return all(v < 0 for v in values), _safe_float(values[-1])
        except Exception:
            return False, None

    def _signal(symbol: str, intraday_ctx: dict[str, Any] | None = None) -> dict[str, Any] | None:
        base_payload = _model_signal(symbol, universe, model_path, use_alpaca=use_alpaca)
        base_signal = float(base_payload.get('signal') or 0) if base_payload else 0.0
        payload = base_payload or {
            'signal': None,
            'features': {},
            'explanations': [],
            'model_version': '',
            'model_name': universe,
        }

        if sandbox == 'WATCHLIST':
            score, rsi, lower_band = _mean_reversion_score(symbol)
            if rsi is None:
                return payload if base_payload else None
            payload['signal'] = score
            payload['model_name'] = 'MEAN_REVERSION'
            payload['features'] = dict(payload.get('features') or {})
            payload['features'].update({
                'rsi14': float(rsi),
                'mean_reversion_score': float(score),
                'bollinger_lower': float(lower_band) if lower_band is not None else 0.0,
            })
            return payload

        if sandbox == 'AI_BLUECHIP':
            fundamentals_score, fundamentals = _bluechip_fundamental_score(symbol)
            corr_score = _index_correlation_score(symbol)
            composite = (0.2 * base_signal) + (0.4 * fundamentals_score) + (0.4 * corr_score)

            growth_min = float(os.getenv('BLUECHIP_GROWTH_PRIORITY_MIN', '0.10'))
            growth_boost = float(os.getenv('BLUECHIP_GROWTH_PRIORITY_BOOST', '0.10'))
            revenue_growth = fundamentals.get('revenue_growth')
            growth_bonus = 0.0
            if revenue_growth is not None:
                try:
                    growth_value = float(revenue_growth)
                    if growth_value >= growth_min:
                        growth_bonus = growth_boost
                except Exception:
                    growth_value = None
            else:
                growth_value = None

            composite = min(1.0, composite + growth_bonus)
            payload['signal'] = float(composite)
            payload['model_name'] = 'BLUECHIP_FUNDAMENTAL'
            payload['features'] = dict(payload.get('features') or {})
            payload['features'].update({
                'fundamental_score': float(fundamentals_score),
                'index_correlation_score': float(corr_score),
                'revenue_growth': growth_value,
                'growth_priority_boost': float(growth_bonus),
            })
            return payload

        if sandbox == 'AI_PENNY':
            ctx = intraday_ctx or {}
            rvol = float(ctx.get('rvol') or 0.0)
            bid_ask_spread_pct = float(ctx.get('bid_ask_spread_pct') or 0.0)
            breakout, breakout_score = _penny_breakout_score(ctx.get('bars'))
            sentiment, _ = _finbert_recent_sentiment(symbol, days=2)
            sentiment_score = max(0.0, min(1.0, (sentiment + 1.0) / 2.0))
            rvol_score = max(0.0, min(1.0, rvol / 3.0))
            composite = (0.1 * base_signal) + (0.9 * (
                (0.45 * rvol_score) + (0.35 * sentiment_score) + (0.20 * breakout_score)
            ))
            payload['signal'] = float(composite)
            payload['model_name'] = 'PENNY_RVOL_SENTIMENT'
            payload['features'] = dict(payload.get('features') or {})
            payload['features'].update({
                'intraday_rvol': rvol,
                'finbert_sentiment': float(sentiment),
                'breakout_1m': 1.0 if breakout else 0.0,
                'breakout_score': float(breakout_score),
                'bid_ask_spread_pct': bid_ask_spread_pct,
            })
            return payload

        return payload

    created = 0
    closed = 0
    decision_stats = {
        'watchlist': len(watchlist),
        'created': 0,
        'blocked_low_capital': 0,
        'blocked_market_sentiment': 0,
        'blocked_threshold': 0,
        'blocked_confidence': 0,
        'blocked_volume_z': 0,
        'blocked_intraday': 0,
    }

    market_sentiment, market_meta = get_market_sentiment()
    market_score = _market_sentiment_score()
    master_entries = {
        row.symbol: row
        for row in MasterWatchlistEntry.objects.all()
    }

    for symbol, trades in open_trades_by_symbol.items():
        price = _latest_price(symbol)
        if price is None:
            continue
        intraday_ctx = None
        if use_alpaca:
            intraday_ctx = get_intraday_context(
                symbol,
                minutes=int(os.getenv('ALPACA_INTRADAY_MINUTES', '390')),
                rvol_window=int(os.getenv('ALPACA_RVOL_WINDOW', '20')),
            )
        signal_payload = _signal(symbol, intraday_ctx=intraday_ctx)
        signal = signal_payload['signal'] if signal_payload else None
        if signal is not None and intraday_ctx:
            now_ny = _ny_time_now()
            if now_ny.time() >= dt_time(14, 30) and float(intraday_ctx.get('rvol') or 0) < 1.0:
                signal = float(signal) * 0.9
        for trade in trades:
            entry_price = float(trade.entry_price or 0)
            profit_pct = (price - entry_price) / entry_price if entry_price else 0.0
            entry_features = dict(trade.entry_features or {})
            partial_taken = bool(entry_features.get('partial_taken'))
            partial_tp_pct = float(os.getenv('PARTIAL_TAKE_PROFIT_PCT', '0.03'))
            partial_qty_pct = float(os.getenv('PARTIAL_TAKE_PROFIT_QTY_PCT', '0.5'))
            if profit_pct >= partial_tp_pct and not partial_taken:
                sell_qty = max(1, int(float(trade.quantity) * partial_qty_pct))
                if sell_qty < trade.quantity:
                    trade.quantity = int(trade.quantity - sell_qty)
                    trade.pnl = float(trade.pnl or 0) + float(price - entry_price) * float(sell_qty)
                    entry_features['partial_taken'] = True
                    trade.entry_features = entry_features
                    trade.notes = (trade.notes or '') + ' | Partial TP exécuté.'
                    trade.save(update_fields=['quantity', 'pnl', 'entry_features', 'notes'])
            time_exit_min = int(os.getenv('TIME_BASED_EXIT_MINUTES', '120'))
            if trade.entry_date and time_exit_min > 0:
                age_min = (timezone.now() - trade.entry_date).total_seconds() / 60
                if age_min >= time_exit_min and profit_pct > 0:
                    trade.status = 'CLOSED'
                    trade.exit_price = price
                    trade.exit_date = timezone.now()
                    trade.pnl = float(price - float(trade.entry_price)) * float(trade.quantity) + float(trade.pnl or 0)
                    trade.outcome = 'WIN' if float(trade.pnl or 0) > 0 else 'LOSS'
                    trade.notes = (trade.notes or '') + ' | Sortie temps (profit positif).'
                    trade.save(update_fields=['status', 'exit_price', 'exit_date', 'pnl', 'outcome', 'notes'])
                    closed += 1
                    continue
            if profit_pct > 0 and take_profit_volume_days > 0:
                negative_streak, latest_volume_z = _volume_z_negative_streak(symbol, take_profit_volume_days)
                if negative_streak:
                    cooldown_hours = getattr(settings, 'ALERT_COOLDOWN_HOURS', 12)
                    cutoff = timezone.now() - timedelta(hours=cooldown_hours)
                    stock = Stock.objects.filter(symbol__iexact=symbol).first()
                    existing = AlertEvent.objects.filter(
                        category='TAKE_PROFIT_SUGGESTION',
                        stock=stock,
                        created_at__gte=cutoff,
                    ).exists()
                    if not existing:
                        latest_z_text = f"{latest_volume_z:.2f}" if latest_volume_z is not None else 'n/a'
                        message = (
                            f"Suggestion take profit partiel: {symbol} profit {profit_pct * 100:.2f}% "
                            f"avec VolumeZ négatif depuis {take_profit_volume_days} jours (dernier {latest_z_text})."
                        )
                        AlertEvent.objects.create(category='TAKE_PROFIT_SUGGESTION', message=message, stock=stock)
            break_even_trigger_pct = max(0.0, break_even_pct + break_even_fee_pct)
            break_even_stop = entry_price if entry_price and price >= (entry_price * (1 + break_even_trigger_pct)) else 0.0
            atr = _atr(symbol)
            dynamic_atr_stop = 0.0
            if profit_pct >= trail_profit_trigger and atr > 0:
                dynamic_atr_stop = price - (trail_atr_mult * atr)
            profit_trail_pct = None
            if profit_pct >= 0.06:
                profit_trail_pct = 0.005
            elif profit_pct >= 0.03:
                profit_trail_pct = 0.01
            stop_mult = _stop_loss_multiplier(trade.entry_explanations)
            adjusted_trail = max(0.005, min(0.25, trail_pct * stop_mult))
            new_stop = max(
                float(trade.stop_loss),
                price * (1 - adjusted_trail),
                break_even_stop,
                dynamic_atr_stop,
            )
            if profit_trail_pct is not None:
                new_stop = max(new_stop, price * (1 - profit_trail_pct))
            if entry_price and price >= entry_price * (1 + lock_profit_trigger_pct):
                new_stop = max(new_stop, entry_price * (1 + lock_profit_stop_pct))
            if intraday_ctx:
                pattern_signal = float(intraday_ctx.get('pattern_signal') or 0)
                rvol = float(intraday_ctx.get('rvol') or 0)
                if pattern_signal < 0 and rvol >= 2:
                    new_stop = max(new_stop, price * (1 - (trail_pct * 0.5)))
                    trade.notes = (trade.notes or '') + ' | Pattern baissier + RVOL élevé: stop resserré.'
            trade.stop_loss = new_stop
            stop_hit = price <= new_stop
            signal_exit = signal is not None and signal < sell_threshold
            velocity_exit = _price_velocity_drop(intraday_ctx)
            should_close = stop_hit or signal_exit or velocity_exit
            if should_close:
                trade.status = 'CLOSED'
                trade.exit_price = price
                trade.exit_date = timezone.now()
                trade.pnl = float(price - float(trade.entry_price)) * float(trade.quantity)
                trade.outcome = 'WIN' if float(trade.pnl or 0) > 0 else 'LOSS'
                volume_z = _safe_float((signal_payload or {}).get('features', {}).get('VolumeZ'))
                if stop_hit:
                    _mark_reentry(symbol, sandbox)
                if velocity_exit:
                    exit_reason = 'Vitesse baisse'
                else:
                    exit_reason = 'Stop Loss' if stop_hit else 'Signal IA'
                volume_note = f" VolumeZ {volume_z:.2f}." if volume_z is not None else ''
                trade.notes = (
                    f"Trade fermé à cause de {exit_reason}. Cause probable : Volume trop faible lors de l'entrée." + volume_note
                )
                closed += 1
            trade.save(update_fields=['stop_loss', 'status', 'exit_price', 'exit_date', 'pnl', 'outcome', 'notes'])

    for symbol in watchlist:
        existing_trades = open_trades_by_symbol.get(symbol, [])
        if _midday_blackout():
            _decision_log(symbol, sandbox, 'SKIP', 'midday_blackout')
            decision_stats['blocked_intraday'] += 1
            continue
        reentry_candidate = _is_reentry_candidate(symbol, sandbox)
        master_entry = master_entries.get(symbol)
        if master_entry and master_entry.category == 'WEAK_SHORT':
            threshold = master_entry.block_if_market_sentiment_lt
            if threshold is None:
                threshold = 0.0
            if market_score is not None and market_score < float(threshold):
                continue
        intraday_ctx = None
        pattern_signal = None
        rvol = None
        if use_alpaca:
            intraday_ctx = get_intraday_context(
                symbol,
                minutes=int(os.getenv('ALPACA_INTRADAY_MINUTES', '390')),
                rvol_window=int(os.getenv('ALPACA_RVOL_WINDOW', '20')),
            )
            if intraday_ctx:
                pattern_signal = float(intraday_ctx.get('pattern_signal') or 0)
                rvol = float(intraday_ctx.get('rvol') or 0)
                if _is_halted(intraday_ctx) or _flash_crash(intraday_ctx):
                    _decision_log(symbol, sandbox, 'SKIP', 'halt_or_flash')
                    continue
                min_intraday_rvol = float(os.getenv('MIN_INTRADAY_RVOL', '1.0'))
                if rvol < min_intraday_rvol:
                    _decision_log(symbol, sandbox, 'SKIP', f'rvol<{min_intraday_rvol}')
                    decision_stats['blocked_intraday'] += 1
                    continue
                if _vwap_filter_block(intraday_ctx):
                    _decision_log(symbol, sandbox, 'SKIP', 'below_vwap')
                    decision_stats['blocked_intraday'] += 1
                    continue
                spread_pct = float(intraday_ctx.get('bid_ask_spread_pct') or 0.0)
                penny_max_spread = float(os.getenv('PENNY_MAX_SPREAD_PCT', '0.02'))
                spread_limit = penny_max_spread if sandbox == 'AI_PENNY' else max_spread_pct
                if spread_pct > 0 and spread_pct > spread_limit:
                    _decision_log(symbol, sandbox, 'SKIP', f'spread>{spread_limit}')
                    decision_stats['blocked_intraday'] += 1
                    continue
        signal_payload = _signal(symbol, intraday_ctx=intraday_ctx)
        signal = signal_payload['signal'] if signal_payload else None
        if signal is not None:
            signal = float(signal) * _time_of_day_penalty()
        if pattern_signal is not None and rvol is not None:
            if pattern_signal < 0 and rvol >= 2:
                continue
            if signal is not None and pattern_signal > 0 and rvol >= 2:
                signal = min(1.0, float(signal) * 1.02)
        if signal is not None and intraday_ctx:
            now_ny = _ny_time_now()
            if now_ny.time() >= dt_time(14, 30) and float(intraday_ctx.get('rvol') or 0) < 1.0:
                signal = float(signal) * 0.9
        if signal is not None and sandbox == 'AI_BLUECHIP':
            multiplier = _bluechip_aggressive_multiplier()
            if multiplier > 1.0:
                signal = min(1.0, float(signal) * multiplier)
        if signal is not None:
            pattern_payload = cache.get(f"pattern_scan:{symbol}") or {}
            if pattern_payload.get('hammer_support'):
                signal = min(1.0, float(signal) * pattern_boost)
        if market_sentiment in {'BEARISH', 'CAUTION'}:
            notify_key = f"market_sentiment_block:{sandbox}:{_ny_time_now().strftime('%Y%m%d%H')}"
            if not cache.get(notify_key):
                cache.set(notify_key, True, timeout=60 * 60)
                _send_telegram_alert(
                    "🧭 Marché défavorable : trade bloqué ("
                    f"{market_sentiment}). SPY {market_meta.get('spy_change')}%, "
                    f"TSX {market_meta.get('tsx_change')}%, VIX {market_meta.get('vix')}.",
                    allow_during_blackout=True,
                    category='risk',
                )
            decision_stats['blocked_market_sentiment'] += 1
            continue
        if signal is None or signal < buy_threshold:
            decision_stats['blocked_threshold'] += 1
            continue
        if block_new_entries:
            decision_stats['blocked_low_capital'] += 1
            continue
        if _atr_spike(symbol, use_alpaca=use_alpaca):
            _decision_log(symbol, sandbox, 'SKIP', 'atr_spike')
            continue
        if not _btc_trend_ok(symbol):
            _decision_log(symbol, sandbox, 'SKIP', 'btc_trend_down')
            continue
        earnings_blocked, _ = _earnings_blackout(symbol, days=2)
        if earnings_blocked:
            continue
        if os.getenv('MULTI_TIMEFRAME_DAILY_ENABLED', 'true').lower() in {'1', 'true', 'yes', 'y'}:
            if not _daily_trend_ok(symbol, use_alpaca=use_alpaca):
                continue
        if _correlation_blocked(symbol, list(open_trades_by_symbol.keys())):
            continue
        if _spread_too_wide(symbol, max_spread_pct):
            continue
        if reentry_candidate:
            if not _reentry_confirmed(intraday_ctx):
                _decision_log(symbol, sandbox, 'SKIP', 'reentry_no_vwap')
                continue
        reentry_window = int(os.getenv('REENTRY_WINDOW_MINUTES', '30'))
        reentry_bonus = float(os.getenv('REENTRY_MIN_SIGNAL_BONUS', '0.05'))
        reentry_size_factor = 1.0
        if reentry_window > 0:
            cutoff = timezone.now() - timedelta(minutes=reentry_window)
            recent_loss = PaperTrade.objects.filter(
                ticker__iexact=symbol,
                sandbox=sandbox,
                status='CLOSED',
                outcome='LOSS',
                exit_date__gte=cutoff,
            ).exists()
            if recent_loss and (signal is None or signal < (buy_threshold + reentry_bonus)):
                continue
        reentry_stop_minutes = int(os.getenv('REENTRY_STOPLOSS_WINDOW_MINUTES', '120'))
        reentry_stop_min_signal = float(os.getenv('REENTRY_STOPLOSS_MIN_SIGNAL', '0.8'))
        if reentry_stop_minutes > 0:
            cutoff = timezone.now() - timedelta(minutes=reentry_stop_minutes)
            recent_stop = PaperTrade.objects.filter(
                ticker__iexact=symbol,
                sandbox=sandbox,
                status='CLOSED',
                outcome='LOSS',
                exit_date__gte=cutoff,
                notes__icontains='Stop Loss',
            ).exists()
            if recent_stop and signal is not None and signal >= reentry_stop_min_signal:
                reentry_size_factor = float(os.getenv('REENTRY_STOPLOSS_SIZE_FACTOR', '0.75'))
        if reentry_candidate:
            reentry_size_factor = float(os.getenv('REENTRY_STOPLOSS_SIZE_FACTOR', '0.75'))
        if _loss_blacklist(symbol, sandbox):
            _decision_log(symbol, sandbox, 'SKIP', 'loss_blacklist')
            continue
        if not _sector_trend_ok(symbol):
            _decision_log(symbol, sandbox, 'SKIP', 'sector_trend')
            continue
        if existing_trades and signal < reinforce_min_score:
            continue
        if sandbox == 'AI_PENNY' and _penny_blocked():
            continue
        if sandbox != 'AI_PENNY':
            volume_z = _safe_float((signal_payload or {}).get('features', {}).get('VolumeZ'))
            if volume_z is not None and volume_z < min_volume_z:
                ai_score = float(signal or 0) * 100
                message = (
                    f"Non-Trade [{sandbox}]: {symbol} signal {ai_score:.2f}% "
                    f"volume_z {float(volume_z):.2f} < {min_volume_z:.2f}"
                )
                stock = Stock.objects.filter(symbol__iexact=symbol).first()
                AlertEvent.objects.create(category='PAPER_NON_TRADE', message=message, stock=stock)
                decision_stats['blocked_volume_z'] += 1
                continue
        if sandbox == 'AI_PENNY':
            min_rvol = float(os.getenv('AI_PENNY_MIN_RVOL', '1.8'))
            intraday_rvol = float((intraday_ctx or {}).get('rvol') or 0.0)
            breakout_flag = float((signal_payload or {}).get('features', {}).get('breakout_1m') or 0.0)
            if intraday_rvol < min_rvol or breakout_flag <= 0:
                continue
            if _reddit_hype_risk(symbol):
                continue
            sentiment_raw = float((signal_payload or {}).get('features', {}).get('finbert_sentiment') or 0.0)
            if _pump_dump_risk(intraday_ctx, sentiment_raw):
                continue
            if os.getenv('ORDER_BOOK_IMBALANCE_ENABLED', 'false').lower() in {'1', 'true', 'yes', 'y'}:
                imbalance = get_order_book_imbalance(symbol)
                min_imbalance = float(os.getenv('ORDER_BOOK_IMBALANCE_MIN', '1.0'))
                if imbalance is not None and imbalance < min_imbalance:
                    continue
        price = _latest_price(symbol)
        if price is None:
            continue
        atr = _atr(symbol)
        stop_mult = _stop_loss_multiplier((signal_payload or {}).get('explanations'))
        adjusted_trail = max(0.005, min(0.25, trail_pct * stop_mult))
        reentry_atr_mult = float(os.getenv('REENTRY_STOPLOSS_ATR_MULTI', str(atr_mult)))
        if reentry_candidate:
            stop_distance = max(price * adjusted_trail, reentry_atr_mult * atr, price * 0.01)
        else:
            stop_distance = max(price * adjusted_trail, atr_mult * atr, price * 0.01)
        if intraday_ctx:
            vol = float(intraday_ctx.get('volatility') or 0)
            if vol > 0:
                vol_stop = price * min(max(vol * 3, 0.01), 0.06)
                stop_distance = max(stop_distance, vol_stop)
        risk_budget = capital * risk_pct
        position_cap = capital * position_cap_pct
        exposure_pct = (open_value / capital) if capital else 1.0
        if exposure_pct >= 0.8:
            continue
        confidence_factor = max(0.0, min(1.0, (float(signal) - 0.65) / 0.35)) if signal is not None else 0.0
        if confidence_factor <= 0:
            decision_stats['blocked_confidence'] += 1
            continue
        position_value = _dynamic_position_target(capital)
        position_value = min(position_value, available, position_cap, risk_budget)
        position_value *= _multi_model_boost(symbol, float(signal or 0), universe, use_alpaca)
        position_value *= regime_risk_factor
        position_value *= confidence_factor
        position_value *= reentry_size_factor
        quantity = int(position_value / price) if price else 0
        if quantity <= 0:
            continue
        _system_log(
            'SYSTEM',
            'INFO',
            f"💰 Dynamic Sizing: Equity {capital:.2f} | Risk 1.5% | Target {position_value:.2f}$ | Qty {quantity}",
            symbol=symbol,
            metadata={
                'equity': round(capital, 2),
                'target_value': round(position_value, 2),
                'qty': quantity,
            },
        )
        stop_loss = price - stop_distance
        entry_features = dict((signal_payload or {}).get('features') or {})
        if intraday_ctx:
            entry_features.update({
                'intraday_pattern_signal': float(intraday_ctx.get('pattern_signal') or 0),
                'intraday_rvol': float(intraday_ctx.get('rvol') or 0),
                'intraday_volatility': float(intraday_ctx.get('volatility') or 0),
                'intraday_rsi14': float(intraday_ctx.get('rsi14') or 0),
                'intraday_ema20': float(intraday_ctx.get('ema20') or 0),
                'intraday_ema50': float(intraday_ctx.get('ema50') or 0),
                'intraday_vwap': float(intraday_ctx.get('vwap') or 0),
                'intraday_price_to_vwap': float(intraday_ctx.get('price_to_vwap') or 0),
            })
        entry_features.update(_entry_time_features(symbol))
        PaperTrade.objects.create(
            ticker=symbol,
            sandbox=sandbox,
            entry_price=round(price, 2),
            quantity=quantity,
            entry_signal=signal,
            entry_features=entry_features,
            entry_explanations=(signal_payload or {}).get('explanations'),
            model_name=(signal_payload or {}).get('model_name', universe),
            model_version=(signal_payload or {}).get('model_version', ''),
            broker='SIM',
            stop_loss=round(stop_loss, 2),
            status='OPEN',
            pnl=0,
            notes=(
                f"Signal {signal:.2f} / ATR {atr:.2f} | Mise suggérée {allocation:.0f}$ "
                f"(Confiance {float(signal or 0) * 100:.1f}%)"
            ),
        )
        if os.getenv('TRADE_REASON_ALERTS', 'false').lower() in {'1', 'true', 'yes', 'y'}:
            reason = (signal_payload or {}).get('explanations') or []
            AlertEvent.objects.create(
                category='TRADE_REASON',
                message=f"{symbol} entry: {reason}",
                stock=Stock.objects.filter(symbol__iexact=symbol).first(),
            )
        _decision_log(symbol, sandbox, 'BUY', 'paper_trade_created', signal)
        created += 1
        decision_stats['created'] += 1
        available = max(0.0, available - (quantity * price))

    if created == 0 and decision_stats['watchlist']:
        _system_log(
            'SYSTEM',
            'WARNING',
            f"Paper trade: aucun trade pour {sandbox}",
            metadata={
                **decision_stats,
                'buy_threshold': buy_threshold,
                'min_volume_z': min_volume_z,
                'available_capital': round(available, 2),
                'block_new_entries': block_new_entries,
                'market_sentiment': market_sentiment,
            },
        )

    return {
        'created': created,
        'closed': closed,
        'available': round(available, 2),
        'entries_blocked_low_capital': block_new_entries,
        'market_regime': regime,
    }


def _alpaca_position_qty(position: Any | None) -> float:
    if position is None:
        return 0.0
    qty = getattr(position, 'qty', None) or getattr(position, 'quantity', None)
    try:
        return float(qty)
    except Exception:
        return 0.0


def _alpaca_position_avg_price(position: Any | None) -> float:
    if position is None:
        return 0.0
    price = getattr(position, 'avg_entry_price', None) or getattr(position, 'average_entry_price', None)
    try:
        return float(price)
    except Exception:
        return 0.0


def _alpaca_buying_power() -> float:
    account = get_account()
    if not account:
        return 0.0
    for attr in ('buying_power', 'cash', 'equity'):
        val = getattr(account, attr, None)
        try:
            if val is not None:
                return float(val)
        except Exception:
            continue
    return 0.0


def _execute_alpaca_paper_trades_for_sandbox(sandbox: str, prefix: str) -> dict[str, Any]:
    """Execute paper trades on Alpaca using model signals."""
    watchlist = _get_watchlist(sandbox, prefix, 'SPY,AAPL,MSFT,NVDA,AMZN')
    universe = 'PENNY' if sandbox == 'AI_PENNY' else 'BLUECHIP'
    model_path = get_model_path(universe)
    buy_threshold = _env_float(prefix, 'BUY_THRESHOLD', '0.82')
    sell_threshold = _env_float(prefix, 'SELL_THRESHOLD', '0.4')
    trail_pct = _env_float(prefix, 'TRAIL_PCT', '0.04')
    atr_mult = _env_float(prefix, 'ATR_MULT', '1.5')
    risk_pct = _env_float(prefix, 'RISK_PCT', '0.015')
    position_cap_pct = _env_float(prefix, 'POSITION_CAP_PCT', '0.10')
    min_volume_z = _env_float(prefix, 'VOLUME_ZSCORE_MIN', '0.5')
    max_cost_pct = _env_float(prefix, 'ALPACA_MAX_COST_PCT', '0.01')
    commission_pct = _env_float(prefix, 'ALPACA_COMMISSION_PCT', '0.0')
    max_spread_pct = _env_float(prefix, 'MAX_SPREAD_PCT', '0.01')
    hard_position_cap_pct = _env_float(prefix, 'HARD_POSITION_CAP_PCT', '0.10')
    lock_profit_trigger_pct = _env_float(prefix, 'LOCK_PROFIT_TRIGGER_PCT', '0.05')
    lock_profit_stop_pct = _env_float(prefix, 'LOCK_PROFIT_STOP_PCT', '0.03')
    allow_gemini = os.getenv('ALPACA_GEMINI_REASON', 'true').lower() in {'1', 'true', 'yes', 'y'}
    min_confidence = _env_float(prefix, 'ALPACA_MIN_CONFIDENCE', '0.7')
    min_sentiment = _env_float(prefix, 'ALPACA_MIN_SENTIMENT', '0.0')
    min_imbalance = _env_float(prefix, 'ALPACA_MIN_IMBALANCE', '1.0')
    manual_approval = os.getenv('ALPACA_MANUAL_APPROVAL', 'false').lower() in {'1', 'true', 'yes', 'y'}

    shadow_mode = os.getenv('ALPACA_SHADOW_TRADING', 'false').lower() in {'1', 'true', 'yes', 'y'}
    positions = {getattr(p, 'symbol', '').strip().upper(): p for p in get_open_positions() if getattr(p, 'symbol', None)}
    market_sentiment, market_meta = get_market_sentiment()
    market_score = _market_sentiment_score()
    master_entries = {
        row.symbol: row
        for row in MasterWatchlistEntry.objects.all()
    }
    regime = get_market_regime_context()
    risk_off = bool(regime.get('risk_off')) if isinstance(regime, dict) else False
    regime_risk_factor = float(os.getenv('MARKET_RISK_OFF_POSITION_FACTOR', '0.5')) if risk_off else 1.0
    tier1_size_mult = float(os.getenv('TIER1_SIZE_MULT', '1.5'))
    tier3_size_mult = float(os.getenv('TIER3_SIZE_MULT', '0.5'))
    tier1_atr_mult = float(os.getenv('TIER1_TRAIL_ATR_MULT', '2.0'))
    use_limit_orders = os.getenv('ALPACA_USE_LIMIT_ORDERS', 'true').lower() in {'1', 'true', 'yes', 'y'}
    limit_offset = float(os.getenv('ALPACA_LIMIT_OFFSET', '0.01'))

    created = 0
    closed = 0
    buying_power = _alpaca_buying_power()
    account = get_account()
    equity_now = None
    try:
        equity_now = float(getattr(account, 'equity', None)) if account is not None else None
    except Exception:
        equity_now = None
    equity_now = equity_now if equity_now is not None else buying_power
    breaker = _daily_equity_circuit_breaker(sandbox, float(equity_now or 0))
    if breaker.get('triggered'):
        if breaker.get('first_trigger'):
            for symbol, pos in positions.items():
                try:
                    close_position(symbol)
                except Exception:
                    continue
            baseline = float(breaker.get('baseline') or equity_now or 0)
            loss_amount = max(0.0, baseline - float(equity_now or 0))
            threshold_amount = abs(float(breaker.get('threshold') or 0)) * baseline
            drawdown_pct = abs(float(breaker.get('drawdown') or 0)) * 100
            now_str = _ny_time_now().strftime('%H:%M')
            _system_log('SYSTEM', 'ERROR', f'Circuit breaker triggered for {sandbox}', metadata={'equity': equity_now})
            _send_telegram_alert(
                "\n".join([
                    f"⛔ Circuit breaker Alpaca {sandbox} (arrêt journée)",
                    f"📉 Perte: ${loss_amount:.2f} / seuil ${threshold_amount:.2f} ({drawdown_pct:.2f}%)",
                    f"🕒 Heure: {now_str}",
                ]),
                allow_during_blackout=True,
                category='risk',
            )
        return {
            'created': 0,
            'closed': len(positions),
            'buying_power': round(buying_power, 2),
            'market_regime': regime,
            'market_meta': market_meta,
            'circuit_breaker': True,
        }
    if shadow_mode:
        positions = {}
    shadow_trades = []
    shadow_by_symbol: dict[str, PaperTrade] = {}
    if shadow_mode:
        shadow_trades = list(PaperTrade.objects.filter(broker='SHADOW', status='OPEN', sandbox=sandbox))
        shadow_by_symbol = {str(t.ticker or '').strip().upper(): t for t in shadow_trades}
    position_cap_pct = min(position_cap_pct, hard_position_cap_pct)
    weak_health = _weak_list_health()

    def _alpaca_latest_price(symbol: str) -> float | None:
        price = get_latest_trade_price(symbol)
        if price is not None:
            return float(price)
        try:
            hist = yf.Ticker(symbol).history(period='5d', interval='1d', timeout=10)
            if hist is not None and not hist.empty and 'Close' in hist:
                return float(hist['Close'].iloc[-1])
        except Exception:
            return None
        return None

    def _alpaca_fusion(symbol: str) -> pd.DataFrame:
        fusion = DataFusionEngine(symbol, use_alpaca=True)
        return fusion.fuse_all() or pd.DataFrame()

    def _update_open_trade(trade: PaperTrade, price: float, signal: float | None) -> bool:
        frame = _alpaca_fusion(trade.ticker)
        tier, tier_reasons = _classify_tier(frame, {'signal': signal})
        entry_features = dict(trade.entry_features or {})
        entry_features['tier'] = tier
        entry_features['tier_reasons'] = tier_reasons

        adx_val, adx_slope = _compute_adx(frame)
        atr_val = _atr_from_frame(frame) or 0.0
        high_water = float(entry_features.get('alpaca_high_water') or trade.entry_price or 0.0)
        if price > high_water:
            high_water = price
        entry_features['alpaca_high_water'] = high_water

        should_sell = False
        entry_price = float(trade.entry_price or 0.0)
        profit_pct = (price - entry_price) / entry_price if entry_price else 0.0
        profit_trail_pct = None
        if profit_pct >= 0.06:
            profit_trail_pct = 0.005
        elif profit_pct >= 0.03:
            profit_trail_pct = 0.01
        if entry_price and price >= entry_price * (1 + lock_profit_trigger_pct):
            trade.stop_loss = max(float(trade.stop_loss or 0.0), entry_price * (1 + lock_profit_stop_pct))
        if profit_trail_pct is not None:
            trade.stop_loss = max(float(trade.stop_loss or 0.0), price * (1 - profit_trail_pct))
        intraday_ctx = get_intraday_context(
            trade.ticker,
            minutes=int(os.getenv('ALPACA_INTRADAY_MINUTES', '390')),
            rvol_window=int(os.getenv('ALPACA_RVOL_WINDOW', '20')),
        )
        if _price_velocity_drop(intraday_ctx):
            should_sell = True
        if trade.stop_loss and price <= float(trade.stop_loss):
            _mark_reentry(trade.ticker, sandbox)
            should_sell = True
        if tier == 'T1':
            trail_stop = high_water - (tier1_atr_mult * atr_val) if atr_val else None
            if trail_stop is not None and price <= trail_stop:
                should_sell = True
            if adx_val is not None and adx_val < 20:
                should_sell = True
        elif tier == 'T2':
            age_days = (timezone.now() - trade.entry_date).days if trade.entry_date else 0
            if age_days >= int(os.getenv('TIER2_MAX_DAYS', '3')):
                if adx_val is None or adx_val < 25 or (adx_slope is not None and adx_slope <= 0):
                    should_sell = True
            if signal is not None and signal < sell_threshold:
                should_sell = True
        else:
            now_ny = _ny_time_now()
            if now_ny.time() >= dt_time(15, 45) or (trade.entry_date and trade.entry_date.date() < timezone.localdate()):
                should_sell = True
            if signal is not None and signal < sell_threshold:
                should_sell = True

        trade.entry_features = entry_features
        trade.save(update_fields=['entry_features'])
        return should_sell

    for symbol in watchlist:
        symbol = (symbol or '').strip().upper()
        if not symbol:
            continue
        if _midday_blackout():
            _decision_log(symbol, sandbox, 'SKIP', 'midday_blackout')
            continue
        reentry_candidate = _is_reentry_candidate(symbol, sandbox)
        if weak_health.get('defensive'):
            continue
        master_entry = master_entries.get(symbol)
        if master_entry and master_entry.category == 'WEAK_SHORT':
            threshold = master_entry.block_if_market_sentiment_lt
            if threshold is None:
                threshold = 0.0
            if market_score is not None and market_score < float(threshold):
                continue
        earnings_blocked, _ = _earnings_blackout(symbol, days=2)
        if earnings_blocked:
            continue
        if os.getenv('MULTI_TIMEFRAME_DAILY_ENABLED', 'true').lower() in {'1', 'true', 'yes', 'y'}:
            if not _daily_trend_ok(symbol, use_alpaca=True):
                continue
        if _correlation_blocked(symbol, list(positions.keys())):
            continue
        if _symbol_currency(symbol) == 'CAD':
            notify_key = f"alpaca_cad_alert:{symbol}:{_ny_time_now().strftime('%Y%m%d')}"
            if not cache.get(notify_key):
                cache.set(notify_key, True, timeout=60 * 60 * 6)
                signal_payload = _model_signal(symbol, universe, model_path, use_alpaca=False)
                confidence = float(signal_payload.get('signal') or 0.0) if signal_payload else 0.0
                feature_snapshot = (signal_payload or {}).get('features') or {}
                sentiment_score = _safe_float(feature_snapshot.get('sentiment_score'))
                if sentiment_score is None:
                    sentiment_score, _ = _news_sentiment_score(symbol, days=1)
                corr, driver = _tsx_driver_correlation(symbol)
                price = _latest_price_snapshot(symbol)
                dip_low_pct = float(os.getenv('TSX_ENTRY_DIP_LOW', '0.006'))
                dip_high_pct = float(os.getenv('TSX_ENTRY_DIP_HIGH', '0.012'))
                target_pct = float(os.getenv('TSX_TARGET_PCT', '0.04'))
                stop_pct = float(os.getenv('TSX_STOP_PCT', '0.03'))
                entry_low = price * (1 - dip_high_pct) if price else None
                entry_high = price * (1 - dip_low_pct) if price else None
                target_price = price * (1 + target_pct) if price else None
                stop_price = price * (1 - stop_pct) if price else None
                move_type = _tsx_move_type(
                    rvol=float(feature_snapshot.get('RVOL10') or 0.0),
                    volatility=float(feature_snapshot.get('Volatility') or 0.0),
                    sentiment=sentiment_score,
                )
                action_label = f"ACHAT ({move_type.title()})" if confidence >= min_confidence else "SURVEILLER"
                message = _build_tsx_manual_trade_alert(
                    ticker=symbol,
                    action_label=action_label,
                    confidence=confidence,
                    confidence_threshold=min_confidence,
                    sentiment=sentiment_score,
                    sentiment_threshold=min_sentiment,
                    corr=corr,
                    driver=driver,
                    price=price,
                    entry_low=entry_low,
                    entry_high=entry_high,
                    target_price=target_price,
                    stop_price=stop_price,
                )
                _send_telegram_alert(
                    message,
                    allow_during_blackout=True,
                    category='signal',
                )
            continue
        if market_sentiment in {'BEARISH', 'CAUTION'}:
            continue

        signal_payload = _model_signal(symbol, universe, model_path, use_alpaca=True)
        signal = signal_payload.get('signal') if signal_payload else None
        if signal is None:
            continue
        intraday_ctx = get_intraday_context(
            symbol,
            minutes=int(os.getenv('ALPACA_INTRADAY_MINUTES', '390')),
            rvol_window=int(os.getenv('ALPACA_RVOL_WINDOW', '20')),
        )
        if intraday_ctx:
            if _is_halted(intraday_ctx) or _flash_crash(intraday_ctx):
                _decision_log(symbol, sandbox, 'SKIP', 'halt_or_flash')
                continue
            min_intraday_rvol = float(os.getenv('MIN_INTRADAY_RVOL', '1.0'))
            if float(intraday_ctx.get('rvol') or 0.0) < min_intraday_rvol:
                _decision_log(symbol, sandbox, 'SKIP', f'rvol<{min_intraday_rvol}')
                continue
            if _vwap_filter_block(intraday_ctx):
                _decision_log(symbol, sandbox, 'SKIP', 'below_vwap')
                continue
            spread_pct = float(intraday_ctx.get('bid_ask_spread_pct') or 0.0)
            penny_max_spread = float(os.getenv('PENNY_MAX_SPREAD_PCT', '0.02'))
            spread_limit = penny_max_spread if sandbox == 'AI_PENNY' else max_spread_pct
            if spread_pct > 0 and spread_pct > spread_limit:
                _decision_log(symbol, sandbox, 'SKIP', f'spread>{spread_limit}')
                continue
        if reentry_candidate and not _reentry_confirmed(intraday_ctx):
            _decision_log(symbol, sandbox, 'SKIP', 'reentry_no_vwap')
            continue
        signal = float(signal) * _time_of_day_penalty()
        if signal < min_confidence:
            continue

        feature_snapshot = (signal_payload or {}).get('features') or {}
        sentiment_score = float(feature_snapshot.get('sentiment_score') or 0.0)
        if sentiment_score < min_sentiment:
            continue

        if _loss_blacklist(symbol, sandbox):
            _decision_log(symbol, sandbox, 'SKIP', 'loss_blacklist')
            continue
        if not _sector_trend_ok(symbol):
            _decision_log(symbol, sandbox, 'SKIP', 'sector_trend')
            continue

        if _atr_spike(symbol, use_alpaca=True):
            _decision_log(symbol, sandbox, 'SKIP', 'atr_spike')
            continue
        if not _btc_trend_ok(symbol):
            _decision_log(symbol, sandbox, 'SKIP', 'btc_trend_down')
            continue

        if sandbox != 'AI_PENNY':
            volume_z = _safe_float((signal_payload or {}).get('features', {}).get('VolumeZ'))
            if volume_z is not None and volume_z < min_volume_z:
                continue

        spread_pct = get_latest_bid_ask_spread_pct(symbol)
        if spread_pct is None:
            bid, ask = _latest_bid_ask(symbol)
            if bid and ask and ask > 0:
                spread_pct = float((ask - bid) / ((ask + bid) / 2))
        if spread_pct is None:
            spread_pct = 0.0
        if spread_pct > max_spread_pct:
            continue
        if (spread_pct + commission_pct) > max_cost_pct:
            continue

        imbalance = get_order_book_imbalance(symbol)
        if imbalance is not None and imbalance < min_imbalance:
            continue

        price = _alpaca_latest_price(symbol)
        if price is None:
            continue
        if sandbox == 'AI_PENNY':
            intraday_ctx = get_intraday_context(
                symbol,
                minutes=int(os.getenv('ALPACA_INTRADAY_MINUTES', '390')),
                rvol_window=int(os.getenv('ALPACA_RVOL_WINDOW', '20')),
            )
            finbert_sentiment = float((signal_payload or {}).get('features', {}).get('finbert_sentiment') or 0.0)
            if _pump_dump_risk(intraday_ctx, finbert_sentiment):
                continue
            if _reddit_hype_risk(symbol):
                continue

        existing_pos = positions.get(symbol)
        open_trade = PaperTrade.objects.filter(broker='ALPACA', status='OPEN', ticker__iexact=symbol).first()
        shadow_trade = shadow_by_symbol.get(symbol) if shadow_mode else None

        if shadow_mode and shadow_trade:
            should_sell = _update_open_trade(shadow_trade, price, signal)
            if should_sell:
                shadow_trade.status = 'CLOSED'
                shadow_trade.exit_price = price
                shadow_trade.exit_date = timezone.now()
                shadow_trade.pnl = float(price - float(shadow_trade.entry_price)) * float(shadow_trade.quantity)
                shadow_trade.outcome = 'WIN' if float(shadow_trade.pnl or 0) > 0 else 'LOSS'
                shadow_trade.save(update_fields=['status', 'exit_price', 'exit_date', 'pnl', 'outcome'])
                closed += 1
            continue

        if existing_pos and open_trade:
            should_sell = _update_open_trade(open_trade, price, signal)
            if should_sell:
                qty = int(abs(_alpaca_position_qty(existing_pos)))
                if qty <= 0:
                    continue
                if use_limit_orders and price:
                    order = submit_limit_order(symbol, qty, 'sell', max(0.01, price - limit_offset))
                else:
                    order = submit_market_order(symbol, qty, 'sell')
                if order is None:
                    _send_telegram_alert(
                        f"⚠️ Order sell failed {symbol} qty {qty} (Alpaca)",
                        allow_during_blackout=True,
                        category='order_error',
                    )
                    continue
                open_trade.broker_order_id = str(getattr(order, 'id', '') or '')
                open_trade.broker_status = str(getattr(order, 'status', '') or '')
                open_trade.broker_side = 'SELL'
                open_trade.broker_updated_at = timezone.now()
                open_trade.save(update_fields=['broker_order_id', 'broker_status', 'broker_side', 'broker_updated_at'])
                closed += 1
            continue

        if existing_pos and signal < sell_threshold:
            qty = int(abs(_alpaca_position_qty(existing_pos)))
            if qty <= 0:
                continue
            if use_limit_orders and price:
                order = submit_limit_order(symbol, qty, 'sell', max(0.01, price - limit_offset))
            else:
                order = submit_market_order(symbol, qty, 'sell')
            if order is None:
                _send_telegram_alert(
                    f"⚠️ Order sell failed {symbol} qty {qty} (Alpaca)",
                    allow_during_blackout=True,
                    category='order_error',
                )
                continue
            if open_trade:
                open_trade.broker_order_id = str(getattr(order, 'id', '') or '')
                open_trade.broker_status = str(getattr(order, 'status', '') or '')
                open_trade.broker_side = 'SELL'
                open_trade.broker_updated_at = timezone.now()
                open_trade.save(update_fields=['broker_order_id', 'broker_status', 'broker_side', 'broker_updated_at'])
            closed += 1
            continue

        if existing_pos:
            continue
        if signal < buy_threshold:
            continue

        frame = _alpaca_fusion(symbol)
        tier, tier_reasons = _classify_tier(frame, signal_payload or {})
        atr_val = _atr_from_frame(frame) or 0.0

        reentry_size_factor = 1.0
        reentry_stop_minutes = int(os.getenv('REENTRY_STOPLOSS_WINDOW_MINUTES', '120'))
        reentry_stop_min_signal = float(os.getenv('REENTRY_STOPLOSS_MIN_SIGNAL', '0.8'))
        if reentry_stop_minutes > 0:
            cutoff = timezone.now() - timedelta(minutes=reentry_stop_minutes)
            recent_stop = PaperTrade.objects.filter(
                ticker__iexact=symbol,
                sandbox=sandbox,
                status='CLOSED',
                outcome='LOSS',
                exit_date__gte=cutoff,
                notes__icontains='Stop Loss',
            ).exists()
            if recent_stop and signal >= reentry_stop_min_signal:
                reentry_size_factor = float(os.getenv('REENTRY_STOPLOSS_SIZE_FACTOR', '0.75'))
        if reentry_candidate:
            reentry_size_factor = float(os.getenv('REENTRY_STOPLOSS_SIZE_FACTOR', '0.75'))

        position_cap = buying_power * position_cap_pct
        risk_budget = buying_power * risk_pct
        position_value = _dynamic_position_target(float(equity_now or buying_power or 0))
        position_value = min(position_value, position_cap, risk_budget, buying_power)
        position_value *= _multi_model_boost(symbol, float(signal or 0), universe, True)
        position_value *= reentry_size_factor
        if tier == 'T1':
            position_value *= tier1_size_mult
        elif tier == 'T3':
            position_value *= tier3_size_mult
        qty = int(position_value / price) if price else 0
        if qty <= 0:
            _send_telegram_alert(
                f"⚠️ Qty=0 {symbol} (buying power insuffisant)",
                allow_during_blackout=True,
                category='order_error',
            )
            continue
        _system_log(
            'SYSTEM',
            'INFO',
            f"💰 Dynamic Sizing: Equity {float(equity_now or buying_power or 0):.2f} | Risk 1.5% | Target {position_value:.2f}$ | Qty {qty}",
            symbol=symbol,
            metadata={
                'equity': round(float(equity_now or buying_power or 0), 2),
                'target_value': round(position_value, 2),
                'qty': qty,
            },
        )

        stop_distance = max(price * trail_pct, atr_mult * atr_val, price * 0.01)
        stop_loss = max(0.01, price - stop_distance)

        if shadow_mode:
            PaperTrade.objects.create(
                ticker=symbol,
                sandbox=sandbox,
                entry_price=round(price, 2),
                quantity=qty,
                entry_signal=signal,
                entry_features={
                    **((signal_payload or {}).get('features') or {}),
                    'tier': tier,
                    'tier_reasons': tier_reasons,
                    'alpaca_high_water': price,
                },
                entry_explanations=(signal_payload or {}).get('explanations'),
                model_name=(signal_payload or {}).get('model_name', universe),
                model_version=(signal_payload or {}).get('model_version', ''),
                broker='SHADOW',
                broker_status='SHADOW',
                broker_side='BUY',
                broker_updated_at=timezone.now(),
                stop_loss=round(stop_loss, 2),
                status='OPEN',
                pnl=0,
                notes=f"shadow | tier {tier}",
            )
            created += 1
            continue

        if manual_approval:
            limit_price = (price + limit_offset) if use_limit_orders else None
            queued = _queue_alpaca_trade_approval(
                symbol=symbol,
                sandbox=sandbox,
                qty=qty,
                price=price,
                limit_price=limit_price,
                stop_loss=stop_loss,
                signal_payload=signal_payload,
                tier=tier,
                tier_reasons=tier_reasons,
                use_limit_orders=use_limit_orders,
                notes=None,
            )
            if queued:
                _decision_log(symbol, sandbox, 'PENDING', 'manual_approval')
                continue

        if use_limit_orders:
            order = submit_limit_order(symbol, qty, 'buy', price + limit_offset)
        else:
            order = submit_market_order(symbol, qty, 'buy')
        if order is None:
            _send_telegram_alert(
                f"⚠️ Order buy failed {symbol} qty {qty} (Alpaca)",
                allow_during_blackout=True,
                category='order_error',
            )
            continue

        notes = _gemini_trade_reason(symbol, signal_payload) if allow_gemini else None
        PaperTrade.objects.create(
            ticker=symbol,
            sandbox=sandbox,
            entry_price=round(price, 2),
            quantity=qty,
            entry_signal=signal,
            entry_features={
                **((signal_payload or {}).get('features') or {}),
                'tier': tier,
                'tier_reasons': tier_reasons,
                'alpaca_high_water': price,
            },
            entry_explanations=(signal_payload or {}).get('explanations'),
            model_name=(signal_payload or {}).get('model_name', universe),
            model_version=(signal_payload or {}).get('model_version', ''),
            broker='ALPACA',
            broker_order_id=str(getattr(order, 'id', '') or ''),
            broker_status=str(getattr(order, 'status', '') or ''),
            broker_side='BUY',
            broker_updated_at=timezone.now(),
            stop_loss=round(stop_loss, 2),
            status='OPEN',
            pnl=0,
            notes=(notes or '') + f" | tier {tier}",
        )
        if os.getenv('TRADE_REASON_ALERTS', 'false').lower() in {'1', 'true', 'yes', 'y'}:
            reason = (signal_payload or {}).get('explanations') or []
            AlertEvent.objects.create(
                category='TRADE_REASON',
                message=f"{symbol} entry: {reason}",
                stock=Stock.objects.filter(symbol__iexact=symbol).first(),
            )
        _decision_log(symbol, sandbox, 'BUY', 'alpaca_order', float(signal))
        created += 1

    return {
        'created': created,
        'closed': closed,
        'buying_power': round(buying_power, 2),
        'market_regime': regime,
        'market_meta': market_meta,
    }


@shared_task
def sync_alpaca_paper_trades() -> dict[str, Any]:
    """Sync Alpaca order status and update PaperTrade rows."""
    def _infer_sandbox(symbol: str) -> str:
        sym = (symbol or '').strip().upper()
        if not sym:
            return 'WATCHLIST'
        for sandbox in ['AI_PENNY', 'AI_BLUECHIP', 'WATCHLIST']:
            watch = SandboxWatchlist.objects.filter(sandbox=sandbox).first()
            if not watch:
                continue
            symbols = [str(s).strip().upper() for s in (watch.symbols or [])]
            if sym in symbols:
                return sandbox
        return 'WATCHLIST'

    def _parse_dt(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            text = str(value).replace('Z', '+00:00')
            return datetime.fromisoformat(text)
        except Exception:
            return None

    def _enum_value(value: Any) -> str:
        if value is None:
            return ''
        if hasattr(value, 'value'):
            return str(getattr(value, 'value'))
        return str(value)

    def _sync_recent_orders(lookback_days: int = 7) -> dict[str, int]:
        created = 0
        closed = 0
        orders = get_recent_orders(lookback_days=lookback_days)
        if not orders:
            return {'created': created, 'closed': closed}
        def _order_time(order: Any) -> datetime:
            dt = _parse_dt(getattr(order, 'filled_at', None))
            if dt is None:
                dt = _parse_dt(getattr(order, 'created_at', None))
            if dt is None:
                dt = _parse_dt(getattr(order, 'updated_at', None))
            return dt or datetime.min.replace(tzinfo=timezone.utc)

        orders = sorted(list(orders), key=_order_time)
        cutoff = timezone.now() - timedelta(days=lookback_days)
        for order in orders:
            symbol = str(getattr(order, 'symbol', '') or '').strip().upper()
            if not symbol:
                continue
            status = _enum_value(getattr(order, 'status', '')).lower()
            filled_qty = getattr(order, 'filled_qty', None)
            filled_avg_price = getattr(order, 'filled_avg_price', None)
            try:
                filled_qty = float(filled_qty) if filled_qty is not None else 0.0
            except Exception:
                filled_qty = 0.0
            try:
                filled_avg_price = float(filled_avg_price) if filled_avg_price is not None else 0.0
            except Exception:
                filled_avg_price = 0.0
            if filled_qty <= 0 or filled_avg_price <= 0:
                continue
            filled_at = _parse_dt(getattr(order, 'filled_at', None)) or _parse_dt(getattr(order, 'updated_at', None))
            if filled_at and filled_at < cutoff:
                continue
            side = _enum_value(getattr(order, 'side', '')).lower()
            order_id = str(getattr(order, 'id', '') or '')
            if side == 'buy':
                existing = PaperTrade.objects.filter(broker='ALPACA', broker_order_id=order_id).first()
                if existing:
                    continue
                sandbox = _infer_sandbox(symbol)
                PaperTrade.objects.create(
                    ticker=symbol,
                    sandbox=sandbox,
                    entry_price=round(float(filled_avg_price), 2),
                    quantity=int(filled_qty),
                    entry_signal=None,
                    broker='ALPACA',
                    broker_order_id=order_id,
                    broker_status=status,
                    broker_side='BUY',
                    broker_updated_at=timezone.now(),
                    broker_filled_qty=filled_qty,
                    broker_avg_price=filled_avg_price,
                    stop_loss=round(float(filled_avg_price) * 0.96, 2),
                    status='OPEN',
                    pnl=0,
                    notes='imported from alpaca orders',
                )
                created += 1
            elif side == 'sell':
                open_trade = PaperTrade.objects.filter(
                    broker='ALPACA',
                    status='OPEN',
                    ticker__iexact=symbol,
                ).order_by('entry_date').first()
                if not open_trade:
                    continue
                open_trade.exit_price = round(float(filled_avg_price), 2)
                open_trade.exit_date = timezone.now()
                open_trade.status = 'CLOSED'
                open_trade.pnl = float(open_trade.exit_price - float(open_trade.entry_price)) * float(open_trade.quantity)
                open_trade.outcome = 'WIN' if float(open_trade.pnl or 0) > 0 else 'LOSS'
                open_trade.broker_status = status
                open_trade.broker_side = 'SELL'
                open_trade.broker_updated_at = timezone.now()
                open_trade.save(update_fields=[
                    'exit_price',
                    'exit_date',
                    'status',
                    'pnl',
                    'outcome',
                    'broker_status',
                    'broker_side',
                    'broker_updated_at',
                ])
                closed += 1
        return {'created': created, 'closed': closed}

    lookback_days = int(os.getenv('ALPACA_SYNC_LOOKBACK_DAYS', '7'))
    _sync_recent_orders(lookback_days=lookback_days)
    updated = 0
    trades = PaperTrade.objects.filter(broker='ALPACA', status='OPEN')
    for trade in trades:
        if not trade.broker_order_id:
            continue
        order = get_order_by_id(trade.broker_order_id)
        if order is None:
            continue
        status = str(getattr(order, 'status', '') or '')
        filled_qty = getattr(order, 'filled_qty', None) or getattr(order, 'filled_qty', None)
        filled_avg_price = getattr(order, 'filled_avg_price', None)
        trade.broker_status = status
        trade.broker_updated_at = timezone.now()
        try:
            if filled_qty is not None:
                trade.broker_filled_qty = float(filled_qty)
        except Exception:
            pass
        try:
            if filled_avg_price is not None:
                trade.broker_avg_price = float(filled_avg_price)
        except Exception:
            pass

        if status.lower() == 'filled' and trade.broker_side == 'BUY' and filled_avg_price:
            trade.entry_price = round(float(filled_avg_price), 2)
        if status.lower() == 'filled' and trade.broker_side == 'SELL' and filled_avg_price:
            trade.exit_price = round(float(filled_avg_price), 2)
            trade.exit_date = timezone.now()
            trade.status = 'CLOSED'
            trade.pnl = float(trade.exit_price - trade.entry_price) * float(trade.quantity)
            trade.outcome = 'WIN' if float(trade.pnl or 0) > 0 else 'LOSS'
        trade.save()
        updated += 1

    return {'updated': updated}


@shared_task
def execute_alpaca_paper_trades_watchlist() -> dict[str, Any]:
    return _execute_alpaca_paper_trades_for_sandbox('WATCHLIST', 'ALPACA_WATCHLIST')


@shared_task
def execute_alpaca_paper_trades_ai_bluechip() -> dict[str, Any]:
    return _execute_alpaca_paper_trades_for_sandbox('AI_BLUECHIP', 'ALPACA_BLUECHIP')


@shared_task
def execute_alpaca_paper_trades_ai_penny() -> dict[str, Any]:
    return _execute_alpaca_paper_trades_for_sandbox('AI_PENNY', 'ALPACA_PENNY')


@shared_task
def execute_paper_trades() -> dict[str, Any]:
    return _execute_paper_trades_for_sandbox('WATCHLIST', 'PAPER')


@shared_task
def execute_paper_trades_ai_bluechip() -> dict[str, Any]:
    return _execute_paper_trades_for_sandbox('AI_BLUECHIP', 'AI_BLUECHIP')


@shared_task
def execute_paper_trades_ai_penny() -> dict[str, Any]:
    return _execute_paper_trades_for_sandbox('AI_PENNY', 'AI_PENNY')


def _default_scanner_symbols() -> list[str]:
    return [
        'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'AMD', 'INTC',
        'QCOM', 'NFLX', 'ASML', 'AMAT', 'ADBE', 'CSCO', 'TMUS', 'ORCL', 'CRM', 'PANW',
        'SNOW', 'PLTR', 'MU', 'SMCI', 'TSM', 'COIN', 'SOFI', 'RIVN', 'LCID', 'NIO',
        'SHOP', 'SQ', 'PYPL', 'UBER', 'LYFT', 'ABNB', 'ROKU', 'PINS', 'DKNG', 'HOOD',
        'MARA', 'RIOT', 'HUT', 'HIVE', 'BITF', 'CLSK', 'MSTR', 'PLUG', 'FCEL', 'SOUN',
        'SNDL', 'RIG',
        'HIVE.V', 'DMGI.V',
        'WEED.TO', 'BITF.TO', 'NINE.TO',
        'RY.TO', 'TD.TO', 'SHOP.TO', 'ATD.TO', 'ENB.TO',
    ]


@shared_task
def auto_discover_top_movers(min_score: float | None = None) -> dict[str, Any]:
    """Discover active TSX/NASDAQ movers and run the scanner on them."""
    screener_ids = [
        s.strip()
        for s in os.getenv('TOP_MOVER_SCREENERS', 'day_gainers,most_actives').split(',')
        if s.strip()
    ]
    tsx_hot = [
        'SHOP.TO', 'HIVE.V', 'BITF.TO', 'WEED.TO', 'LSPD.TO',
    ]
    fallback_hot = ['NVDA', 'TSLA', 'MARA', 'RIOT', 'HIVE']
    discovered: list[str] = []
    try:
        quotes = _fetch_yfinance_screeners(screener_ids, count=25)
        gainers = [
            str(item.get('symbol') or '').strip().upper()
            for item in quotes
            if item.get('symbol')
        ]
        discovered.extend(gainers[:5])
    except Exception:
        discovered.extend([])

    discovered.extend(tsx_hot)
    if not discovered:
        discovered.extend(fallback_hot)

    discovered = [s for s in dict.fromkeys(discovered) if s]
    if not discovered:
        return {'status': 'empty', 'count': 0, 'symbols': []}

    payload = market_scanner_task(symbols=discovered)
    try:
        threshold = float(min_score if min_score is not None else os.getenv('SCANNER_MIN_SCORE', '0.8'))
        results = payload.get('results') or []
        filtered = [r for r in results if float(r.get('score') or 0) >= threshold]
        if filtered:
            top = filtered[0]
            send_telegram_signal(
                top['symbol'],
                score=float(top.get('score') or 0) * 100,
                diagnostic=f"Top Movers RVOL {top.get('rvol')}",
                deepseek_score=None,
            )
        payload['filtered'] = filtered[:5]
        payload['min_score'] = threshold
    except Exception:
        pass
    payload['symbols'] = discovered
    payload['count'] = len(payload.get('results') or [])
    payload['status'] = payload.get('status') or 'ok'
    return payload


@shared_task
def bluechip_dip_scanner() -> dict[str, Any]:
    """Find bluechips that dipped unjustly and may rebound."""
    if os.getenv('BLUECHIP_DIP_ENABLED', 'true').lower() not in {'1', 'true', 'yes', 'y'}:
        return {'status': 'disabled'}

    screener_ids = [
        s.strip()
        for s in os.getenv('BLUECHIP_DIP_SCREENERS', 'day_losers,most_actives').split(',')
        if s.strip()
    ]
    min_price = float(os.getenv('BLUECHIP_DIP_MIN_PRICE', '10'))
    max_price = float(os.getenv('BLUECHIP_DIP_MAX_PRICE', '500'))
    min_score = float(os.getenv('BLUECHIP_DIP_MIN_SCORE', '0.65'))
    limit = int(os.getenv('BLUECHIP_DIP_LIMIT', '40'))
    require_rsi_divergence = os.getenv('BLUECHIP_REQUIRE_RSI_DIVERGENCE', 'true').lower() in {'1', 'true', 'yes', 'y'}

    quotes = _fetch_yfinance_screeners(screener_ids, count=limit)
    candidates: list[dict[str, Any]] = []
    for item in quotes:
        symbol = (item.get('symbol') or '').strip().upper()
        price = item.get('regularMarketPrice') or item.get('price') or item.get('lastPrice')
        try:
            price = float(price)
        except Exception:
            price = None
        if not symbol or price is None:
            continue
        if not (min_price <= price <= max_price):
            continue

        ctx_5m = _intraday_context_for_timeframe(
            symbol,
            minutes=int(os.getenv('ALPACA_INTRADAY_MINUTES', '390')),
            timeframe=int(os.getenv('REVERSAL_TIMEFRAME', '5')),
            rvol_window=int(os.getenv('ALPACA_RVOL_WINDOW', '20')),
        )
        if require_rsi_divergence and not _rsi_divergence(ctx_5m):
            continue

        score, rsi, lower = _mean_reversion_score(symbol)
        score = _canadian_boost(score, symbol)
        if score < min_score:
            continue
        candidates.append({
            'symbol': symbol,
            'price': round(price, 2),
            'score': round(score, 4),
            'rsi': None if rsi is None else round(float(rsi), 2),
            'lower_band': None if lower is None else round(float(lower), 2),
        })

    candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
    if candidates:
        cache.set('bluechip_dip_candidates', candidates, timeout=60 * 30)
        SandboxWatchlist.objects.update_or_create(
            sandbox='AI_BLUECHIP',
            defaults={'symbols': [row['symbol'] for row in candidates[:25]], 'source': 'bluechip_dip_scanner'},
        )
    _system_log('AI_BLUECHIP', 'SUCCESS', f'Dip scanner: {len(candidates)} candidates', metadata={'count': len(candidates)})
    return {'status': 'ok', 'count': len(candidates), 'results': candidates[:10]}


@shared_task
def penny_opportunity_scanner() -> dict[str, Any]:
    """Find penny stocks with unusual dips or momentum and refresh AI_PENNY watchlist."""
    if os.getenv('PENNY_SCANNER_ENABLED', 'true').lower() not in {'1', 'true', 'yes', 'y'}:
        return {'status': 'disabled'}

    screener_ids = [
        s.strip()
        for s in os.getenv('PENNY_SCAN_SCREENERS', 'small_cap_gainers,day_gainers,most_actives').split(',')
        if s.strip()
    ]
    min_price = float(os.getenv('PENNY_MIN_PRICE', '0.10'))
    max_price = float(os.getenv('PENNY_MAX_PRICE', '5.0'))
    min_volume = float(os.getenv('PENNY_MIN_VOLUME', '100000'))
    min_score = float(os.getenv('PENNY_SCAN_MIN_SCORE', '0.7'))
    limit = int(os.getenv('PENNY_SCAN_LIMIT', '80'))
    require_hammer = os.getenv('PENNY_REQUIRE_HAMMER', 'true').lower() in {'1', 'true', 'yes', 'y'}
    min_rsi = float(os.getenv('PENNY_MEANREV_RSI_MAX', '30'))
    min_volume_z = float(os.getenv('PENNY_MEANREV_VOLUME_Z', '2'))
    min_drawdown = float(os.getenv('PENNY_MEANREV_DRAWDOWN', '-0.30'))
    strong_news_sent = float(os.getenv('PENNY_MEANREV_NEWS_SENT', '0.5'))

    quotes = _fetch_yfinance_screeners(screener_ids, count=limit)
    base_candidates: list[dict[str, Any]] = []
    for item in quotes:
        symbol = (item.get('symbol') or '').strip().upper()
        price = item.get('regularMarketPrice') or item.get('price') or item.get('lastPrice')
        volume = item.get('regularMarketVolume') or item.get('volume') or 0
        try:
            price = float(price)
        except Exception:
            price = None
        try:
            volume = float(volume)
        except Exception:
            volume = 0.0
        if not symbol or price is None:
            continue
        if not (min_price <= price <= max_price):
            continue
        if volume < min_volume:
            continue

        base_candidates.append({'symbol': symbol, 'price': price, 'volume': volume})

    candidates: list[dict[str, Any]] = []
    if base_candidates:
        from concurrent.futures import ThreadPoolExecutor

        def _enrich(symbol: str) -> dict[str, Any] | None:
            snapshot = _penny_mean_reversion_snapshot(symbol)
            if not snapshot:
                return None
            ctx_5m = _intraday_context_for_timeframe(
                symbol,
                minutes=int(os.getenv('ALPACA_INTRADAY_MINUTES', '390')),
                timeframe=int(os.getenv('REVERSAL_TIMEFRAME', '5')),
                rvol_window=int(os.getenv('ALPACA_RVOL_WINDOW', '20')),
            )
            if require_hammer and not _has_hammer_pattern(ctx_5m):
                return None

            score, rsi, lower = _mean_reversion_score(symbol)
            score = _canadian_boost(score, symbol)
            rsi_val = snapshot.get('rsi')
            volume_z = snapshot.get('volume_z')
            drawdown = snapshot.get('drawdown')
            news_sentiment = snapshot.get('news_sentiment')

            if rsi_val is None or rsi_val > min_rsi:
                return None
            if volume_z is None or volume_z < min_volume_z:
                return None
            if drawdown is None or drawdown > min_drawdown:
                return None

            if news_sentiment is not None and news_sentiment > strong_news_sent and rsi_val < min_rsi:
                score = min(1.0, score + 0.15)

            if score < min_score:
                return None

            return {
                'symbol': symbol,
                'price': round(float(snapshot.get('price') or 0), 4),
                'score': round(score, 4),
                'rsi': None if rsi_val is None else round(float(rsi_val), 2),
                'lower_band': None if lower is None else round(float(lower), 4),
                'volume_z': None if volume_z is None else round(float(volume_z), 2),
                'distance_from_ma200': None if snapshot.get('distance_from_ma200') is None else round(float(snapshot.get('distance_from_ma200')), 4),
                'rsi_edge': None if snapshot.get('rsi_edge') is None else round(float(snapshot.get('rsi_edge')), 3),
                'drawdown_52w': None if drawdown is None else round(float(drawdown) * 100, 2),
                'news_sentiment': None if news_sentiment is None else round(float(news_sentiment), 3),
                'news_titles': snapshot.get('news_titles') or [],
                'target_price': None if snapshot.get('target_price') is None else round(float(snapshot.get('target_price')), 4),
            }

        max_workers = int(os.getenv('PENNY_SCAN_WORKERS', '8'))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(lambda x: _enrich(x['symbol']), base_candidates[:limit]):
                if result:
                    candidates.append(result)

    candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
    if candidates:
        cache.set('penny_opportunity_candidates', candidates, timeout=60 * 30)
        SandboxWatchlist.objects.update_or_create(
            sandbox='AI_PENNY',
            defaults={'symbols': [row['symbol'] for row in candidates[:25]], 'source': 'penny_opportunity_scanner'},
        )
    _system_log('AI_PENNY', 'SUCCESS', f'Penny scanner: {len(candidates)} candidates', metadata={'count': len(candidates)})
    return {'status': 'ok', 'count': len(candidates), 'results': candidates[:10]}


@shared_task
def market_scanner_task(symbols: list[str] | None = None) -> dict[str, Any]:
    """Scan market for high-momentum candidates and cache results."""
    if _market_closed_now():
        return _afterhours_market_scan(symbols)
    min_price = float(os.getenv('SCANNER_MIN_PRICE', '0.5'))
    max_price = float(os.getenv('SCANNER_MAX_PRICE', '10'))
    min_volume = float(os.getenv('SCANNER_MIN_VOLUME', '500000'))
    min_change = float(os.getenv('SCANNER_MIN_CHANGE_PCT', '2'))
    min_rvol = float(os.getenv('SCANNER_MIN_RVOL', '2.5'))
    limit = int(os.getenv('SCANNER_LIMIT', '50'))
    minutes = int(os.getenv('SCANNER_INTRADAY_MINUTES', '180'))
    timeframe_5m = int(os.getenv('SCANNER_TIMEFRAME_5M', '5'))
    timeframe_15m = int(os.getenv('SCANNER_TIMEFRAME_15M', '15'))
    min_confidence = float(os.getenv('SCANNER_MIN_CONFIDENCE', '65'))
    swing_target_pct = float(os.getenv('SWING_TARGET_PCT', '0.08'))
    swing_min_rvol_target_pct = float(os.getenv('SWING_MIN_RVOL_TARGET_PCT', '0.05'))
    swing_stop_pct = float(os.getenv('SWING_STOP_PCT', '0.03'))
    investment = float(os.getenv('SWING_INVESTMENT', '200'))
    buy_limit_buffer = float(os.getenv('SWING_BUY_LIMIT_BUFFER', '0.005'))
    update_watchlist = os.getenv('AI_SCANNER_UPDATE_WATCHLIST', 'true').lower() in {'1', 'true', 'yes', 'y'}
    notify = os.getenv('AI_SCANNER_TELEGRAM', 'false').lower() in {'1', 'true', 'yes', 'y'}

    symbols_env = os.getenv('SCANNER_SYMBOLS', '').strip()
    if symbols:
        symbols = [s.strip().upper() for s in symbols if s and str(s).strip()]
    elif symbols_env:
        symbols = [s.strip().upper() for s in symbols_env.split(',') if s.strip()]
    else:
        symbols = get_tradable_symbols(limit=800)

    if not symbols:
        return {'status': 'empty', 'count': 0}

    candidates: list[dict[str, Any]] = []
    chunk_size = 200
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        snapshots = get_stock_snapshots(chunk)
        for symbol, snap in (snapshots or {}).items():
            try:
                latest_trade = getattr(snap, 'latest_trade', None) or getattr(snap, 'latestTrade', None)
                daily_bar = getattr(snap, 'daily_bar', None) or getattr(snap, 'dailyBar', None)
                if latest_trade is None or daily_bar is None:
                    continue
                price = float(getattr(latest_trade, 'price', 0) or 0)
                prev_close = float(getattr(daily_bar, 'close', 0) or 0)
                volume = float(getattr(daily_bar, 'volume', 0) or 0)
                if price <= 0 or prev_close <= 0:
                    continue
                change_pct = ((price - prev_close) / prev_close) * 100
                if not (min_price <= price <= max_price):
                    continue
                if volume < min_volume or change_pct < min_change:
                    continue
                candidates.append({
                    'symbol': symbol,
                    'price': price,
                    'change_pct': round(change_pct, 2),
                    'volume': int(volume),
                })
            except Exception:
                continue

    candidates = sorted(candidates, key=lambda x: x['change_pct'], reverse=True)[:limit]
    results: list[dict[str, Any]] = []
    market_ctx = get_intraday_context('QQQ', minutes=minutes) or {}
    market_ok = float(market_ctx.get('ema20') or 0) >= float(market_ctx.get('ema50') or 0)

    for candidate in candidates:
        symbol = candidate['symbol']
        ctx_5m = _intraday_context_for_timeframe(symbol, minutes=minutes, timeframe=timeframe_5m, rvol_window=20)
        ctx_15m = _intraday_context_for_timeframe(symbol, minutes=minutes, timeframe=timeframe_15m, rvol_window=20)
        if not ctx_5m or not ctx_15m:
            continue
        rvol = float(ctx_5m.get('rvol') or 0)
        patterns = ctx_5m.get('patterns') or []
        pattern_signal_15 = float(ctx_15m.get('pattern_signal') or 0)
        if rvol < min_rvol:
            continue
        if pattern_signal_15 <= 0:
            if rvol > 3:
                _queue_telegram_candidate({
                    'ticker': symbol,
                    'score': 0,
                    'diagnostic': f"Pattern négatif (RVOL {rvol:.2f})",
                    'ts': timezone.now().isoformat(),
                })
            continue
        if pattern_signal_15 <= 0.5:
            continue
        if not {'Hammer', 'Bullish Engulfing'}.intersection(set(patterns)):
            continue

        base_score = 0.5 + (pattern_signal_15 * 0.1) + (0.05 if rvol >= 4 else 0.0)
        confidence_pct = base_score * 100
        news_sentiment, news_titles = _news_sentiment_score(symbol, days=1)
        news_note = None
        if news_titles:
            if news_sentiment > 0.3:
                confidence_pct *= 1.1
        else:
            confidence_pct *= 0.85
            news_note = '⚠️ Mouvement purement technique (Attention aux faux rebonds)'
        if not market_ok:
            confidence_pct *= 0.95
        confidence_pct = max(0.0, min(95.0, confidence_pct))
        if confidence_pct < min_confidence:
            continue

        score = round(confidence_pct / 100, 4)
        results.append({
            **candidate,
            'rvol': round(rvol, 2),
            'patterns': patterns,
            'pattern_signal': round(pattern_signal_15, 2),
            'market_ok': market_ok,
            'score': score,
            'confidence_pct': round(confidence_pct, 2),
            'news_titles': news_titles[:3],
            'news_sentiment': round(news_sentiment, 3),
            'news_note': news_note,
        })

    if results:
        cache.set('market_scanner_results', results, timeout=60 * 60 * 24)

    if update_watchlist and results:
        SandboxWatchlist.objects.update_or_create(
            sandbox='AI_PENNY',
            defaults={'symbols': [entry['symbol'] for entry in results[:25]]},
        )
        if os.getenv('AI_SCANNER_UPDATE_WATCHLIST_MAIN', 'false').lower() in {'1', 'true', 'yes', 'y'}:
            main_limit = int(os.getenv('AI_SCANNER_MAIN_LIMIT', '15'))
            SandboxWatchlist.objects.update_or_create(
                sandbox='WATCHLIST',
                defaults={'symbols': [entry['symbol'] for entry in results[:max(1, main_limit)]]},
            )

    if notify and results:
        top = results[0]
        entry_price = float(top['price'])
        buy_limit = entry_price * (1 + buy_limit_buffer)
        target_pct = swing_target_pct
        if float(top.get('rvol') or 0) > 3:
            target_pct = max(target_pct, swing_min_rvol_target_pct)
        target_price = buy_limit * (1 + target_pct)
        stop_pct = swing_stop_pct
        master_entry = MasterWatchlistEntry.objects.filter(symbol__iexact=top['symbol']).first()
        if master_entry and master_entry.category == 'HIGH_VOL' and master_entry.stop_loss_pct:
            stop_pct = float(master_entry.stop_loss_pct)
        stop_price = buy_limit * (1 - stop_pct)
        title = (top.get('news_titles') or [''])[0] or None
        if _symbol_currency(top['symbol']) == 'CAD':
            ctx_5m = _intraday_context_for_timeframe(
                top['symbol'],
                minutes=int(os.getenv('ALPACA_INTRADAY_MINUTES', '390')),
                timeframe=int(os.getenv('REVERSAL_TIMEFRAME', '5')),
                rvol_window=int(os.getenv('ALPACA_RVOL_WINDOW', '20')),
            )
            bars = ctx_5m.get('bars') if ctx_5m else None
            rsi = _compute_rsi(bars['close']) if bars is not None and 'close' in bars else None
            sentiment = float(top.get('news_sentiment') or 0.0)
            corr, driver = _tsx_driver_correlation(top['symbol'])
            move_type = _tsx_move_type(
                rvol=float(top.get('rvol') or 0.0),
                volatility=float(ctx_5m.get('volatility') or 0.0) if ctx_5m else 0.0,
                sentiment=sentiment,
            )
            strategy = _tsx_strategy_text(
                patterns=top.get('patterns') or [],
                rvol=float(top.get('rvol') or 0.0),
                rsi=rsi,
                sentiment=sentiment,
            )
            message = _build_tsx_signal_message(
                ticker=top['symbol'],
                move_type=move_type,
                strategy=strategy,
                buy_limit=buy_limit,
                target_price=target_price,
                stop_price=stop_price,
                rsi=rsi,
                sentiment=sentiment,
                corr=corr,
                driver=driver,
            )
        else:
            message = _build_swing_signal_message(
                ticker=top['symbol'],
                confidence_pct=float(top.get('confidence_pct') or 0),
                entry_price=entry_price,
                buy_limit=buy_limit,
                target_price=target_price,
                stop_price=stop_price,
                investment=investment,
                news_title=title,
                news_source='Google News' if title else None,
                news_note=top.get('news_note'),
            )
        send_telegram_signal(
            top['symbol'],
            score=float(top.get('confidence_pct') or 0),
            diagnostic=message[:400],
            deepseek_score=None,
        )
        if not _active_signal_exists(top['symbol']):
            ActiveSignal.objects.create(
                ticker=top['symbol'],
                pattern=', '.join(top.get('patterns') or []),
                rvol=float(top.get('rvol') or 0),
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_price,
                confidence=float(top.get('confidence_pct') or 0),
                liquidity_note=top.get('news_note') or '',
                meta={
                    'buy_limit': round(buy_limit, 4),
                    'investment': investment,
                    'news_titles': top.get('news_titles') or [],
                    'news_sentiment': float(top.get('news_sentiment') or 0),
                    'timeframe': f"{timeframe_5m}m/{timeframe_15m}m",
                    'mode': 'swing',
                },
            )

    return {'status': 'ok', 'count': len(results), 'results': results}


@shared_task
def scan_market_for_opportunities(min_score: float | None = None) -> dict[str, Any]:
    """Wrapper around market_scanner_task with a configurable score threshold."""
    symbols_env = os.getenv('SCANNER_SYMBOLS', '').strip()
    symbols = None if symbols_env else _default_scanner_symbols()
    payload = market_scanner_task(symbols=symbols)
    try:
        min_score = float(min_score if min_score is not None else os.getenv('SCANNER_MIN_SCORE', '0.8'))
        results = payload.get('results') or []
        filtered = [r for r in results if float(r.get('score') or 0) >= min_score]
        if filtered:
            top = filtered[0]
            engine = DecisionEngine()
            news_titles = top.get('news_titles') or []
            ml_score = float(top.get('score') or 0) * 100
            rsi = float(top.get('rsi') or 0) if top.get('rsi') is not None else None
            rvol = float(top.get('rvol') or 0) if top.get('rvol') is not None else None
            prophet_price, prophet_rec = run_predictions(top['symbol'])
            news_sentiment = float(top.get('news_sentiment') or 0.0)
            ctx_5m = _intraday_context_for_timeframe(
                top['symbol'],
                minutes=int(os.getenv('ALPACA_INTRADAY_MINUTES', '390')),
                timeframe=int(os.getenv('REVERSAL_TIMEFRAME', '5')),
                rvol_window=int(os.getenv('ALPACA_RVOL_WINDOW', '20')),
            )
            volatility = float(ctx_5m.get('volatility') or 0.0) if ctx_5m else None
            verdict = engine.evaluate(
                top['symbol'],
                ml_score,
                rsi,
                rvol,
                news_titles,
                prophet_price=prophet_price,
                prophet_rec=prophet_rec,
                news_sentiment=news_sentiment,
                volatility=volatility,
            )
            sentiment_score = (news_sentiment + 1) * 50
            deepseek_score = float(verdict.get('score') or 0) * 10
            convergence = (ml_score + sentiment_score + deepseek_score) / 3
            payload['decision'] = {
                'symbol': top['symbol'],
                'verdict': verdict.get('verdict'),
                'deepseek_score': verdict.get('score'),
                'prophet': prophet_rec,
                'prophet_price': prophet_price,
                'volatility': volatility,
                'convergence': round(convergence, 2),
            }
            min_convergence = float(os.getenv('DECISION_CONVERGENCE_MIN', '75'))
            risk_flag = ' ⚠️ RISQUÉE' if float(verdict.get('score') or 0) < 7 else ''
            if verdict.get('verdict') == 'BUY' and convergence >= min_convergence:
                diagnostic = (verdict.get('raw') or verdict.get('verdict') or '').strip() or "Validation DeepSeek OK."
                send_telegram_signal(
                    top['symbol'],
                    score=round(convergence, 2),
                    diagnostic=f"{diagnostic}{risk_flag}",
                    deepseek_score=float(verdict.get('score') or 0),
                )
        payload['filtered'] = filtered[:5]
        payload['min_score'] = min_score
        return payload
    except Exception:
        return payload


@shared_task
def global_screener_task(limit: int | None = None) -> dict[str, Any]:
    """Global screener that finds top losers/actives and scores with DanasMLRouter."""
    screener_ids = [
        s.strip()
        for s in os.getenv('GLOBAL_SCREENER_IDS', 'day_losers,most_actives').split(',')
        if s.strip()
    ]
    min_price = float(os.getenv('GLOBAL_SCREENER_MIN_PRICE', '0.5'))
    max_price = float(os.getenv('GLOBAL_SCREENER_MAX_PRICE', '500'))
    min_volume = float(os.getenv('GLOBAL_SCREENER_MIN_VOLUME', '200000'))
    min_rvol = float(os.getenv('GLOBAL_SCREENER_MIN_RVOL', '2.0'))
    limit = int(limit if limit is not None else os.getenv('GLOBAL_SCREENER_LIMIT', '50'))
    prefer_canadian = os.getenv('PREFER_CANADIAN_SYMBOLS', 'true').lower() in {'1', 'true', 'yes', 'y'}

    quotes = _fetch_yfinance_screeners(screener_ids, count=max(200, limit * 3))
    router = DanasMLRouter()
    candidates: list[dict[str, Any]] = []

    for item in quotes:
        symbol = (item.get('symbol') or '').strip().upper()
        if not symbol:
            continue
        price = item.get('regularMarketPrice') or item.get('price') or item.get('lastPrice')
        volume = item.get('regularMarketVolume') or item.get('volume') or 0
        avg_volume = item.get('averageDailyVolume10Day') or item.get('averageVolume') or 0
        try:
            price = float(price)
        except Exception:
            price = None
        try:
            volume = float(volume)
        except Exception:
            volume = 0.0
        try:
            avg_volume = float(avg_volume)
        except Exception:
            avg_volume = 0.0
        if price is None or not (min_price <= price <= max_price):
            continue
        if volume < min_volume:
            continue
        rvol = (volume / avg_volume) if avg_volume else 0.0
        if rvol and rvol < min_rvol:
            continue

        ml = router.predict(symbol)
        candidates.append({
            'symbol': symbol,
            'price': round(price, 4),
            'volume': int(volume),
            'rvol': round(rvol, 3),
            'signal': ml.get('signal'),
            'confidence': ml.get('confidence'),
        })

    def _sort_key(item: dict[str, Any]) -> tuple:
        sym = item.get('symbol') or ''
        is_ca = sym.endswith('.TO') or sym.endswith('.V')
        return (1 if (prefer_canadian and is_ca) else 0, float(item.get('confidence') or 0))

    candidates.sort(key=_sort_key, reverse=True)
    candidates = candidates[:limit]
    cache.set('global_screener_candidates', candidates, timeout=60 * 30)

    if candidates and os.getenv('GLOBAL_SCREENER_TELEGRAM', 'true').lower() in {'1', 'true', 'yes', 'y'}:
        lines = ["🌍 Global Screener (Top Opportunités)"]
        for row in candidates[:5]:
            lines.append(
                f"• {row.get('symbol')}: conf {float(row.get('confidence') or 0):.1f}% "
                f"RVOL {row.get('rvol')} price {row.get('price')}"
            )
        _send_telegram_alert("\n".join(lines), allow_during_blackout=True, category='report')

    return {'status': 'ok', 'count': len(candidates), 'results': candidates}


def get_full_market_tickers(limit: int | None = None) -> list[str]:
    def _read_ticker_file(path: str) -> list[str]:
        if not path:
            return []
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                raw = handle.read()
            parts = [p.strip().upper() for p in re.split(r"[\n,;\t ]+", raw) if p.strip()]
            return [p for p in parts if p]
        except Exception:
            return []

    def _write_ticker_file(path: str, symbols: list[str]) -> None:
        if not path or not symbols:
            return
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write("\n".join(symbols))
        except Exception:
            return

    us_url = os.getenv(
        'US_TICKERS_URL',
        'https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed_csv/data/7660c753dc68d118ee2a9022e355262e/nasdaq-listed_csv.csv',
    )
    us_file = os.getenv('US_TICKERS_FILE', '').strip()
    ca_file = os.getenv('CA_TICKERS_FILE', '').strip()

    ca_symbols = _read_ticker_file(ca_file)
    if not ca_symbols:
        ca_symbols = [s.strip().upper() for s in os.getenv('CA_TICKERS', '').split(',') if s.strip()]
        if ca_symbols:
            _write_ticker_file(ca_file, ca_symbols)

    us_symbols_file = _read_ticker_file(us_file)
    if us_symbols_file:
        symbols = us_symbols_file
    else:
        symbols = []
    symbols: list[str] = []
    if not symbols:
        try:
            us_df = pd.read_csv(us_url)
            us_symbols = [str(s).strip().upper() for s in us_df.get('Symbol', []) if str(s).strip()]
            symbols.extend(us_symbols)
            _write_ticker_file(us_file, us_symbols)
        except Exception:
            pass

    if ca_symbols:
        symbols.extend([_normalize_yf_symbol(s) for s in ca_symbols])

    # Deduplicate, keep order
    symbols = [s for s in dict.fromkeys(symbols) if s and s.isalnum() or '.' in s]
    if limit:
        symbols = symbols[:limit]
    return symbols


@shared_task
def master_market_scanner_task() -> dict[str, Any]:
    """Scan US+CA universe with RVOL, Hammer/RSI divergence, then DeepSeek validation."""
    if os.getenv('MASTER_SCANNER_ENABLED', 'false').lower() not in {'1', 'true', 'yes', 'y'}:
        return {'status': 'disabled'}

    universe_limit = int(os.getenv('MASTER_SCANNER_UNIVERSE_LIMIT', '5000'))
    rvol_min = float(os.getenv('MASTER_SCANNER_MIN_RVOL', '1.5'))
    change_min = float(os.getenv('MASTER_SCANNER_MIN_CHANGE', '3'))
    chunk_size = int(os.getenv('MASTER_SCANNER_CHUNK_SIZE', '200'))
    candidate_limit = int(os.getenv('MASTER_SCANNER_CANDIDATES', '100'))
    final_limit = int(os.getenv('MASTER_SCANNER_FINAL_COUNT', '5'))

    require_hammer = os.getenv('PENNY_REQUIRE_HAMMER', 'true').lower() in {'1', 'true', 'yes', 'y'}
    require_div = os.getenv('BLUECHIP_REQUIRE_RSI_DIVERGENCE', 'true').lower() in {'1', 'true', 'yes', 'y'}

    tickers = get_full_market_tickers(limit=universe_limit)
    if not tickers:
        return {'status': 'empty', 'count': 0}

    candidates: list[dict[str, Any]] = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            data = yfin.download(chunk, period='2d', interval='5m', group_by='ticker', threads=True, progress=False)
        except Exception:
            continue

        for symbol in chunk:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    df = data[symbol]
                else:
                    df = data
                if df is None or df.empty or 'Close' not in df or 'Volume' not in df:
                    continue
                df = df.dropna()
                if df.empty:
                    continue
                price = float(df['Close'].iloc[-1])
                open_price = float(df['Open'].iloc[0]) if 'Open' in df else price
                change = ((price - open_price) / open_price * 100) if open_price else 0.0
                current_vol = float(df['Volume'].tail(5).mean())
                avg_vol = float(df['Volume'].mean() or 0.0)
                rvol = (current_vol / avg_vol) if avg_vol else 0.0
                if rvol < rvol_min or abs(change) < change_min:
                    continue
                candidates.append({
                    'symbol': symbol,
                    'price': price,
                    'change': round(change, 2),
                    'rvol': round(rvol, 2),
                })
            except Exception:
                continue

    candidates.sort(key=lambda x: (abs(x.get('change') or 0), x.get('rvol') or 0), reverse=True)
    candidates = candidates[:candidate_limit]

    survivors: list[dict[str, Any]] = []
    for item in candidates:
        symbol = item['symbol']
        ctx_5m = _intraday_context_for_timeframe(
            symbol,
            minutes=int(os.getenv('ALPACA_INTRADAY_MINUTES', '390')),
            timeframe=int(os.getenv('REVERSAL_TIMEFRAME', '5')),
            rvol_window=int(os.getenv('ALPACA_RVOL_WINDOW', '20')),
        )
        if not ctx_5m:
            continue
        is_penny = float(item.get('price') or 0) < 7
        if is_penny and require_hammer and not _has_hammer_pattern(ctx_5m):
            continue
        if (not is_penny) and require_div and not _rsi_divergence(ctx_5m):
            continue
        survivors.append({**item, 'ctx': ctx_5m})

    survivors = survivors[:final_limit]
    if not survivors:
        return {'status': 'ok', 'count': 0, 'results': []}

    router = DanasMLRouter()
    engine = DecisionEngine()
    best = None
    for item in survivors:
        symbol = item['symbol']
        ctx = item.get('ctx') or {}
        rsi = float(ctx.get('rsi14') or 0) if ctx.get('rsi14') is not None else None
        rvol = float(ctx.get('rvol') or item.get('rvol') or 0)
        ml_conf = router.predict(symbol).get('confidence') or 50.0
        news_sentiment, news_titles = _news_sentiment_score(symbol, days=1)
        verdict = engine.evaluate(
            symbol,
            ml_score=float(ml_conf),
            rsi=rsi,
            rvol=rvol,
            titles=news_titles,
            news_sentiment=news_sentiment,
            volatility=float(ctx.get('volatility') or 0) if ctx else None,
        )
        deepseek_score = float(verdict.get('score') or 0) * 10
        sentiment_score = (news_sentiment + 1) * 50
        convergence = (float(ml_conf) + sentiment_score + deepseek_score) / 3
        item.update({
            'ml_confidence': ml_conf,
            'deepseek': verdict,
            'convergence': round(convergence, 2),
        })
        if verdict.get('verdict') == 'BUY' and should_send_to_telegram(convergence, verdict.get('raw') or ''):
            if best is None or convergence > float(best.get('convergence') or 0):
                best = item

    if not best:
        return {'status': 'ok', 'count': 0, 'results': []}

    diagnostic = (best.get('deepseek') or {}).get('raw') or 'Validation DeepSeek OK.'
    extra = f"Prix {best.get('price'):.2f} | Drop {best.get('change')}% | RVOL {best.get('rvol')}"
    send_telegram_signal(
        best['symbol'],
        score=float(best.get('convergence') or 0),
        diagnostic=f"{extra}\n{diagnostic[:350]}",
        deepseek_score=float((best.get('deepseek') or {}).get('score') or 0),
    )
    return {'status': 'ok', 'count': 1, 'results': [best]}


@shared_task
def update_ticker_files_task() -> dict[str, Any]:
    """Refresh US/CA ticker files from sources and save to disk."""
    us_file = os.getenv('US_TICKERS_FILE', '').strip()
    ca_file = os.getenv('CA_TICKERS_FILE', '').strip()
    us_url = os.getenv(
        'US_TICKERS_URL',
        'https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed_csv/data/7660c753dc68d118ee2a9022e355262e/nasdaq-listed_csv.csv',
    )
    updated = {'us': 0, 'ca': 0}
    try:
        us_df = pd.read_csv(us_url)
        us_symbols = [str(s).strip().upper() for s in us_df.get('Symbol', []) if str(s).strip()]
        if us_file and us_symbols:
            Path(us_file).parent.mkdir(parents=True, exist_ok=True)
            with open(us_file, 'w', encoding='utf-8') as handle:
                handle.write("\n".join(us_symbols))
            updated['us'] = len(us_symbols)
    except Exception:
        pass

    ca_symbols = [s.strip().upper() for s in os.getenv('CA_TICKERS', '').split(',') if s.strip()]
    if ca_file and ca_symbols:
        try:
            Path(ca_file).parent.mkdir(parents=True, exist_ok=True)
            with open(ca_file, 'w', encoding='utf-8') as handle:
                handle.write("\n".join(ca_symbols))
            updated['ca'] = len(ca_symbols)
        except Exception:
            pass

    return {'status': 'ok', 'updated': updated}


@shared_task
def premarket_deep_scan() -> dict[str, Any]:
    screener_ids = [
        s.strip()
        for s in os.getenv('PREMARKET_SCREENERS', 'day_gainers,day_losers,most_actives').split(',')
        if s.strip()
    ]
    min_price = float(os.getenv('PREMARKET_MIN_PRICE', '0.5'))
    max_price = float(os.getenv('PREMARKET_MAX_PRICE', '250'))
    min_volume = float(os.getenv('PREMARKET_MIN_VOLUME', '250000'))
    spread_limit = float(os.getenv('PREMARKET_MAX_SPREAD_PCT', '0.015'))
    limit = int(os.getenv('PREMARKET_LIMIT', '200'))

    quotes = _fetch_yfinance_screeners(screener_ids, count=limit)
    candidates: list[dict[str, Any]] = []
    for item in quotes:
        symbol = (item.get('symbol') or '').strip().upper()
        if not symbol:
            continue
        price = item.get('regularMarketPrice') or item.get('price') or item.get('lastPrice')
        volume = item.get('regularMarketVolume') or item.get('volume') or 0
        try:
            price = float(price)
        except Exception:
            price = None
        try:
            volume = float(volume)
        except Exception:
            volume = 0
        if price is None or price <= 0:
            continue
        if not (min_price <= price <= max_price):
            continue
        if volume < min_volume:
            continue
        spread_pct = _bid_ask_spread_pct(symbol)
        if spread_pct is not None and spread_pct > spread_limit:
            continue
        candidates.append({
            'symbol': symbol,
            'price': price,
            'volume': int(volume),
        })

    cache.set('premarket_candidates', candidates, timeout=60 * 60 * 3)
    _system_log('SYSTEM', 'SUCCESS', f'Premarket scan: {len(candidates)} candidates', metadata={'count': len(candidates)})
    return {'count': len(candidates), 'candidates': candidates}


@shared_task
def premarket_ai_prediction() -> dict[str, Any]:
    candidates = cache.get('premarket_candidates') or []
    if not candidates:
        return {'status': 'empty', 'count': 0}

    min_score = float(os.getenv('PREMARKET_MIN_SCORE', '0.65'))
    penny_max = float(os.getenv('AI_PENNY_MAX_PRICE', '15'))
    predictions: list[dict[str, Any]] = []

    for item in candidates:
        symbol = (item.get('symbol') or '').strip().upper()
        price = item.get('price')
        if not symbol:
            continue
        universe = 'PENNY' if price is not None and float(price) <= penny_max else 'BLUECHIP'
        payload = _model_signal(symbol, universe, get_model_path(universe), use_alpaca=False)
        score = float(payload.get('signal') or 0.0) if payload else 0.0
        score = _canadian_boost(score, symbol)
        if score < min_score:
            continue
        entry = {
            'symbol': symbol,
            'score': round(score, 4),
            'price': price,
            'universe': universe,
            'features': (payload or {}).get('features') or {},
        }
        _record_ai_scan(symbol, score, 'premarket_ai_prediction')
        predictions.append(entry)
        _system_log(
            'AI_PENNY' if universe == 'PENNY' else 'AI_BLUECHIP',
            'INFO',
            f'{symbol} score {score:.4f}',
            symbol=symbol,
            metadata={'score': score, 'universe': universe},
        )
        _prediction_log(symbol, universe, score)

    predictions.sort(key=lambda x: x.get('score', 0), reverse=True)
    cache.set('premarket_predictions', predictions, timeout=60 * 60 * 3)
    return {'count': len(predictions), 'predictions': predictions[:25]}


@shared_task
def send_match_plan_report() -> dict[str, Any]:
    today_key = _ny_time_now().strftime('%Y%m%d')
    sent_key = f'match_plan_sent:{today_key}'
    if cache.get(sent_key):
        return {'status': 'skipped', 'reason': 'already_sent'}

    predictions = cache.get('premarket_predictions') or []
    if not predictions:
        _system_log('SYSTEM', 'WARNING', 'Match plan skipped: no predictions')
        return {'status': 'empty'}

    min_score = float(os.getenv('PLAN_MIN_SCORE', '0.75'))
    limit = int(os.getenv('PLAN_MAX_TICKERS', '5'))
    atr_mult = float(os.getenv('PLAN_ATR_MULT', '1.5'))
    stop_pct = float(os.getenv('PLAN_STOP_PCT', '0.03'))

    selected = [p for p in predictions if float(p.get('score') or 0) >= min_score][:limit]
    if not selected:
        _system_log('SYSTEM', 'WARNING', 'Match plan skipped: no candidates above threshold')
        return {'status': 'empty'}

    lines = [f"🚀 **PLAN DE MATCH - {timezone.now().strftime('%Y-%m-%d')}**"]
    plan_payload: list[dict[str, Any]] = []
    for item in selected:
        symbol = item['symbol']
        universe = item.get('universe') or 'BLUECHIP'
        score = float(item.get('score') or 0)
        price = _latest_price_snapshot(symbol) or item.get('price')
        if price is None:
            continue
        atr = _atr(symbol)
        stop_distance = max((atr_mult * atr) if atr else 0.0, price * stop_pct)
        stop_loss = max(0.01, price - stop_distance)
        targets = [round(price * 1.05, 4), round(price * 1.10, 4), round(price * 1.20, 4)]
        trade_type = 'SWING/LONG' if universe == 'BLUECHIP' else 'DAY TRADE'

        plan_payload.append({
            'symbol': symbol,
            'trade_type': trade_type,
            'score': score,
            'entry_price': round(price, 4),
            'targets': targets,
            'stop_loss': round(stop_loss, 4),
        })
        lines.append(
            "\n━━━━━━━━━━━━━━━\n"
            f"📈 **SYMBOLE : {symbol}**\n"
            f"🎯 Type : {trade_type} (IA: {score:.2f})\n"
            f"💰 Entrée : {price:.4f}$\n"
            f"✅ Objectifs : {targets[0]:.4f} / {targets[1]:.4f} / {targets[2]:.4f}\n"
            f"🛡️ Stop Loss : {stop_loss:.4f}$"
        )

    message = "\n".join(lines) + "\n\n⚠️ *Gérez votre risque. Ordres Bracket recommandés.*"
    _send_telegram_message(message)
    cache.set(sent_key, True, timeout=60 * 60 * 24)
    _system_log('TELEGRAM', 'SUCCESS', f'Match plan sent ({len(plan_payload)} items)', metadata={'count': len(plan_payload)})
    return {'status': 'sent', 'count': len(plan_payload), 'plan': plan_payload}


@shared_task
def cleanup_system_logs() -> dict[str, Any]:
    days = int(os.getenv('SYSTEM_LOG_RETENTION_DAYS', '30'))
    cutoff = timezone.now() - timedelta(days=days)
    deleted, _ = SystemLog.objects.filter(timestamp__lt=cutoff).delete()
    return {'deleted': deleted}


@shared_task
def task_crypto_scan() -> dict[str, Any]:
    enabled = os.getenv('CRYPTO_SCAN_ENABLED', 'true').lower() in {'1', 'true', 'yes', 'y'}
    if not enabled:
        return {'status': 'disabled'}

    results = scan_crypto_drip()
    min_score = float(os.getenv('CRYPTO_MIN_SCORE_ALERT', '0.8'))
    target_pct = float(os.getenv('CRYPTO_TARGET_PCT', '0.025'))
    oversold_threshold = float(os.getenv('CRYPTO_RSI_OVERSOLD', '25'))
    chat_id = os.getenv('TELEGRAM_CRYPTO_CHAT_ID') or os.getenv('TELEGRAM_CHAT_ID')

    alerts = []
    for item in results:
        symbol = item.get('symbol')
        if not symbol:
            continue
        score = item.get('score')
        rsi_val = float(item.get('rsi') or 0.0)
        if item.get('blocked'):
            _system_log('AI_CRYPTO', 'WARNING', f'{symbol} blocked (BTC panic)', symbol=symbol)
            continue
        if not item.get('drip'):
            continue
        if rsi_val > oversold_threshold:
            continue
        if score is None or float(score) < min_score:
            _system_log('AI_CRYPTO', 'INFO', f'{symbol} drip but score low', symbol=symbol, metadata={'score': score})
            continue
        price = float(item.get('price') or 0.0)
        target = price * (1 + target_pct)
        stop = price * (1 - max(0.005, target_pct / 2))
        message = (
            f"⚡ **CRYPTO DRIP** {symbol}\n"
            f"Score: {float(score):.2f} | RSI: {rsi_val:.1f}\n"
            f"Entrée: {price:.4f}$\n"
            f"Target: {target:.4f}$ (+{target_pct * 100:.1f}%)\n"
            f"Stop: {stop:.4f}$"
        )
        _send_telegram_message(message, chat_id=chat_id)
        _system_log('AI_CRYPTO', 'SUCCESS', f'{symbol} alert sent', symbol=symbol, metadata={'score': score})
        alerts.append({'symbol': symbol, 'score': score, 'price': price})

    return {'status': 'ok', 'alerts': alerts, 'count': len(alerts)}


@shared_task
def task_global_market_discovery() -> dict[str, Any]:
    """Backward-compatible alias for legacy task imports."""
    return scan_market_for_opportunities()


@shared_task
def monitor_active_trade() -> dict[str, Any]:
    log = _task_log_start('monitor_active_trade')
    try:
        enabled = os.getenv('TSX_GUARDIAN_ENABLED', 'true').lower() in {'1', 'true', 'yes', 'y'}
        if not enabled:
            result = {'status': 'disabled'}
            _task_log_finish(log, 'SUCCESS', result)
            return result

        now_ny = _ny_time_now()
        if now_ny.weekday() >= 5:
            result = {'status': 'weekend'}
            _task_log_finish(log, 'SUCCESS', result)
            return result

        start_time = dt_time(9, 30)
        end_time = dt_time(16, 0)
        if not (start_time <= now_ny.time() <= end_time):
            result = {'status': 'outside_market_hours'}
            _task_log_finish(log, 'SUCCESS', result)
            return result

        watch = SandboxWatchlist.objects.filter(sandbox='WATCHLIST').first()
        symbols = [str(s).strip().upper() for s in (watch.symbols if watch else []) if str(s).strip()]
        symbols = [s for s in symbols if _symbol_currency(s) == 'CAD']
        if not symbols:
            result = {'status': 'no_symbols'}
            _task_log_finish(log, 'SUCCESS', result)
            return result

        open_signals = ActiveSignal.objects.filter(status='OPEN', ticker__in=symbols)
        if not open_signals.exists():
            result = {'status': 'empty', 'checked': 0}
            _task_log_finish(log, 'SUCCESS', result)
            return result

        min_delta = float(os.getenv('TSX_GUARDIAN_MIN_DELTA', '0.10'))
        danger_imbalance = float(os.getenv('TSX_GUARDIAN_IMBALANCE_SELL', '0.5'))
        spread_limit = float(os.getenv('TSX_GUARDIAN_SPREAD_MAX', '0.02'))
        rsi_sell = float(os.getenv('TSX_GUARDIAN_RSI_SELL', '70'))
        rvol_fade = float(os.getenv('TSX_GUARDIAN_RVOL_FADE', '1.2'))
        sentiment_extend = float(os.getenv('TSX_GUARDIAN_SENTIMENT_EXTEND', '0.35'))
        extend_pct = float(os.getenv('TSX_GUARDIAN_EXTEND_PCT', '0.03'))
        trail_trigger_pct = float(os.getenv('TSX_TRAIL_TRIGGER_PCT', '0.03'))
        trail_stop_pct = float(os.getenv('TSX_TRAIL_STOP_PCT', '0.0'))
        exhaustion_wick_ratio = float(os.getenv('TSX_EXHAUSTION_WICK_RATIO', '2.0'))
        exhaustion_rvol = float(os.getenv('TSX_EXHAUSTION_RVOL', '2.5'))
        corr_inverse_threshold = float(os.getenv('TSX_CORR_INVERSE_PCT', '0.02'))
        leader_symbol = os.getenv('TSX_LEADER_SYMBOL', 'AEM.TO').strip().upper()
        leader_follower = os.getenv('TSX_LEADER_FOLLOWER', 'BTO.TO').strip().upper()
        leader_drop_threshold = float(os.getenv('TSX_LEADER_DROP_PCT', '0.02'))

        results: list[dict[str, Any]] = []
        for signal in open_signals:
            symbol = (signal.ticker or '').strip().upper()
            if not symbol:
                continue

            meta = signal.meta or {}
            if not meta.get('guardian_started'):
                _send_telegram_alert(
                    f"🛡️ Guardian actif pour {symbol}. Suivi des bougies & carnet toutes 30s.",
                    allow_during_blackout=True,
                    category='tracker',
                )
                meta['guardian_started'] = True
                signal.meta = meta
                signal.save(update_fields=['meta'])

            ctx_5m = _intraday_context_for_timeframe(
                symbol,
                minutes=int(os.getenv('ALPACA_INTRADAY_MINUTES', '390')),
                timeframe=5,
                rvol_window=int(os.getenv('ALPACA_RVOL_WINDOW', '20')),
            )
            ctx_15m = _intraday_context_for_timeframe(
                symbol,
                minutes=int(os.getenv('ALPACA_INTRADAY_MINUTES', '390')),
                timeframe=15,
                rvol_window=int(os.getenv('ALPACA_RVOL_WINDOW', '20')),
            )
            bars = ctx_5m.get('bars') if ctx_5m else None
            rsi = _compute_rsi(bars['close']) if bars is not None and 'close' in bars else None
            price = float(ctx_5m.get('last_close') or 0.0) if ctx_5m else None
            if not _is_valid_price(price):
                price = _latest_price_snapshot(symbol)
            if not _is_valid_price(price):
                continue

            currency = _symbol_currency(symbol)
            entry_price = float(signal.entry_price)
            target_price = float(signal.target_price)
            stop_price = float(signal.stop_loss)

            rvol = float(ctx_5m.get('rvol') or 0.0) if ctx_5m else 0.0
            pattern_signal = float(ctx_15m.get('pattern_signal') or 0.0) if ctx_15m else 0.0
            patterns = ctx_5m.get('patterns') or [] if ctx_5m else []
            sentiment, news_titles = _news_sentiment_score(symbol, days=1)
            imbalance = get_order_book_imbalance(symbol)
            spread_pct = _bid_ask_spread_pct(symbol)

            if price >= (entry_price * (1 + trail_trigger_pct)):
                trail_key = _cache_key(symbol, 'trail_profit')
                if not cache.get(trail_key):
                    cache.set(trail_key, True, timeout=60 * 60)
                    new_stop = entry_price if trail_stop_pct <= 0 else price * (1 - trail_stop_pct)
                    _send_telegram_alert(
                        f"✅ Trailing Profit : {symbol} touche {price:.2f} {currency}. "
                        f"Remonte ton Stop-Loss à {new_stop:.2f} {currency} pour sécuriser le profit.",
                        allow_during_blackout=True,
                        category='tracker',
                    )

            if bars is not None and not bars.empty and {'open', 'high', 'low', 'close'}.issubset(bars.columns):
                last_bar = bars.tail(1).iloc[0]
                open_p = float(last_bar.get('open') or 0.0)
                close_p = float(last_bar.get('close') or 0.0)
                high_p = float(last_bar.get('high') or 0.0)
                body = abs(close_p - open_p)
                upper_wick = max(0.0, high_p - max(open_p, close_p))
                if upper_wick > 0 and body > 0 and upper_wick >= (body * exhaustion_wick_ratio):
                    if rvol >= exhaustion_rvol and close_p < open_p:
                        wick_key = _cache_key(symbol, 'exhaustion_wick')
                        if not cache.get(wick_key):
                            cache.set(wick_key, True, timeout=60 * 60)
                            _send_telegram_alert(
                                f"⚠️ Bougie d'épuisement : rejet détecté sur {symbol}. "
                                "Mèche haute + volume rouge massif. Vends 50% de ta position maintenant.",
                                allow_during_blackout=True,
                                category='alert',
                            )

            driver = _tsx_driver_for_symbol(symbol)
            driver_drop = _leader_drop_pct(driver)
            if driver_drop is not None and driver_drop <= -abs(corr_inverse_threshold):
                corr_key = _cache_key(symbol, 'driver_inverse')
                if not cache.get(corr_key):
                    cache.set(corr_key, True, timeout=60 * 60)
                    _send_telegram_alert(
                        f"⚠️ Divergence {driver}/{symbol} : {driver} chute {driver_drop * 100:.2f}% "
                        "alors que le titre tient. Prépare-toi à sortir.",
                        allow_during_blackout=True,
                        category='alert',
                    )

            if symbol == leader_follower and leader_symbol:
                leader_drop = _leader_drop_pct(leader_symbol)
                if leader_drop is not None and leader_drop <= -abs(leader_drop_threshold):
                    alert_key = _cache_key(symbol, 'leader_drop')
                    if not cache.get(alert_key):
                        cache.set(alert_key, True, timeout=60 * 60)
                        _send_telegram_alert(
                            f"🚨 ALERTE LEADER : {leader_symbol} chute {leader_drop * 100:.2f}%. "
                            f"{symbol} suit souvent le leader. ACTION : VENDRE IMMÉDIATEMENT.",
                            allow_during_blackout=True,
                            category='alert',
                        )

            action = 'GARDER'
            level = 'update'
            reason = None
            new_target = None

            if spread_pct is not None and spread_pct >= spread_limit:
                action = 'VENDRE'
                level = 'danger'
                reason = f"Spread élevé ({spread_pct * 100:.2f}%)"
            elif imbalance is not None and imbalance <= danger_imbalance:
                action = 'VENDRE'
                level = 'danger'
                reason = f"Imbalance vente {imbalance:.2f}"
            elif rsi is not None and rsi >= rsi_sell and rvol <= rvol_fade:
                action = 'VENDRE'
                level = 'update'
                reason = f"RSI {rsi:.0f} + RVOL en baisse"
            elif sentiment is not None and sentiment >= sentiment_extend and pattern_signal >= 0.5:
                action = 'ÉTENDRE'
                level = 'boost'
                new_target = float(signal.target_price) * (1 + extend_pct)
                headline = (news_titles or [''])[0] or 'News positive'
                reason = f"{headline} + patterns bullish"

            score = _guardian_score(pattern_signal, rvol, sentiment, imbalance)
            score_key = _cache_key(symbol, 'guardian_score')
            action_key = _cache_key(symbol, 'guardian_action')
            last_score = cache.get(score_key)
            last_action = cache.get(action_key)

            urgent = level == 'danger'
            if last_score is not None and not urgent:
                if abs(float(score) - float(last_score)) < min_delta and action == last_action:
                    results.append({'symbol': symbol, 'status': 'skipped_delta'})
                    continue

            if action == 'GARDER' and not urgent:
                results.append({'symbol': symbol, 'status': 'hold'})
                cache.set(score_key, score, timeout=60 * 30)
                cache.set(action_key, action, timeout=60 * 30)
                continue

            cache.set(score_key, score, timeout=60 * 30)
            cache.set(action_key, action, timeout=60 * 30)

            if action == 'VENDRE' and level == 'danger':
                exit_price = max(entry_price, price)
                message = (
                    f"🚨 ALERTE DANGER : Grosse pression vendeuse sur {symbol} ({reason}). "
                    f"On sort à {exit_price:.2f} {currency} pour protéger le capital."
                )
            elif action == 'VENDRE':
                sell_price = max(price * 0.995, entry_price)
                message = (
                    f"⚠️ UPDATE {symbol} : Le momentum s'essouffle ({reason}). "
                    f"On sécurise le profit maintenant à {sell_price:.2f} {currency} "
                    f"au lieu d'attendre {target_price:.2f}. ACTION : VENDRE TOUT."
                )
            else:
                final_target = new_target if new_target else target_price
                message = (
                    f"🔥 UPDATE {symbol} : Sentiment renforcé ({reason}). "
                    f"On garde (HOLD) pour un swing. Nouvelle cible : {final_target:.2f} {currency}."
                )

            gemini_note = _gemini_dynamic_recommendation(
                symbol,
                action,
                {
                    'price': price,
                    'target': target_price,
                    'stop': stop_price,
                    'rsi': rsi,
                    'rvol': rvol,
                    'pattern_signal': pattern_signal,
                    'patterns': patterns,
                    'sentiment': sentiment,
                    'spread_pct': spread_pct,
                    'imbalance': imbalance,
                },
            )
            if gemini_note:
                message = f"{message}\n🧠 Recommandation IA: {gemini_note}"

            _send_telegram_alert(message, allow_during_blackout=True, category='tracker')
            results.append({'symbol': symbol, 'action': action, 'score': score})

        payload = {'status': 'ok', 'checked': len(open_signals), 'results': results}
        _task_log_finish(log, 'SUCCESS', payload)
        return payload
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


@shared_task
def monitor_active_signals() -> dict[str, Any]:
    open_signals = ActiveSignal.objects.filter(status='OPEN')
    if not open_signals.exists():
        return {'status': 'empty', 'checked': 0}

    trailing_enabled = os.getenv('TRAILING_STOP_ENABLED', 'true').lower() in {'1', 'true', 'yes', 'y'}
    swing_sell_on_target = os.getenv('SWING_SELL_ON_TARGET', 'true').lower() in {'1', 'true', 'yes', 'y'}
    break_even_pct = float(os.getenv('TRAIL_BREAK_EVEN_PCT', '0.01'))
    trail_trigger_pct = float(os.getenv('TRAIL_TRIGGER_PCT', '0.02'))
    trail_distance_pct = float(os.getenv('TRAIL_DISTANCE_PCT', '0.01'))

    symbols = [s.ticker for s in open_signals]
    snapshots = get_stock_snapshots(symbols)
    now_ny = _ny_time_now()
    market_close = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)

    closed = 0
    for signal in open_signals:
        snap = (snapshots or {}).get(signal.ticker)
        price = None
        if snap is not None:
            latest_trade = getattr(snap, 'latest_trade', None) or getattr(snap, 'latestTrade', None)
            if latest_trade is not None:
                price = float(getattr(latest_trade, 'price', 0) or 0)
        if not price:
            continue

        status = None
        message = None
        meta = signal.meta or {}
        entry_price = float(signal.entry_price or 0)
        profit_pct = ((price - entry_price) / entry_price) if entry_price else 0.0
        peak_price = float(meta.get('peak_price') or entry_price or price)

        if trailing_enabled and profit_pct >= break_even_pct and not meta.get('break_even_notified'):
            _send_telegram_alert(
                f"🛡️ SÉCURITÉ : ${signal.ticker} atteint +{break_even_pct * 100:.1f}%. "
                "Suggéré: remonter le stop-loss au prix d'entrée.",
                allow_during_blackout=True,
                category='trail',
            )
            meta['break_even_notified'] = True

        if trailing_enabled and profit_pct >= trail_trigger_pct:
            meta['trail_active'] = True

        if meta.get('trail_active') and not meta.get('reversal_notified'):
            reversal_ctx = _intraday_context_for_timeframe(
                signal.ticker,
                minutes=int(os.getenv('ALPACA_INTRADAY_MINUTES', '390')),
                timeframe=int(os.getenv('REVERSAL_TIMEFRAME', '15')),
                rvol_window=int(os.getenv('ALPACA_RVOL_WINDOW', '20')),
            )
            if reversal_ctx and float(reversal_ctx.get('pattern_signal') or 0) <= -0.5:
                _send_telegram_alert(
                    f"🚪 ALERTE SORTIE : Le momentum faiblit sur ${signal.ticker}. "
                    f"Gain actuel: {profit_pct * 100:.2f}%.",
                    allow_during_blackout=True,
                    category='trail',
                )
                meta['reversal_notified'] = True

        if meta.get('trail_active'):
            if price > peak_price:
                peak_price = price
            trail_stop = peak_price * (1 - trail_distance_pct)
            if trail_stop > float(signal.stop_loss or 0):
                signal.stop_loss = trail_stop
            meta['peak_price'] = peak_price
            meta['trail_stop'] = trail_stop
            if price <= trail_stop:
                status = 'CLOSED'
                message = (
                    f"🚪 SORTIE : ${signal.ticker} a touché le stop suiveur {trail_stop:.2f}$. "
                    f"Profit final: {profit_pct * 100:.2f}%."
                )

        if price >= float(signal.target_price) and not meta.get('target_notified'):
            if meta.get('mode') == 'swing' and swing_sell_on_target:
                status = 'TARGET'
                message = (
                    f"🔔 VENDRE MAINTENANT : ${signal.ticker} a touché la target. "
                    f"Profit actuel: {profit_pct * 100:.2f}%."
                )
            else:
                _send_telegram_alert(
                    f"💰 TARGET ATTEINTE : ${signal.ticker} est à +{profit_pct * 100:.2f}%. "
                    "On laisse courir ! Stop suiveur actif.",
                    allow_during_blackout=True,
                    category='trail',
                )
            meta['target_notified'] = True

        if status is None:
            if price <= float(signal.stop_loss):
                status = 'STOP'
                message = (
                    f"❌ STOP-LOSS TOUCHÉ sur ${signal.ticker}. Le prix est descendu à {price:.2f}$."
                )
            elif now_ny >= market_close:
                status = 'TIMEOUT'
                message = (
                    f"⏰ SIGNAL FERMÉ À 16H sur ${signal.ticker}. Prix de clôture: {price:.2f}$."
                )
            elif price >= float(signal.target_price) and not trailing_enabled:
                status = 'TARGET'
                message = (
                    f"✅ OBJECTIF ATTEINT sur ${signal.ticker} ! Le prix a touché {price:.2f}$."
                )

        if status:
            signal.status = status
            signal.closed_at = timezone.now()
            signal.closed_price = price
            signal.outcome = status
            signal.meta = meta
            signal.save(update_fields=['status', 'closed_at', 'closed_price', 'outcome', 'stop_loss', 'meta'])
            if message:
                _send_telegram_alert(message, allow_during_blackout=True, category='outcome')
            closed += 1
        else:
            signal.meta = meta
            signal.save(update_fields=['stop_loss', 'meta'])

    return {'status': 'ok', 'checked': len(symbols), 'closed': closed}


@shared_task
def generate_daily_performance_report() -> dict[str, Any]:
    """Daily recap of AI signals for the day."""
    log = _task_log_start('generate_daily_performance_report')
    try:
        today = _ny_time_now().date()
        signals = ActiveSignal.objects.filter(opened_at__date=today)
        scan_summary = _get_ai_scan_summary(today)
        if not signals.exists() and not scan_summary:
            message = (
                f"📋 **BILAN DE JOURNÉE IA** ({today})\n"
                "---\n"
                "Aucun signal généré aujourd'hui."
            )
            _send_telegram_alert(message, allow_during_blackout=True, category='report')
            payload = {'status': 'empty', 'count': 0, 'date': today.isoformat()}
            _task_log_finish(log, 'SUCCESS', payload)
            return payload

        total_signals = signals.count() if signals.exists() else len(scan_summary)
        avg_confidence = signals.aggregate(avg=models.Avg('confidence')).get('avg') if signals.exists() else None
        if avg_confidence is None and scan_summary:
            valid = [row.get('confidence') for row in scan_summary if row.get('confidence') is not None]
            avg_confidence = (sum(valid) / len(valid)) if valid else None

        def _fmt_conf(value: float | None) -> str:
            if value is None:
                return 'n/a'
            conf = float(value)
            if conf <= 1:
                conf *= 100
            return f"{conf:.1f}%"

        if signals.exists():
            top_picks = signals.order_by('-confidence', '-opened_at')[:3]
            picks_lines = [f"• {signal.ticker}: {_fmt_conf(signal.confidence)}" for signal in top_picks]
        else:
            ranked = sorted(
                [row for row in scan_summary if row.get('confidence') is not None],
                key=lambda x: x.get('confidence') or 0,
                reverse=True,
            )
            picks_lines = [f"• {row['symbol']}: {_fmt_conf(row.get('confidence'))}" for row in ranked[:3]]
        picks_str = "\n".join(picks_lines) if picks_lines else "—"

        report = (
            f"📋 **BILAN DE JOURNÉE IA** ({today})\n"
            "---\n"
            f"🔍 **Opportunités scannées :** {total_signals}\n"
            f"🧠 **Confiance moyenne :** {_fmt_conf(avg_confidence)}\n\n"
            f"🏆 **Top 3 de l'IA :**\n{picks_str}\n\n"
            "💰 **Statut Risk Manager :** Opérationnel (Limite 300$/trade)"
        )

        _send_telegram_alert(report, allow_during_blackout=True, category='report')
        payload = {
            'status': 'sent',
            'count': total_signals,
            'date': today.isoformat(),
            'avg_confidence': avg_confidence,
        }
        _task_log_finish(log, 'SUCCESS', payload)
        return payload
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


@shared_task
def send_morning_scout_report() -> dict[str, Any]:
    log = _task_log_start('send_morning_scout_report')
    try:
        channel = os.getenv('MORNING_REPORT_CHANNEL', 'telegram').lower().strip()
        use_telegram = channel in {'telegram', 'both'}
        use_email = channel in {'email', 'both'}

        has_telegram = bool(os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'))
        email_to = settings.ALERT_EMAIL_TO

        if use_telegram and not has_telegram:
            use_telegram = False
        if use_email and not email_to:
            use_email = False
        if not use_telegram and not use_email:
            result = {'status': 'skipped', 'reason': 'no alert channel configured'}
            _task_log_finish(log, 'SUCCESS', result)
            return result

        vix_threshold = float(os.getenv('VIX_STRESS_THRESHOLD', '30'))
        vix_level = _get_vix_level()
        market_stress = vix_level is not None and vix_level >= vix_threshold
        if market_stress:
            _send_alert(
                '⚠️ Market Stress High - Trading Restricted',
                f"VIX {float(vix_level or 0):.2f} >= {vix_threshold:.2f}",
            )

        sandboxes = [
            ('WATCHLIST', 'PAPER', 'BLUECHIP', 'SPY,AAPL,MSFT,NVDA,AMZN'),
            ('AI_BLUECHIP', 'AI_BLUECHIP', 'BLUECHIP', 'SPY,AAPL,MSFT,NVDA,AMZN'),
            ('AI_PENNY', 'AI_PENNY', 'PENNY', 'SIRI,PLUG,SOFI,RIOT,MARA'),
        ]

        def _fmt_money(value: float | None) -> str:
            if value is None:
                return 'n/a'
            return f"${float(value):,.2f}"

        def _fmt_qty(value: float | None) -> str:
            if value is None:
                return 'n/a'
            return f"{float(value):,.2f}"

        def _format_table(headers: list[str], rows: list[list[str]]) -> list[str]:
            if not rows:
                return []
            col_count = len(headers)
            widths = [len(h) for h in headers]
            for row in rows:
                for idx in range(col_count):
                    widths[idx] = max(widths[idx], len(str(row[idx])))
            header_line = ' | '.join(headers[i].ljust(widths[i]) for i in range(col_count))
            sep_line = '-+-'.join('-' * widths[i] for i in range(col_count))
            data_lines = [
                ' | '.join(str(row[i]).ljust(widths[i]) for i in range(col_count))
                for row in rows
            ]
            return [header_line, sep_line, *data_lines]

        def _avg_cost_for_stock(portfolio: Portfolio, stock: Stock) -> float | None:
            txs = (
                Transaction.objects.filter(portfolio=portfolio, stock=stock)
                .order_by('date')
            )
            total_shares = 0.0
            total_cost = 0.0
            for tx in txs:
                shares = float(tx.shares or 0)
                price = float(tx.price_per_share or 0)
                if tx.transaction_type == 'BUY':
                    total_cost += shares * price
                    total_shares += shares
                elif tx.transaction_type == 'SELL' and total_shares > 0:
                    avg_cost = total_cost / total_shares if total_shares else 0.0
                    total_cost -= shares * avg_cost
                    total_shares = max(0.0, total_shares - shares)
            if total_shares > 0:
                return total_cost / total_shares
            return None

        def _positions_from_account(account: Account) -> list[dict[str, Any]]:
            txs = (
                AccountTransaction.objects.filter(account=account)
                .select_related('stock')
                .order_by('date')
            )
            positions: dict[int, dict[str, Any]] = {}
            for tx in txs:
                stock = tx.stock
                if not stock:
                    continue
                pos = positions.setdefault(
                    stock.id,
                    {'stock': stock, 'shares': 0.0, 'cost': 0.0},
                )
                shares = float(tx.quantity or 0)
                price = float(tx.price or 0)
                if tx.type == 'BUY':
                    pos['cost'] += shares * price
                    pos['shares'] += shares
                elif tx.type == 'SELL' and pos['shares'] > 0:
                    avg_cost = pos['cost'] / pos['shares'] if pos['shares'] else 0.0
                    pos['cost'] -= shares * avg_cost
                    pos['shares'] = max(0.0, pos['shares'] - shares)

            results: list[dict[str, Any]] = []
            for pos in positions.values():
                if pos['shares'] > 0:
                    pos['avg_cost'] = pos['cost'] / pos['shares'] if pos['shares'] else None
                    results.append(pos)

            results.sort(key=lambda x: (x['stock'].symbol or ''))
            return results

        def _tg_safe(text: str) -> str:
            return str(text).replace('_', '\\_')

        def _format_table_block(headers: list[str], rows: list[list[str]]) -> list[str]:
            if not rows:
                return ['— Aucun titre']
            table_lines = _format_table(headers, rows)
            return ['```', *table_lines, '```']

        def _send_telegram_chunks(message: str) -> None:
            max_len = 3800
            chunk: list[str] = []
            current_len = 0
            for line in message.splitlines():
                if current_len + len(line) + 1 > max_len and chunk:
                    _send_telegram_alert("\n".join(chunk).strip(), allow_during_blackout=True, category='report')
                    chunk = [line]
                    current_len = len(line) + 1
                else:
                    chunk.append(line)
                    current_len += len(line) + 1
            if chunk:
                _send_telegram_alert("\n".join(chunk).strip(), allow_during_blackout=True, category='report')

        lines: list[str] = []
        today = timezone.now().date()
        lines.append(f"☀️ *Rapport du matin* — {today}")
        if market_stress:
            lines.append(
                f"⚠️ *Market Stress* — VIX {float(vix_level or 0):.2f} ≥ {vix_threshold:.2f}"
            )
        calendar_note = _economic_calendar_note_for_today()
        if calendar_note:
            lines.append(f"📅 {_tg_safe(calendar_note)}")
        lines.append('')

        lines.append("📈 *Performance Modèles* ")
        for model_name in ['BLUECHIP', 'PENNY']:
            active = ModelRegistry.objects.filter(model_name=model_name, status='ACTIVE').order_by('-trained_at').first()
            if active:
                win_rate = float(active.backtest_win_rate or 0)
                sharpe = float(active.backtest_sharpe or 0)
                lines.append(f"• {model_name}: WinRate {win_rate:.2f}% | Sharpe {sharpe:.2f}")
            else:
                latest_eval = (
                    ModelEvaluationDaily.objects.filter(model_name=model_name)
                    .order_by('-as_of')
                    .first()
                )
                if latest_eval:
                    win_rate = float(latest_eval.win_rate or 0) * 100
                    lines.append(
                        f"• {model_name}: WinRate {win_rate:.2f}% | (dernier suivi {latest_eval.as_of})"
                    )
                else:
                    lines.append(f"• {model_name}: Aucun modèle actif")
        lines.append('')

        payload_summary: dict[str, Any] = {}

        for sandbox, prefix, universe, default_watchlist in sandboxes:
            watchlist = _get_watchlist(sandbox, prefix, default_watchlist)
            model_path = get_model_path(universe)
            buy_threshold = _env_float(prefix, 'BUY_THRESHOLD', '0.82')
            min_volume_z = _env_float(prefix, 'VOLUME_ZSCORE_MIN', '0.5')
            use_alpaca = sandbox in {'AI_BLUECHIP', 'AI_PENNY'}

            validated: list[dict[str, Any]] = []
            excluded: list[tuple[str, str]] = []

            for symbol in watchlist:
                if not _is_valid_symbol(symbol):
                    continue
                signal_payload = _model_signal(symbol, universe, model_path, use_alpaca=use_alpaca)
                if not signal_payload:
                    continue
                signal = float(signal_payload.get('signal') or 0.0)
                if signal < buy_threshold:
                    continue
                volume_z = _safe_float((signal_payload.get('features') or {}).get('VolumeZ'))
                if volume_z is not None and volume_z < min_volume_z:
                    continue
                blacklisted, reason = _is_blacklisted(symbol)
                if blacklisted:
                    excluded.append((symbol, reason))
                    continue

                explanations = signal_payload.get('explanations') or []
                top_reason = ', '.join(
                    [f"{item['feature']} {item['contribution']}%" for item in explanations[:3]]
                )
                validated.append(
                    {
                        'symbol': symbol,
                        'signal': round(signal, 4),
                        'volume_z': None if volume_z is None else round(float(volume_z), 2),
                        'reason': top_reason,
                    }
                )

            lines.append(f"🔎 *{sandbox}*")
            lines.append(f"✅ Validées: {len(validated)}")
            if validated:
                for entry in validated[:5]:
                    extra = f" | {entry['reason']}" if entry['reason'] else ''
                    vol_txt = (
                        f" | VolZ {entry['volume_z']:.2f}" if entry['volume_z'] is not None else ''
                    )
                    lines.append(
                        f"• {_tg_safe(entry['symbol'])} | signal {entry['signal']:.2f}{vol_txt}{extra}"
                    )
            else:
                lines.append('• Aucune')

            lines.append(f"🚫 Exclues aujourd’hui: {len(excluded)}")
            if excluded:
                for symbol, reason in excluded[:5]:
                    lines.append(f"• {_tg_safe(symbol)}: {_tg_safe(reason)}")
            else:
                lines.append('• Aucune')

            lines.append('')

            payload_summary[sandbox] = {
                'validated': len(validated),
                'excluded': len(excluded),
                'watchlist': len(watchlist),
            }

        def _build_table_rows(items: list[tuple[Any, float, float | None]]) -> tuple[list[list[str]], dict[str, list[str]]]:
            table_rows: list[list[str]] = []
            news_map: dict[str, list[str]] = {}
            for stock, shares, avg_cost in items:
                latest_price = (
                    float(stock.latest_price)
                    if stock.latest_price is not None
                    else None
                )
                value_now = latest_price * shares if latest_price is not None else None
                cost_now = avg_cost * shares if avg_cost is not None else None
                pnl = (value_now - cost_now) if value_now is not None and cost_now is not None else None

                prediction = (
                    Prediction.objects.filter(stock=stock)
                    .order_by('-date')
                    .first()
                )
                projection = _fmt_money(prediction.predicted_price) if prediction else 'n/a'
                reco = prediction.recommendation if prediction else 'n/a'

                day_low = _fmt_money(stock.day_low) if stock.day_low is not None else 'n/a'
                day_high = _fmt_money(stock.day_high) if stock.day_high is not None else 'n/a'

                table_rows.append(
                    [
                        stock.symbol,
                        _fmt_qty(shares),
                        _fmt_money(latest_price),
                        _fmt_money(avg_cost),
                        _fmt_money(value_now),
                        _fmt_money(pnl),
                        f"{day_low}/{day_high}",
                        projection,
                        reco,
                    ]
                )

                news_items = (
                    StockNews.objects.filter(stock=stock)
                    .order_by('-published_at')[:1]
                )
                if news_items:
                    news_map[stock.symbol] = [
                        f"{item.headline} ({item.url})" for item in news_items
                    ]

            return table_rows, news_map

        # Portfolio details
        portfolios = Portfolio.objects.all()
        if portfolios.exists():
            lines.append('💼 *Portefeuilles*')
            for portfolio in portfolios:
                lines.append(f"*{_tg_safe(portfolio.name)}*:")
                holdings = (
                    PortfolioHolding.objects.select_related('stock')
                    .filter(portfolio=portfolio)
                    .order_by('stock__symbol')
                )
                if not holdings:
                    lines.append('— Aucun titre')
                    lines.append('')
                    continue

                items: list[tuple[Any, float, float | None]] = []
                for holding in holdings:
                    stock = holding.stock
                    shares = float(holding.shares or 0)
                    if shares <= 0:
                        continue
                    avg_cost = _avg_cost_for_stock(portfolio, stock)
                    items.append((stock, shares, avg_cost))

                table_rows, news_map = _build_table_rows(items)

                headers = ['Symbole', 'Qté', 'Prix', 'Achat', 'Valeur', 'PnL', 'Jour', 'Projection', 'Reco']
                lines.extend(_format_table_block(headers, table_rows))
                if news_map:
                    for symbol, headlines in news_map.items():
                        for headline in headlines:
                            lines.append(f"📰 {_tg_safe(symbol)}: {_tg_safe(headline)}")
                lines.append('')

        # Account details
        accounts = Account.objects.all().order_by('name')
        portfolio_accounts = accounts.filter(account_type__in=['TFSA', 'CRYPTO'])
        if portfolio_accounts.exists():
            if not portfolios.exists():
                lines.append('💼 *Portefeuilles*')
            for account in portfolio_accounts:
                lines.append(f"*{_tg_safe(account.name)}* ({_tg_safe(account.account_type)}):")
                positions = _positions_from_account(account)
                if not positions:
                    lines.append('— Aucun titre')
                    lines.append('')
                    continue

                items: list[tuple[Any, float, float | None]] = []
                for pos in positions:
                    stock = pos['stock']
                    shares = float(pos['shares'] or 0)
                    if shares <= 0:
                        continue
                    avg_cost = pos.get('avg_cost')
                    items.append((stock, shares, avg_cost))

                table_rows, news_map = _build_table_rows(items)
                headers = ['Symbole', 'Qté', 'Prix', 'Achat', 'Valeur', 'PnL', 'Jour', 'Projection', 'Reco']
                lines.extend(_format_table_block(headers, table_rows))
                if news_map:
                    for symbol, headlines in news_map.items():
                        for headline in headlines:
                            lines.append(f"📰 {_tg_safe(symbol)}: {_tg_safe(headline)}")
                lines.append('')

        accounts = accounts.exclude(id__in=portfolio_accounts.values_list('id', flat=True))
        if accounts.exists():
            lines.append('🏦 *Comptes*')
            for account in accounts:
                lines.append(f"*{_tg_safe(account.name)}* ({_tg_safe(account.account_type)}):")
                positions = _positions_from_account(account)
                if not positions:
                    lines.append('— Aucun titre')
                    lines.append('')
                    continue

                items: list[tuple[Any, float, float | None]] = []
                for pos in positions:
                    stock = pos['stock']
                    shares = float(pos['shares'] or 0)
                    if shares <= 0:
                        continue
                    avg_cost = pos.get('avg_cost')
                    items.append((stock, shares, avg_cost))

                table_rows, news_map = _build_table_rows(items)
                headers = ['Symbole', 'Qté', 'Prix', 'Achat', 'Valeur', 'PnL', 'Jour', 'Projection', 'Reco']
                lines.extend(_format_table_block(headers, table_rows))
                if news_map:
                    for symbol, headlines in news_map.items():
                        for headline in headlines:
                            lines.append(f"📰 {_tg_safe(symbol)}: {_tg_safe(headline)}")
                lines.append('')

        lines.append('ℹ️ Notes: simulation uniquement. Les recommandations sont indicatives.')

        message = '\n'.join(lines).strip()
        if use_telegram:
            _send_telegram_chunks(message)
        if use_email:
            subject = f"Daily AI Scout Report - {today}"
            send_mail(
                subject=subject,
                message=message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[email_to],
                fail_silently=True,
            )

        payload = {
            'status': 'sent',
            'vix': None if vix_level is None else round(float(vix_level), 2),
            'market_stress': market_stress,
            'summary': payload_summary,
        }
        _task_log_finish(log, 'SUCCESS', payload)
        return payload
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


@shared_task
def retrain_from_paper_trades_daily(sandbox_override: str | None = None) -> dict[str, Any]:
    """Retrain the fusion model daily from closed paper trades + fresh API/news data."""
    log = _task_log_start('retrain_from_paper_trades_daily')
    learn_days = int(os.getenv('PAPER_TRADE_LEARN_DAYS', '90'))
    min_samples = int(os.getenv('PAPER_TRADE_MIN_SAMPLES', '20'))
    min_win_improve = float(os.getenv('PAPER_TRADE_MIN_WIN_IMPROVEMENT', '0.5'))
    min_promote_trades = int(os.getenv('PAPER_TRADE_PROMOTE_MIN_TRADES', '20'))
    min_promote_win_rate = float(os.getenv('PAPER_TRADE_PROMOTE_MIN_WIN_RATE', '0.5'))
    holdout_ratio = float(os.getenv('PAPER_TRADE_HOLDOUT_RATIO', '0.2'))
    min_holdout_accuracy = float(os.getenv('PAPER_TRADE_MIN_HOLDOUT_ACCURACY', '0.55'))
    holdout_days = int(os.getenv('PAPER_TRADE_HOLDOUT_DAYS', '0'))
    min_holdout_samples = int(os.getenv('PAPER_TRADE_MIN_HOLDOUT_SAMPLES', '10'))
    sandbox = (sandbox_override or os.getenv('PAPER_TRADE_SANDBOX', 'WATCHLIST')).strip().upper() or 'WATCHLIST'
    cutoff = timezone.now() - timedelta(days=learn_days)
    def _label_from_trade(trade: PaperTrade, vol: float | None) -> int:
        try:
            entry_value = float(trade.entry_price) * float(trade.quantity)
            record_pnl = float(trade.pnl or 0)
            if entry_value <= 0:
                return 0
            ret = record_pnl / entry_value
            if vol and vol > 0:
                ret = ret / vol
            return 1 if ret > 0 else 0
        except Exception:
            return 0

    def _ensure_feature_cols(frame: pd.DataFrame) -> pd.DataFrame:
        for col in FEATURE_COLUMNS:
            if col not in frame.columns:
                frame[col] = 0.0
        return frame

    def _split_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if df.empty:
            return df, pd.DataFrame()
        df = df.sort_values('entry_date')
        if holdout_days > 0:
            cut = timezone.now() - timedelta(days=holdout_days)
            train_df = df[df['entry_date'] < cut]
            holdout_df = df[df['entry_date'] >= cut]
            return train_df, holdout_df
        split_index = int(len(df) * (1 - holdout_ratio))
        train_df = df.iloc[:split_index] if split_index > 0 else df
        holdout_df = df.iloc[split_index:] if split_index < len(df) else pd.DataFrame()
        return train_df, holdout_df

    def _train_for_sandbox(selected_sandbox: str) -> dict[str, Any]:
        model_name = 'PENNY' if selected_sandbox == 'AI_PENNY' else 'BLUECHIP'
        trades = (
            PaperTrade.objects.filter(
                status='CLOSED',
                sandbox=selected_sandbox,
                model_name=model_name,
                entry_date__gte=cutoff,
            )
            .exclude(exit_date__isnull=True)
            .order_by('entry_date')
        )
        if os.getenv('PAPER_TRADE_ONLY_WINNERS', 'false').lower() in {'1', 'true', 'yes', 'y'}:
            trades = trades.filter(outcome='WIN')
        if not trades.exists():
            return {'sandbox': selected_sandbox, 'status': 'no_trades', 'samples': 0}

        samples: list[dict[str, Any]] = []
        sample_weights: list[float] = []
        penalized = 0
        for trade in trades:
            symbol = (trade.ticker or '').strip().upper()
            if not symbol:
                continue
            entry_date = trade.entry_date or trade.exit_date
            if not entry_date:
                continue
            entry_signal = float(trade.entry_signal or 0.0)
            if trade.entry_features:
                sample = {col: float((trade.entry_features or {}).get(col, 0.0)) for col in FEATURE_COLUMNS}
                vol = float((trade.entry_features or {}).get('Volatility') or 0) or None
            else:
                fusion = DataFusionEngine(symbol)
                fusion_df = fusion.fuse_all()
                if fusion_df is None or fusion_df.empty:
                    continue
                df = fusion_df.copy().sort_index()
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                entry_ts = pd.to_datetime(entry_date)
                row = df[df.index <= entry_ts].tail(1)
                if row.empty:
                    continue
                sample = {col: float(row.iloc[0].get(col, 0.0)) for col in FEATURE_COLUMNS}
                vol = float(row.iloc[0].get('Volatility') or 0) or None
            sample['label'] = _label_from_trade(trade, vol)
            sample['entry_date'] = pd.to_datetime(entry_date)
            samples.append(sample)

            max_4h, min_4h = _price_move_after_entry(symbol, entry_date, 4)
            max_24h, min_24h = _price_move_after_entry(symbol, entry_date, 24)
            max_move = max([v for v in [max_4h, max_24h] if v is not None], default=None)
            min_move = min([v for v in [min_4h, min_24h] if v is not None], default=None)

            weight = 1.0
            if entry_signal >= 0.6:
                if max_move is not None and max_move >= 0.05:
                    weight = 2.0
                if min_move is not None and min_move <= -0.04:
                    weight = 5.0
                    penalized += 1
            if max_move is not None and min_move is not None:
                if abs(max_move) <= 0.01 and abs(min_move) <= 0.01:
                    weight = 0.5
            sample_weights.append(weight)

        samples_df = pd.DataFrame(samples)
        if samples_df.empty:
            return {'sandbox': selected_sandbox, 'status': 'no_samples', 'samples': 0}

        train_df, holdout_df = _split_time(samples_df)
        if len(train_df) < min_samples:
            return {
                'sandbox': selected_sandbox,
                'status': 'insufficient_samples',
                'samples': int(len(train_df)),
                'min_required': min_samples,
            }

        model_path = get_model_path(selected_sandbox)

        holdout_accuracy = None
        holdout_accuracy_current = None
        guard_failed = False
        holdout_ok = True
        if not holdout_df.empty and len(holdout_df) >= min_holdout_samples:
            train_df = _ensure_feature_cols(train_df.copy())
            holdout_df = _ensure_feature_cols(holdout_df.copy())
            X_train = train_df[FEATURE_COLUMNS].fillna(0).values
            y_train = train_df['label'].fillna(0).astype(int).values
            X_holdout = holdout_df[FEATURE_COLUMNS].fillna(0).values
            y_holdout = holdout_df['label'].fillna(0).astype(int).values
            if len(set(y_train)) >= 2 and len(set(y_holdout)) >= 2:
                holdout_model = RandomForestClassifier(
                    n_estimators=500,
                    max_depth=5,
                    min_samples_split=8,
                    min_samples_leaf=10,
                    random_state=42,
                )
                holdout_model.fit(X_train, y_train)
                preds = holdout_model.predict(X_holdout)
                holdout_accuracy = float((preds == y_holdout).mean())
                holdout_ok = holdout_accuracy >= min_holdout_accuracy

        payload = train_fusion_model_from_labels(
            samples_df.drop(columns=['entry_date']),
            model_path=model_path,
            save=False,
            sample_weight=sample_weights,
        )
        if not payload:
            return {'sandbox': selected_sandbox, 'status': 'failed', 'samples': int(len(train_df)), 'trained': False}

        model_version = get_model_version(payload, model_path)

        if not holdout_df.empty and len(holdout_df) >= min_holdout_samples:
            try:
                import joblib

                current_payload = None
                if model_path.exists():
                    loaded = joblib.load(model_path)
                    if isinstance(loaded, dict) and loaded.get('model') and loaded.get('features'):
                        current_payload = loaded
                    elif hasattr(loaded, 'predict'):
                        current_payload = {'model': loaded, 'features': FEATURE_COLUMNS}
                if current_payload and current_payload.get('model'):
                    feature_list = current_payload.get('features') or FEATURE_COLUMNS
                    X_holdout = holdout_df[feature_list].fillna(0).values
                    y_holdout = holdout_df['label'].fillna(0).astype(int).values
                    preds_current = current_payload['model'].predict(X_holdout)
                    holdout_accuracy_current = float((preds_current == y_holdout).mean())
                    if holdout_accuracy is not None and holdout_accuracy_current is not None:
                        if holdout_accuracy < (holdout_accuracy_current - 0.05):
                            guard_failed = True
            except Exception:
                guard_failed = False

        symbol = os.getenv('BACKTEST_SYMBOL', 'SPY').strip().upper()
        lookback_days = int(os.getenv('BACKTEST_LOOKBACK_DAYS', '60'))
        engine = DataFusionEngine(symbol)
        data = engine.fuse_all()
        if data is None or data.empty:
            return {'sandbox': selected_sandbox, 'status': 'no_data', 'samples': int(len(train_df)), 'trained': False}

        current = load_or_train_model(data, model_path=model_path)
        current_result = AIBacktester(data, current).run_simulation(lookback_days=lookback_days)
        candidate_result = AIBacktester(data, payload).run_simulation(lookback_days=lookback_days)

        improved = (
            candidate_result.win_rate >= current_result.win_rate + min_win_improve
            or candidate_result.sharpe_ratio > current_result.sharpe_ratio
        )

        paper_trades_count = trades.count()
        paper_win_rate = (
            trades.filter(outcome='WIN').count() / paper_trades_count if paper_trades_count else 0.0
        )
        paper_ok = paper_trades_count >= min_promote_trades and paper_win_rate >= min_promote_win_rate

        if guard_failed:
            improved = False
        if improved and paper_ok and holdout_ok:
            save_model_payload(payload, model_path=model_path)
            ModelRegistry.objects.filter(model_name=model_name, status='ACTIVE').update(status='ARCHIVED')
            ModelRegistry.objects.update_or_create(
                model_name=model_name,
                model_version=model_version,
                defaults={
                    'model_path': str(model_path),
                    'status': 'ACTIVE',
                    'backtest_win_rate': candidate_result.win_rate,
                    'backtest_sharpe': candidate_result.sharpe_ratio,
                    'paper_win_rate': paper_win_rate,
                    'paper_trades': paper_trades_count,
                    'notes': {
                        'sandbox': selected_sandbox,
                        'current_win_rate': current_result.win_rate,
                        'candidate_win_rate': candidate_result.win_rate,
                        'current_sharpe': current_result.sharpe_ratio,
                        'candidate_sharpe': candidate_result.sharpe_ratio,
                        'holdout_accuracy': holdout_accuracy,
                        'holdout_accuracy_current': holdout_accuracy_current,
                        'guard_failed': guard_failed,
                        'holdout_ok': holdout_ok,
                        'holdout_days': holdout_days,
                        'penalized_trades': penalized,
                    },
                },
            )
        else:
            ModelRegistry.objects.update_or_create(
                model_name=model_name,
                model_version=model_version,
                defaults={
                    'model_path': str(model_path),
                    'status': 'CANDIDATE',
                    'backtest_win_rate': candidate_result.win_rate,
                    'backtest_sharpe': candidate_result.sharpe_ratio,
                    'paper_win_rate': paper_win_rate,
                    'paper_trades': paper_trades_count,
                    'notes': {
                        'sandbox': selected_sandbox,
                        'current_win_rate': current_result.win_rate,
                        'candidate_win_rate': candidate_result.win_rate,
                        'current_sharpe': current_result.sharpe_ratio,
                        'candidate_sharpe': candidate_result.sharpe_ratio,
                        'paper_ok': paper_ok,
                        'holdout_accuracy': holdout_accuracy,
                        'holdout_ok': holdout_ok,
                        'holdout_days': holdout_days,
                    },
                },
            )

        return {
            'sandbox': selected_sandbox,
            'status': 'ok' if (improved and paper_ok and holdout_ok) else 'skipped',
            'samples': int(len(train_df)),
            'trained': bool(improved and paper_ok and holdout_ok),
            'current_win_rate': current_result.win_rate,
            'candidate_win_rate': candidate_result.win_rate,
            'paper_win_rate': paper_win_rate,
            'paper_trades': paper_trades_count,
            'holdout_accuracy': holdout_accuracy,
            'holdout_ok': holdout_ok,
        }

    if sandbox == 'ALL':
        results = []
        for sb in ['WATCHLIST', 'AI_BLUECHIP', 'AI_PENNY']:
            results.append(_train_for_sandbox(sb))
        payload = {'status': 'ok', 'results': results}
        _task_log_finish(log, 'SUCCESS', payload)
        return payload

    result = _train_for_sandbox(sandbox)
    _task_log_finish(log, 'SUCCESS', result)
    return result


def _calculate_max_drawdown(trades: list[PaperTrade]) -> float:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for trade in trades:
        equity += float(trade.pnl or 0)
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (equity - peak) / peak
            if dd < max_drawdown:
                max_drawdown = dd
    return abs(max_drawdown)


def _journal_output_dir() -> str:
    base_dir = str(getattr(settings, 'BASE_DIR', os.getcwd()))
    return os.getenv('TRADING_JOURNAL_DIR', os.path.join(base_dir, 'reports'))


def _wrap_text(text: str, max_width: float, font_name: str, font_size: int) -> list[str]:
    words = str(text).split()
    if not words:
        return ['']
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        test = f"{current} {word}"
        if pdfmetrics is None:
            approx_width = len(test) * font_size * 0.5
            fits = approx_width <= max_width
        else:
            fits = pdfmetrics.stringWidth(test, font_name, font_size) <= max_width
        if fits:
            current = test
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_wrapped(
    pdf: pdf_canvas.Canvas,
    text: str,
    x: float,
    y: float,
    max_width: float,
    line_height: float,
    font_name: str,
    font_size: int,
) -> float:
    pdf.setFont(font_name, font_size)
    for line in _wrap_text(text, max_width, font_name, font_size):
        pdf.drawString(x, y, line)
        y -= line_height
    return y


def _format_reason(explanations: Any, features: dict[str, Any] | None) -> str:
    reasons = []
    if isinstance(explanations, list) and explanations:
        top = []
        for item in explanations:
            feature = str(item.get('feature') or '')
            value = item.get('value')
            contrib = item.get('contribution')
            if feature:
                top.append(f"{feature}={value} ({contrib}%)")
        if top:
            reasons.append("Poids du modèle: " + ", ".join(top))
    if features:
        focus_keys = ['MA20', 'vol_regime', 'DCOILWTICO', 'CPIAUCSL', 'VolumeZ']
        parts = []
        for key in focus_keys:
            if key in features:
                try:
                    parts.append(f"{key}={float(features.get(key, 0.0)):.4f}")
                except Exception:
                    parts.append(f"{key}={features.get(key)}")
        if parts:
            reasons.append("Indicateurs: " + ", ".join(parts))
    return " | ".join(reasons) if reasons else "Aucune raison détaillée."


def _render_trade_journal_pdf(
    sandbox: str,
    as_of_date: datetime.date,
    trades: list[PaperTrade],
    closed_trades: list[PaperTrade],
    non_trades: list[AlertEvent],
) -> str:
    if pdf_canvas is None:
        raise RuntimeError('reportlab is required to render trading journal PDFs')
    output_dir = _journal_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    filename = f"journal_{sandbox.lower()}_{as_of_date.isoformat()}.pdf"
    path = os.path.join(output_dir, filename)
    pdf = pdf_canvas.Canvas(path, pagesize=letter)
    width, height = letter
    margin_x = 0.8 * inch
    y = height - 0.8 * inch

    pdf.setFont('Helvetica-Bold', 16)
    pdf.drawString(margin_x, y, f"Journal de Trading - {sandbox}")
    y -= 18
    pdf.setFont('Helvetica', 11)
    pdf.drawString(margin_x, y, f"Date: {as_of_date.isoformat()}")
    y -= 24

    pdf.setFont('Helvetica-Bold', 12)
    pdf.drawString(margin_x, y, "Nouveaux trades")
    y -= 14
    if not trades:
        pdf.setFont('Helvetica', 10)
        pdf.drawString(margin_x, y, "Aucun trade ouvert aujourd'hui.")
        y -= 18
    else:
        for trade in trades:
            header = (
                f"{trade.ticker} | Entrée {float(trade.entry_price):.2f} | "
                f"Qté {int(trade.quantity)} | Signal {float(trade.entry_signal or 0):.2f}"
            )
            y = _draw_wrapped(pdf, header, margin_x, y, width - 2 * margin_x, 14, 'Helvetica-Bold', 10)
            reasons = _format_reason(trade.entry_explanations, trade.entry_features or {})
            y = _draw_wrapped(pdf, reasons, margin_x, y, width - 2 * margin_x, 12, 'Helvetica', 9)
            y -= 6
            if y < 1.2 * inch:
                pdf.showPage()
                y = height - 0.8 * inch

    pdf.setFont('Helvetica-Bold', 12)
    pdf.drawString(margin_x, y, "Trades fermés")
    y -= 14
    if not closed_trades:
        pdf.setFont('Helvetica', 10)
        pdf.drawString(margin_x, y, "Aucun trade fermé aujourd'hui.")
        y -= 18
    else:
        for trade in closed_trades:
            pnl = float(trade.pnl or 0)
            header = (
                f"{trade.ticker} | Sortie {float(trade.exit_price or 0):.2f} | "
                f"P&L {pnl:.2f} | Résultat {trade.outcome or 'N/A'}"
            )
            y = _draw_wrapped(pdf, header, margin_x, y, width - 2 * margin_x, 14, 'Helvetica-Bold', 10)
            reasons = _format_reason(trade.entry_explanations, trade.entry_features or {})
            y = _draw_wrapped(pdf, reasons, margin_x, y, width - 2 * margin_x, 12, 'Helvetica', 9)
            y -= 6
            if y < 1.2 * inch:
                pdf.showPage()
                y = height - 0.8 * inch

    pdf.setFont('Helvetica-Bold', 12)
    pdf.drawString(margin_x, y, "Non-Trades (volume insuffisant)")
    y -= 14
    if not non_trades:
        pdf.setFont('Helvetica', 10)
        pdf.drawString(margin_x, y, "Aucun non-trade aujourd'hui.")
        y -= 18
    else:
        for event in non_trades:
            y = _draw_wrapped(pdf, event.message, margin_x, y, width - 2 * margin_x, 12, 'Helvetica', 9)
            y -= 4
            if y < 1.2 * inch:
                pdf.showPage()
                y = height - 0.8 * inch

    pdf.save()
    return path


@shared_task
def generate_trading_journal_daily(as_of: str | None = None) -> dict[str, Any]:
    """Generate daily trading journal PDFs for watchlist, bluechip, and penny sandboxes."""
    log = _task_log_start('generate_trading_journal_daily')
    if as_of:
        try:
            as_of_date = datetime.strptime(as_of, '%Y-%m-%d').date()
        except Exception:
            as_of_date = timezone.localdate()
    else:
        as_of_date = timezone.localdate()

    results = []
    for sandbox in ['WATCHLIST', 'AI_BLUECHIP', 'AI_PENNY']:
        trades = list(
            PaperTrade.objects.filter(sandbox=sandbox, entry_date__date=as_of_date).order_by('entry_date')
        )
        closed_trades = list(
            PaperTrade.objects.filter(sandbox=sandbox, exit_date__date=as_of_date).order_by('exit_date')
        )
        non_trades = list(
            AlertEvent.objects.filter(
                category='PAPER_NON_TRADE',
                created_at__date=as_of_date,
                message__icontains=f"[{sandbox}]",
            ).order_by('created_at')
        )
        path = _render_trade_journal_pdf(sandbox, as_of_date, trades, closed_trades, non_trades)
        results.append({
            'sandbox': sandbox,
            'file': path,
            'trades': len(trades),
            'closed_trades': len(closed_trades),
            'non_trades': len(non_trades),
        })

    payload = {'status': 'ok', 'as_of': as_of_date.isoformat(), 'results': results}
    _task_log_finish(log, 'SUCCESS', payload)
    return payload


@shared_task
def compute_model_evaluation_daily(as_of: str | None = None) -> dict[str, Any]:
    """Compute daily win rate, drawdown, and calibration per model version."""
    log = _task_log_start('compute_model_evaluation_daily')
    if as_of:
        try:
            as_of_date = datetime.fromisoformat(as_of).date()
        except ValueError:
            as_of_date = timezone.now().date()
    else:
        as_of_date = timezone.now().date()

    closed_today = PaperTrade.objects.filter(status='CLOSED', exit_date__date=as_of_date)
    if not closed_today.exists():
        payload = {'status': 'no_trades', 'as_of': str(as_of_date)}
        _task_log_finish(log, 'SUCCESS', payload)
        return payload

    groups = closed_today.values('model_name', 'model_version', 'sandbox').distinct()
    created = 0

    for group in groups:
        model_name = group.get('model_name') or 'BLUECHIP'
        model_version = group.get('model_version') or ''
        sandbox = group.get('sandbox') or ''

        day_trades = closed_today.filter(
            model_name=model_name,
            model_version=model_version,
            sandbox=sandbox,
        )
        trades_count = day_trades.count()
        if trades_count == 0:
            continue

        wins = day_trades.filter(outcome='WIN').count()
        total_pnl = float(sum([float(t.pnl or 0) for t in day_trades]))
        avg_pnl = total_pnl / trades_count if trades_count else 0.0
        win_rate = wins / trades_count if trades_count else 0.0

        signals = []
        outcomes = []
        for trade in day_trades:
            if trade.entry_signal is None:
                continue
            signals.append(float(trade.entry_signal))
            outcomes.append(1.0 if (trade.outcome == 'WIN' or float(trade.pnl or 0) > 0) else 0.0)

        if signals:
            brier = sum([(signals[i] - outcomes[i]) ** 2 for i in range(len(signals))]) / len(signals)
            mean_pred = sum(signals) / len(signals)
            mean_out = sum(outcomes) / len(outcomes)
        else:
            brier = None
            mean_pred = None
            mean_out = None

        history_trades = list(
            PaperTrade.objects.filter(
                status='CLOSED',
                model_name=model_name,
                model_version=model_version,
                sandbox=sandbox,
                exit_date__date__lte=as_of_date,
            ).order_by('exit_date')
        )
        max_drawdown = _calculate_max_drawdown(history_trades)

        ModelEvaluationDaily.objects.update_or_create(
            as_of=as_of_date,
            model_name=model_name,
            model_version=model_version,
            sandbox=sandbox,
            defaults={
                'trades': trades_count,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_pnl': total_pnl,
                'max_drawdown': max_drawdown,
                'brier_score': brier,
                'mean_predicted': mean_pred,
                'mean_outcome': mean_out,
            },
        )
        created += 1

    payload = {'status': 'ok', 'as_of': str(as_of_date), 'models': created}
    _task_log_finish(log, 'SUCCESS', payload)
    return payload


@shared_task
def compute_data_qa_daily(as_of: str | None = None) -> dict[str, Any]:
    log = _task_log_start('compute_data_qa_daily')
    if as_of:
        try:
            as_of_date = datetime.fromisoformat(as_of).date()
        except ValueError:
            as_of_date = timezone.now().date()
    else:
        as_of_date = timezone.now().date()

    cutoff = timezone.now() - timedelta(days=2)
    price_total = Stock.objects.count()
    price_missing = Stock.objects.filter(latest_price__isnull=True).count()
    price_stale = Stock.objects.filter(
        models.Q(latest_price_updated_at__lt=cutoff) | models.Q(latest_price_updated_at__isnull=True)
    ).count()
    price_bad = Stock.objects.filter(latest_price__lte=0).count()

    last_macro = MacroIndicator.objects.order_by('-date').first()
    macro_days_stale = None
    if last_macro:
        macro_days_stale = (as_of_date - last_macro.date).days

    news_cutoff = timezone.now() - timedelta(hours=24)
    news_count = StockNews.objects.filter(fetched_at__gte=news_cutoff).count()
    news_symbols = StockNews.objects.filter(fetched_at__gte=news_cutoff).values('stock_id').distinct().count()

    anomaly_count = 0
    for stock_id in PriceHistory.objects.values_list('stock_id', flat=True).distinct():
        prices = list(
            PriceHistory.objects.filter(stock_id=stock_id).order_by('-date')[:2].values_list('close_price', flat=True)
        )
        if len(prices) == 2:
            prev, curr = float(prices[1] or 0), float(prices[0] or 0)
            if prev > 0:
                move = abs((curr - prev) / prev)
                if move >= 0.3:
                    anomaly_count += 1

    metrics = {
        'price_metrics': {
            'total': price_total,
            'missing': price_missing,
            'stale': price_stale,
            'non_positive': price_bad,
        },
        'macro_metrics': {
            'last_date': str(last_macro.date) if last_macro else None,
            'days_stale': macro_days_stale,
        },
        'news_metrics': {
            'count_24h': news_count,
            'unique_symbols_24h': news_symbols,
        },
        'anomaly_metrics': {
            'large_moves_30pct': anomaly_count,
        },
    }

    DataQADaily.objects.update_or_create(
        as_of=as_of_date,
        defaults=metrics,
    )

    _task_log_finish(log, 'SUCCESS', {'as_of': str(as_of_date), **metrics})
    return {'as_of': str(as_of_date), **metrics}


def _psi(expected: list[float], actual: list[float], bins: int = 10) -> float:
    if not expected or not actual:
        return 0.0
    try:
        import numpy as np
        expected_arr = np.array(expected, dtype=float)
        actual_arr = np.array(actual, dtype=float)
        quantiles = np.linspace(0, 1, bins + 1)
        breaks = np.quantile(expected_arr, quantiles)
        breaks[0] = -float('inf')
        breaks[-1] = float('inf')
        psi = 0.0
        for i in range(len(breaks) - 1):
            exp_pct = ((expected_arr > breaks[i]) & (expected_arr <= breaks[i + 1])).mean()
            act_pct = ((actual_arr > breaks[i]) & (actual_arr <= breaks[i + 1])).mean()
            exp_pct = max(exp_pct, 1e-6)
            act_pct = max(act_pct, 1e-6)
            psi += (act_pct - exp_pct) * float(np.log(act_pct / exp_pct))
        return float(psi)
    except Exception:
        return 0.0


@shared_task
def compute_continuous_evaluation_daily(as_of: str | None = None) -> dict[str, Any]:
    log = _task_log_start('compute_continuous_evaluation_daily')
    if as_of:
        try:
            as_of_date = datetime.fromisoformat(as_of).date()
        except ValueError:
            as_of_date = timezone.now().date()
    else:
        as_of_date = timezone.now().date()

    window_days = int(os.getenv('EVAL_WINDOW_DAYS', '7'))
    baseline_days = int(os.getenv('DRIFT_BASELINE_DAYS', '30'))
    window_start = timezone.now().date() - timedelta(days=window_days)
    baseline_start = timezone.now().date() - timedelta(days=baseline_days)

    results = {'calibration': 0, 'drift': 0}
    for model_name in ['BLUECHIP', 'PENNY']:
        for sandbox in ['WATCHLIST', 'AI_BLUECHIP', 'AI_PENNY']:
            trades = PaperTrade.objects.filter(
                status='CLOSED',
                sandbox=sandbox,
                model_name=model_name,
                entry_date__date__gte=window_start,
            )
            if not trades.exists():
                continue
            model_version = trades.exclude(model_version='').values_list('model_version', flat=True).first() or ''

            signals = []
            outcomes = []
            for trade in trades:
                if trade.entry_signal is None:
                    continue
                signals.append(float(trade.entry_signal))
                outcomes.append(1.0 if (trade.outcome == 'WIN' or float(trade.pnl or 0) > 0) else 0.0)
            bins = []
            if signals:
                import numpy as np
                edges = np.linspace(0, 1, 11)
                for i in range(len(edges) - 1):
                    idx = [j for j, s in enumerate(signals) if edges[i] <= s < edges[i + 1]]
                    if not idx:
                        continue
                    avg_pred = float(np.mean([signals[j] for j in idx]))
                    avg_out = float(np.mean([outcomes[j] for j in idx]))
                    bins.append({'min': float(edges[i]), 'max': float(edges[i + 1]), 'avg_pred': avg_pred, 'avg_out': avg_out, 'count': len(idx)})
                brier = float(np.mean([(signals[i] - outcomes[i]) ** 2 for i in range(len(signals))]))
            else:
                brier = None

            ModelCalibrationDaily.objects.update_or_create(
                as_of=as_of_date,
                model_name=model_name,
                model_version=model_version,
                sandbox=sandbox,
                defaults={'bins': bins, 'count': len(signals), 'brier_score': brier},
            )
            results['calibration'] += 1

            baseline = PaperTrade.objects.filter(
                status='CLOSED',
                sandbox=sandbox,
                model_name=model_name,
                entry_date__date__gte=baseline_start,
                entry_date__date__lt=window_start,
            )
            if not baseline.exists():
                continue
            features = ['Volatility', 'Momentum20', 'RSI14']
            psi_metrics = {}
            feature_stats = {}
            for feat in features:
                expected = [float((t.entry_features or {}).get(feat, 0.0)) for t in baseline]
                actual = [float((t.entry_features or {}).get(feat, 0.0)) for t in trades]
                psi_metrics[feat] = _psi(expected, actual)
                feature_stats[feat] = {
                    'baseline_mean': float(sum(expected) / len(expected)) if expected else 0.0,
                    'current_mean': float(sum(actual) / len(actual)) if actual else 0.0,
                }

            ModelDriftDaily.objects.update_or_create(
                as_of=as_of_date,
                model_name=model_name,
                model_version=model_version,
                sandbox=sandbox,
                defaults={'psi': psi_metrics, 'feature_stats': feature_stats},
            )
            results['drift'] += 1

    _task_log_finish(log, 'SUCCESS', results)
    return results


@shared_task
def auto_rollback_models_daily() -> dict[str, Any]:
    log = _task_log_start('auto_rollback_models_daily')
    results: dict[str, Any] = {'rolled_back': []}
    rolled_models: set[str] = set()
    for sandbox in ['WATCHLIST', 'AI_BLUECHIP', 'AI_PENNY']:
        model_name = 'PENNY' if sandbox == 'AI_PENNY' else 'BLUECHIP'
        if model_name in rolled_models:
            continue
        min_win_rate = _sandbox_env_float(sandbox, 'ROLLBACK_MIN_WIN_RATE', '0.45')
        max_brier = _sandbox_env_float(sandbox, 'ROLLBACK_MAX_BRIER', '0.35')
        lookback_days = _sandbox_env_int(sandbox, 'ROLLBACK_LOOKBACK_DAYS', '3')
        cutoff = timezone.now().date() - timedelta(days=lookback_days)

        active = ModelRegistry.objects.filter(model_name=model_name, status='ACTIVE').order_by('-trained_at').first()
        if not active:
            continue
        evals = ModelEvaluationDaily.objects.filter(
            model_name=model_name,
            model_version=active.model_version,
            sandbox=sandbox,
            as_of__gte=cutoff,
        )
        if not evals.exists():
            continue
        avg_win = float(evals.aggregate(avg=models.Avg('win_rate')).get('avg') or 0)
        avg_brier = evals.aggregate(avg=models.Avg('brier_score')).get('avg')
        avg_brier = float(avg_brier) if avg_brier is not None else None

        degraded = avg_win < min_win_rate or (avg_brier is not None and avg_brier > max_brier)
        if not degraded:
            continue

        fallback = ModelRegistry.objects.filter(model_name=model_name, status='ARCHIVED').order_by('-trained_at').first()
        if not fallback:
            continue

        active.status = 'ARCHIVED'
        active.save(update_fields=['status'])
        fallback.status = 'ACTIVE'
        fallback.save(update_fields=['status'])
        rolled_models.add(model_name)
        results['rolled_back'].append({
            'model': model_name,
            'sandbox': sandbox,
            'from': active.model_version,
            'to': fallback.model_version,
            'avg_win_rate': avg_win,
            'avg_brier': avg_brier,
        })
        _send_alert(
            f"Model rollback: {model_name}",
            f"Sandbox {sandbox} | {active.model_version} -> {fallback.model_version} | win_rate={avg_win:.2f}, brier={avg_brier}",
        )

    _task_log_finish(log, 'SUCCESS', results)
    return results


@shared_task
def auto_retrain_on_drift_daily() -> dict[str, Any]:
    log = _task_log_start('auto_retrain_on_drift_daily')
    lookback_days = int(os.getenv('DRIFT_LOOKBACK_DAYS', '1'))
    cutoff = timezone.now().date() - timedelta(days=lookback_days)
    results: dict[str, Any] = {'retrained': []}

    for sandbox in ['WATCHLIST', 'AI_BLUECHIP', 'AI_PENNY']:
        threshold = _sandbox_env_float(sandbox, 'DRIFT_PSI_THRESHOLD', '0.2')
        entry = ModelDriftDaily.objects.filter(sandbox=sandbox, as_of__gte=cutoff).order_by('-as_of').first()
        if not entry or not entry.psi:
            continue
        max_psi = max([float(v) for v in entry.psi.values()]) if entry.psi else 0.0
        if max_psi < threshold:
            continue
        result = retrain_from_paper_trades_daily(sandbox_override=sandbox)
        results['retrained'].append({
            'sandbox': sandbox,
            'psi': max_psi,
            'threshold': threshold,
            'result': result,
        })

    _task_log_finish(log, 'SUCCESS', results)
    return results


@shared_task
def refresh_ai_bluechip_watchlist() -> dict[str, Any]:
    """Populate AI bluechip sandbox watchlist from AI opportunities endpoint."""
    if os.getenv('AI_BLUECHIP_AUTOFILL', 'true').lower() == 'false':
        return {'status': 'disabled', 'sandbox': 'AI_BLUECHIP'}
    base_url = os.getenv('BACKEND_INTERNAL_URL', 'http://backend:8000').rstrip('/')
    limit = int(os.getenv('AI_BLUECHIP_WATCHLIST_SIZE', '25'))
    min_score = float(os.getenv('AI_BLUECHIP_MIN_SCORE', '0.25'))
    try:
        resp = requests.get(
            f"{base_url}/api/ai/opportunities/",
            params={
                'limit': limit,
                'min_score': min_score,
                'include_universe': 'false',
            },
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json() or []
        if isinstance(payload, dict):
            payload = payload.get('results') or payload.get('data') or []
        symbols = [
            str(item.get('ticker') or '').strip().upper()
            for item in payload
            if isinstance(item, dict)
        ]
        symbols = [s for s in symbols if s]
        SandboxWatchlist.objects.update_or_create(
            sandbox='AI_BLUECHIP',
            defaults={'symbols': symbols, 'source': 'ai/opportunities'},
        )
        return {'status': 'ok', 'sandbox': 'AI_BLUECHIP', 'count': len(symbols)}
    except Exception as exc:
        return {'status': 'error', 'sandbox': 'AI_BLUECHIP', 'error': str(exc)}


@shared_task
def refresh_ai_penny_watchlist() -> dict[str, Any]:
    """Populate AI penny sandbox watchlist from penny analytics endpoint."""
    if os.getenv('AI_PENNY_AUTOFILL', 'true').lower() == 'false':
        return {'status': 'disabled', 'sandbox': 'AI_PENNY'}
    base_url = os.getenv('BACKEND_INTERNAL_URL', 'http://backend:8000').rstrip('/')
    limit = int(os.getenv('AI_PENNY_WATCHLIST_SIZE', '50'))
    min_score = float(os.getenv('AI_PENNY_MIN_SCORE', '0'))
    max_price = float(os.getenv('AI_PENNY_MAX_PRICE', '1.0'))
    min_volume = float(os.getenv('AI_PENNY_MIN_VOLUME', '100000'))
    try:
        resp = requests.get(
            f"{base_url}/api/penny-analytics/",
            params={
                'limit': limit,
                'min_score': min_score,
                'max_price': max_price,
                'min_volume': min_volume,
            },
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json() or []
        if isinstance(payload, dict):
            payload = payload.get('results') or payload.get('data') or []
        symbols = [
            str(item.get('stock_symbol') or item.get('symbol') or '').strip().upper()
            for item in payload
            if isinstance(item, dict)
        ]
        symbols = [s for s in symbols if s]
        SandboxWatchlist.objects.update_or_create(
            sandbox='AI_PENNY',
            defaults={'symbols': symbols, 'source': 'penny-analytics'},
        )
        return {'status': 'ok', 'sandbox': 'AI_PENNY', 'count': len(symbols)}
    except Exception as exc:
        return {'status': 'error', 'sandbox': 'AI_PENNY', 'error': str(exc)}


def _google_news_titles(ticker: str, days: int = 7, limit: int = 8) -> list[str]:
    if not ticker:
        return []
    now = datetime.utcnow()
    start = now - timedelta(days=days)
    rss_url = (
        f"https://news.google.com/rss/search?q={ticker}+stock+when:{days}d"
        "&hl=en-CA&gl=CA&ceid=CA:en"
    )
    feed = feedparser.parse(rss_url)
    titles: list[str] = []
    for entry in feed.entries or []:
        published = getattr(entry, 'published_parsed', None)
        if not published:
            continue
        published_dt = datetime(*published[:6])
        if published_dt < start or published_dt > now:
            continue
        title = getattr(entry, 'title', '')
        if title:
            titles.append(title)
        if len(titles) >= limit:
            break
    return titles


def _news_sentiment_score(ticker: str, days: int = 7) -> tuple[float, list[str]]:
    titles = _google_news_titles(ticker, days=days)
    if not titles:
        return 0.0, []
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(title).get('compound', 0.0) for title in titles]
    if not scores:
        return 0.0, titles
    return float(sum(scores) / len(scores)), titles


class DecisionEngine:
    def __init__(self) -> None:
        base_url = (
            os.getenv('DANAS_CHAT_BASE_URL')
            or os.getenv('OLLAMA_CHAT_BASE_URL')
            or os.getenv('OLLAMA_BASE_URL')
            or ''
        ).strip().rstrip('/')
        if base_url and '/v1' not in base_url:
            base_url = f"{base_url}/v1"
        self.base_url = base_url
        self.model = os.getenv('OLLAMA_MODEL', 'deepseek-r1')
        self.timeout = int(os.getenv('OLLAMA_TIMEOUT', '90'))
        self.lag_seconds = float(os.getenv('BROKER_LAG_SECONDS', '-0.58'))

    def _build_prompt(
        self,
        symbol: str,
        ml_score: float,
        rsi: float | None,
        rvol: float | None,
        titles: list[str],
        prophet_price: float | None,
        prophet_rec: str | None,
        news_sentiment: float | None,
        volatility: float | None,
    ) -> str:
        lines = [
            "VALIDATION DE SIGNAL - COMITÉ D'INVESTISSEMENT",
            "----------------------------------------------",
            f"TICKER: {symbol}",
            f"ML_SCORE: {ml_score:.2f}/100",
            f"RSI14: {rsi if rsi is not None else 'N/A'}",
            f"RVOL: {rvol if rvol is not None else 'N/A'}",
            f"PROPHET_7D: {prophet_price if prophet_price is not None else 'N/A'} ({prophet_rec or 'N/A'})",
            f"NEWS_SENTIMENT: {news_sentiment if news_sentiment is not None else 'N/A'}",
            f"VOLATILITÉ_INTRA: {volatility if volatility is not None else 'N/A'}",
            f"BROKER_LAG: {self.lag_seconds}s",
            "NEWS:",
            *([f"- {t}" for t in titles] if titles else ["- Aucune news disponible"]),
            "",
            "MISSION:",
            "1) Détermine si le dip est technique (opportunité) ou fondamental (danger).",
            "2) Donne un verdict final: BUY, WAIT, ou AVOID.",
            "3) Donne un score de validation sur 10.",
            "",
            "RÉPONSE COURTE:",
            "- DIAGNOSTIC: ...",
            "- VERDICT: BUY/WAIT/AVOID",
            "- VALIDATION: X/10",
        ]
        return "\n".join(lines)

    def evaluate(
        self,
        symbol: str,
        ml_score: float,
        rsi: float | None,
        rvol: float | None,
        titles: list[str],
        prophet_price: float | None = None,
        prophet_rec: str | None = None,
        news_sentiment: float | None = None,
        volatility: float | None = None,
    ) -> dict[str, Any]:
        if not self.base_url:
            return {'verdict': 'WAIT', 'score': 6.0, 'raw': ''}
        prompt = self._build_prompt(
            symbol,
            ml_score,
            rsi,
            rvol,
            titles,
            prophet_price,
            prophet_rec,
            news_sentiment,
            volatility,
        )
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': (
                        f"Tu es Danas. Nous sommes le {timezone.now().strftime('%d/%m/%Y %H:%M')}. "
                        "Réponds uniquement en français."
                    ),
                },
                {'role': 'user', 'content': prompt},
            ],
            'stream': False,
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            text = ((data.get('choices') or [{}])[0].get('message') or {}).get('content') or ''
        except Exception:
            text = ''

        verdict = 'WAIT'
        if re.search(r'\bBUY\b', text, re.IGNORECASE):
            verdict = 'BUY'
        elif re.search(r'\bAVOID\b', text, re.IGNORECASE):
            verdict = 'AVOID'

        score = None
        match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', text)
        if match:
            try:
                score = float(match.group(1))
            except Exception:
                score = None
        if score is None:
            score = 9.0 if verdict == 'BUY' else 6.0 if verdict == 'WAIT' else 2.5
        return {'verdict': verdict, 'score': score, 'raw': text}


@lru_cache(maxsize=1)
def _finbert_pipeline():
    try:
        from transformers import pipeline
        return pipeline(
            'sentiment-analysis',
            model='ProsusAI/finbert',
            tokenizer='ProsusAI/finbert',
            truncation=True,
        )
    except Exception:
        return None


def _finbert_score_from_titles(titles: list[str]) -> float:
    if not titles:
        return 0.0
    pipe = _finbert_pipeline()
    if pipe is None:
        return 0.0
    try:
        results = pipe(titles)
        scores = []
        for item in results:
            label = str(item.get('label') or '').upper()
            score = float(item.get('score') or 0)
            if label == 'POSITIVE':
                scores.append(score)
            elif label == 'NEGATIVE':
                scores.append(-score)
            else:
                scores.append(0.0)
        return float(sum(scores) / len(scores)) if scores else 0.0
    except Exception:
        return 0.0


def _news_sentiment_at_date(symbol: str, as_of: datetime) -> float:
    symbol = (symbol or '').strip().upper()
    if not symbol:
        return 0.0
    stock = Stock.objects.filter(symbol__iexact=symbol).first()
    if not stock:
        return 0.0
    start = as_of - timedelta(days=1)
    end = as_of + timedelta(days=1)
    headlines = list(
        StockNews.objects.filter(stock=stock, published_at__gte=start, published_at__lte=end)
        .order_by('-published_at')
        .values_list('headline', flat=True)[:5]
    )
    return _finbert_score_from_titles([h for h in headlines if h])


def _yahoo_fundamentals(symbol: str) -> dict[str, float]:
    symbol = (symbol or '').strip().upper()
    if not symbol:
        return {}
    cache_key = f"fundamentals:yahoo:{symbol}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        payload = {
            'quick_ratio': float(info.get('quickRatio') or 0),
            'current_ratio': float(info.get('currentRatio') or 0),
            'revenue_growth': float(info.get('revenueGrowth') or 0),
            'profit_margins': float(info.get('profitMargins') or 0),
            'debt_to_equity': float(info.get('debtToEquity') or 0),
            'price_to_book': float(info.get('priceToBook') or 0),
            'trailing_pe': float(info.get('trailingPE') or 0),
        }
        cache.set(cache_key, payload, timeout=60 * 60 * 24)
        return payload
    except Exception:
        return {}


def _fetch_sp500_symbols(limit: int = 50) -> list[str]:
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        frame = tables[0]
        symbols = [str(s).strip().upper().replace('.', '-') for s in frame['Symbol'].tolist()]
        return symbols[:limit]
    except Exception:
        fallback = [s.strip().upper() for s in os.getenv('BLUECHIP_SYMBOLS', '').split(',') if s.strip()]
        return fallback[:limit]


def _analyze_penny_breakouts() -> dict[str, Any]:
    quotes = _fetch_yahoo_screener('most_actives', count=400)
    candidates: list[dict[str, Any]] = []
    for quote in quotes:
        symbol = (quote.get('symbol') or '').strip().upper()
        price = _safe_float(quote.get('regularMarketPrice'))
        volume = _safe_float(quote.get('regularMarketVolume') or quote.get('averageDailyVolume3Month'))
        if not symbol or price is None or volume is None:
            continue
        if price <= 0 or price > 5:
            continue
        if volume < 500000:
            continue
        candidates.append({'symbol': symbol, 'price': price, 'volume': volume})
    candidates = sorted(candidates, key=lambda x: x['volume'], reverse=True)[:200]
    symbols = [c['symbol'] for c in candidates]
    if not symbols:
        return {'watchlist': [], 'count': 0}

    data = yf.download(
        tickers=" ".join(symbols),
        period='1y',
        interval='1d',
        group_by='ticker',
        threads=True,
        auto_adjust=False,
    )

    watchlist: list[dict[str, Any]] = []
    for item in candidates:
        symbol = item['symbol']
        if isinstance(data.columns, pd.MultiIndex):
            if symbol not in data:
                continue
            frame = data[symbol].dropna()
        else:
            frame = data.copy().dropna()
        if frame is None or frame.empty or 'Close' not in frame or 'Volume' not in frame:
            continue
        close = frame['Close'].tail(30)
        volume = frame['Volume'].tail(30)
        if len(close) < 30 or len(volume) < 30:
            continue
        mean_price = float(close.mean()) if close.mean() else 0.0
        if mean_price <= 0:
            continue
        price_range_pct = (float(close.max()) - float(close.min())) / mean_price
        last5 = float(volume.tail(5).mean()) if len(volume) >= 5 else float(volume.mean())
        prev20 = float(volume.head(25).tail(20).mean()) if len(volume) >= 25 else float(volume.mean())
        vol_ratio = (last5 / prev20) if prev20 else 0.0
        if price_range_pct > 0.08 or vol_ratio < 1.2:
            continue
        sentiment, titles = _news_sentiment_score(symbol, days=7)
        catalyst_keywords = ['contract', 'fda', 'patent', 'approval', 'merger', 'acquisition']
        catalyst_hits = [t for t in titles if any(k in t.lower() for k in catalyst_keywords)]
        score = (max(0.0, 1 - price_range_pct) * 0.6) + (min(vol_ratio / 3, 1) * 0.4)
        score = score * (1 + (sentiment * 0.3))
        watchlist.append({
            'symbol': symbol,
            'last_close': float(close.iloc[-1]),
            'price_range_pct': round(price_range_pct * 100, 2),
            'volume_ratio': round(vol_ratio, 2),
            'sentiment': round(sentiment, 3),
            'score': round(score, 3),
            'catalysts': catalyst_hits[:3],
        })

    watchlist = sorted(watchlist, key=lambda x: x['score'], reverse=True)[:30]
    return {'watchlist': watchlist, 'count': len(watchlist)}


@shared_task
def deep_learning_retro_train() -> dict[str, Any]:
    log = _task_log_start('deep_learning_retro_train')
    try:
        import numpy as np
        import pandas_ta as ta
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_squared_error
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        import joblib

        watch = SandboxWatchlist.objects.filter(sandbox='WATCHLIST').first()
        symbols = [str(s).strip().upper() for s in (watch.symbols if watch else []) if str(s).strip()]
        if not symbols:
            fallback = os.getenv('WATCHLIST_SYMBOLS', os.getenv('WATCHLIST', '')).strip()
            if fallback:
                symbols = [s.strip().upper() for s in fallback.split(',') if s.strip()]

        limit = int(os.getenv('RETRO_TRAIN_SYMBOL_LIMIT', '50'))
        symbols = symbols[:limit]
        if not symbols:
            payload = {'status': 'empty'}
            _task_log_finish(log, 'SUCCESS', payload)
            return payload

        lookahead_days = int(os.getenv('RETRO_TRAIN_LOOKAHEAD_DAYS', '5'))
        target_pct = float(os.getenv('RETRO_TRAIN_TARGET_PCT', '0.05'))

        feature_rows: list[dict[str, Any]] = []
        target_rows: list[float] = []
        sample_dates: list[pd.Timestamp] = []

        for symbol in symbols:
            daily = get_daily_bars(symbol, days=365 * 5)
            if daily is None or daily.empty:
                daily = yf.download(symbol, period='5y', interval='1d')
                if daily is None or daily.empty:
                    continue
                if not isinstance(daily.columns, pd.MultiIndex):
                    close_col = 'Adj Close' if 'Adj Close' in daily.columns else 'Close'
                    daily = daily.rename(columns={
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        close_col: 'close',
                        'Volume': 'volume',
                    })
                if isinstance(daily.index, pd.DatetimeIndex):
                    daily['timestamp'] = daily.index

            frame = daily.copy()
            if 'timestamp' in frame.columns:
                frame['timestamp'] = pd.to_datetime(frame['timestamp'], errors='coerce', utc=True)
                frame = frame.dropna(subset=['timestamp']).set_index('timestamp')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col not in frame.columns:
                    frame[col] = pd.NA
            frame = frame.dropna(subset=['open', 'high', 'low', 'close'])
            if frame.empty:
                continue

            frame = enrich_bars_with_patterns(frame)
            if frame.empty:
                continue

            frame = _add_candlestick_features(frame)
            if frame.empty:
                continue

            frame['pattern_success_3d'] = _pattern_success_3d(frame, target_pct=0.05)

            frame['rsi14'] = ta.rsi(frame['close'], length=14)
            frame['ema20'] = ta.ema(frame['close'], length=20)
            frame['ema50'] = ta.ema(frame['close'], length=50)
            frame['atr14'] = ta.atr(frame['high'], frame['low'], frame['close'], length=14)
            frame['momentum10'] = ta.mom(frame['close'], length=10)
            frame['day_of_week'] = pd.to_datetime(frame.index).dayofweek
            frame['hour_of_day'] = pd.to_datetime(frame.index).hour
            frame = frame.fillna(0.0)

            fundamentals = _yahoo_fundamentals(symbol)
            parent = CORRELATION_MAP.get(symbol) or os.getenv('PENNY_SNIPER_DEFAULT_PARENT', 'SPY')
            parent_change = _intraday_pct_change(parent, minutes=60) if parent else None
            parent_change = float(parent_change or 0.0)

            closes = frame['close'].values
            for idx in range(len(frame) - lookahead_days):
                row = frame.iloc[idx]
                if float(row.get('pattern_signal') or 0) <= 0:
                    continue
                if float(row.get('rvol') or 0) < 1.5:
                    continue
                base = float(closes[idx])
                future_max = float(np.max(closes[idx + 1: idx + 1 + lookahead_days]))
                target = 1.0 if future_max >= base * (1 + target_pct) else 0.0
                sentiment = _news_sentiment_at_date(symbol, row.name.to_pydatetime())

                feature_rows.append({
                    'pattern_signal': float(row.get('pattern_signal') or 0),
                    'rvol': float(row.get('rvol') or 0),
                    'pattern_doji': bool(row.get('pattern_doji')),
                    'pattern_hammer': bool(row.get('pattern_hammer')),
                    'pattern_engulfing': bool(row.get('pattern_engulfing')),
                    'pattern_morning_star': bool(row.get('pattern_morning_star')),
                    'pattern_success_3d': bool(row.get('pattern_success_3d')),
                    'rsi14': float(row.get('rsi14') or 0),
                    'ema20': float(row.get('ema20') or 0),
                    'ema50': float(row.get('ema50') or 0),
                    'volatility': float(row.get('volatility') or 0),
                    'atr14': float(row.get('atr14') or 0),
                    'momentum10': float(row.get('momentum10') or 0),
                    'day_of_week': int(row.get('day_of_week') or 0),
                    'hour_of_day': int(row.get('hour_of_day') or 0),
                    'news_sentiment': float(sentiment or 0),
                    'parent_change': parent_change,
                    **fundamentals,
                })
                target_rows.append(float(target))
                sample_dates.append(pd.to_datetime(row.name))

        if not feature_rows:
            payload = {'status': 'no_samples'}
            _task_log_finish(log, 'SUCCESS', payload)
            return payload

        dataset = pd.DataFrame(feature_rows).fillna(0.0)
        dataset['date'] = pd.to_datetime(sample_dates, errors='coerce')
        dataset['target'] = target_rows
        dataset = dataset.dropna(subset=['date']).sort_values('date')
        target = dataset['target'].values

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            (
                'model',
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=8,
                    random_state=42,
                ),
            ),
        ])

        splits = TimeSeriesSplit(n_splits=5)
        rmse_scores: list[float] = []
        last_test_idx = None
        for train_idx, test_idx in splits.split(dataset):
            pipeline.fit(dataset.iloc[train_idx].drop(columns=['date', 'target']), target[train_idx])
            preds = pipeline.predict(dataset.iloc[test_idx].drop(columns=['date', 'target']))
            rmse_scores.append(float(mean_squared_error(target[test_idx], preds, squared=False)))
            last_test_idx = test_idx

        def _walk_forward_rmse() -> list[dict[str, float | int | str]]:
            reports = []
            if dataset.empty:
                return reports
            start = dataset['date'].min()
            end = dataset['date'].max()
            if start is None or end is None:
                return reports
            test_start = start + pd.DateOffset(months=3)
            while test_start < end:
                test_end = test_start + pd.DateOffset(months=3)
                train_mask = dataset['date'] < test_start
                test_mask = (dataset['date'] >= test_start) & (dataset['date'] < test_end)
                if train_mask.sum() < 60 or test_mask.sum() == 0:
                    test_start = test_end
                    continue
                X_train = dataset.loc[train_mask].drop(columns=['date', 'target'])
                y_train = target[train_mask.values]
                X_test = dataset.loc[test_mask].drop(columns=['date', 'target'])
                y_test = target[test_mask.values]
                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)
                rmse = float(mean_squared_error(y_test, preds, squared=False)) if len(y_test) else 0.0
                reports.append({
                    'start': test_start.strftime('%Y-%m-%d'),
                    'end': test_end.strftime('%Y-%m-%d'),
                    'samples': int(len(y_test)),
                    'rmse': rmse,
                })
                test_start = test_end
            return reports

        walk_forward = _walk_forward_rmse()
        pipeline.fit(dataset.drop(columns=['date', 'target']), target)
        rmse = float(np.mean(rmse_scores)) if rmse_scores else 0.0

        model_path = Path(__file__).resolve().parent / 'ml_engine' / 'retro_pattern_success.pkl'
        joblib.dump({'model': pipeline, 'features': [c for c in dataset.columns if c not in {'date', 'target'}]}, model_path)

        payload = {
            'status': 'ok',
            'symbols': len(symbols),
            'samples': len(dataset),
            'rmse': round(rmse, 4),
            'walk_forward': walk_forward,
            'model_path': str(model_path),
        }
        _task_log_finish(log, 'SUCCESS', payload)
        return payload
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


def _error_category(label: str | None) -> str:
    text = (label or '').lower()
    if 'btc' in text or 'bitcoin' in text:
        return 'btc_drop'
    if 'dilution' in text:
        return 'dilution'
    if 'fausse' in text or 'breakout' in text:
        return 'false_breakout'
    return 'news_negative'


def _gemini_error_label(symbol: str, headlines: list[str], parent: str | None, parent_change: float | None) -> str | None:
    api_key = getattr(settings, 'GEMINI_AI_API_KEY', None)
    if not api_key or genai is None:
        return None
    try:
        client = genai.Client(api_key=api_key)
        headline_text = " | ".join(headlines[:5]) if headlines else 'n/a'
        prompt = (
            "Tu analyses un trade perdant de paper trading. Donne une étiquette courte d'erreur "
            "parmi: 'Fausse cassure', 'Baisse du Bitcoin', 'Dilution', 'News négatives', 'Macro défavorable'. "
            f"Ticker: {symbol}. Headlines: {headline_text}. Parent: {parent}, change: {parent_change}."
        )
        response = client.models.generate_content(model='gemini-1.5-flash', contents=prompt)
        label = (getattr(response, 'text', None) or '').strip()
        return label or None
    except Exception:
        return None


@shared_task
def backtesting_critique_task(sandbox_override: str | None = None) -> dict[str, Any]:
    log = _task_log_start('backtesting_critique_task')
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        lookback_days = int(os.getenv('ERROR_TRAIN_LOOKBACK_DAYS', '120'))
        cutoff = timezone.now() - timedelta(days=lookback_days)
        sandbox = (sandbox_override or os.getenv('ERROR_TRAIN_SANDBOX', 'ALL')).strip().upper()
        sandboxes = ['WATCHLIST', 'AI_BLUECHIP', 'AI_PENNY'] if sandbox == 'ALL' else [sandbox]

        results: list[dict[str, Any]] = []

        for sb in sandboxes:
            trades = (
                PaperTrade.objects.filter(status='CLOSED', sandbox=sb, entry_date__gte=cutoff)
                .exclude(entry_features__isnull=True)
                .order_by('entry_date')
            )
            if not trades.exists():
                results.append({'sandbox': sb, 'status': 'no_trades'})
                continue

            error_labels: dict[int, str] = {}
            for trade in trades.filter(outcome='LOSS')[:50]:
                symbol = (trade.ticker or '').strip().upper()
                if not symbol:
                    continue
                start = (trade.entry_date or timezone.now()) - timedelta(hours=12)
                end = (trade.entry_date or timezone.now()) + timedelta(hours=12)
                headlines = list(
                    StockNews.objects.filter(stock__symbol__iexact=symbol, published_at__gte=start, published_at__lte=end)
                    .order_by('-published_at')
                    .values_list('headline', flat=True)[:5]
                )
                if not headlines:
                    headlines = _google_news_titles(symbol, days=2, limit=5)
                parent = CORRELATION_MAP.get(symbol) or os.getenv('PENNY_SNIPER_DEFAULT_PARENT', 'SPY')
                parent_change = _intraday_pct_change(parent, minutes=60) if parent else None
                label = _gemini_error_label(symbol, headlines, parent, parent_change)
                if label is None:
                    if parent_change is not None and parent_change < -0.01:
                        label = f"Baisse du {parent}"
                    else:
                        label = 'Fausse cassure'
                error_labels[trade.id] = label
                if label and label not in (trade.notes or ''):
                    trade.notes = (trade.notes or '') + f" | Error label: {label}"
                    trade.save(update_fields=['notes'])

            feature_rows: list[dict[str, Any]] = []
            target_rows: list[int] = []
            sample_weights: list[float] = []

            for trade in trades:
                features = dict(trade.entry_features or {})
                symbol = (trade.ticker or '').strip().upper()
                features.update(_entry_time_features(symbol, now=trade.entry_date or timezone.now()))
                label = error_labels.get(trade.id)
                category = _error_category(label)
                features.update({
                    'error_flag': 1 if (trade.outcome == 'LOSS' and label) else 0,
                    'error_cat_btc': 1 if category == 'btc_drop' else 0,
                    'error_cat_dilution': 1 if category == 'dilution' else 0,
                    'error_cat_false_breakout': 1 if category == 'false_breakout' else 0,
                    'error_cat_news_negative': 1 if category == 'news_negative' else 0,
                })
                feature_rows.append(features)
                target_rows.append(1 if trade.outcome == 'WIN' else 0)
                weight = 2.5 if (trade.outcome == 'LOSS' and label) else 1.0
                sample_weights.append(weight)

            df = pd.DataFrame(feature_rows).fillna(0.0)
            if df.empty:
                results.append({'sandbox': sb, 'status': 'no_samples'})
                continue
            X = df.values
            y = np.array(target_rows)

            if len(set(y)) < 2:
                results.append({'sandbox': sb, 'status': 'insufficient_labels'})
                continue

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                (
                    'model',
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=6,
                        min_samples_split=6,
                        min_samples_leaf=6,
                        random_state=42,
                    ),
                ),
            ])

            cv_scores: list[float] = []
            if len(y) >= 20:
                splits = TimeSeriesSplit(n_splits=5)
                for train_idx, test_idx in splits.split(X):
                    pipeline.fit(X[train_idx], y[train_idx], sample_weight=np.array(sample_weights)[train_idx])
                    cv_scores.append(float(pipeline.score(X[test_idx], y[test_idx])))

            pipeline.fit(X, y, sample_weight=np.array(sample_weights))

            universe = 'PENNY' if sb == 'AI_PENNY' else 'BLUECHIP'
            model_path = get_model_path(universe)
            joblib.dump({'model': pipeline, 'features': list(df.columns), 'cv_scores': cv_scores}, model_path)

            results.append({
                'sandbox': sb,
                'status': 'ok',
                'samples': int(len(df)),
                'loss_labels': len(error_labels),
                'cv_scores': cv_scores,
                'model_path': str(model_path),
            })

        payload = {'status': 'ok', 'results': results}
        _task_log_finish(log, 'SUCCESS', payload)
        return payload
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


@shared_task
def analyze_ticker_for_ui(symbol: str) -> dict[str, Any]:
    """Analyse complète pour UI: technique, sentiment, fondamentaux, Gemini et mise suggérée."""
    try:
        symbol = (symbol or '').strip().upper()
        if not symbol:
            return {'error': 'Ticker requis'}

        candidates = [symbol]
        if '.' not in symbol:
            candidates.extend([f"{symbol}.TO", f"{symbol}.V"])
        else:
            candidates.append(symbol.split('.')[0])

        selected_symbol = None
        daily = None
        for candidate in candidates:
            daily = get_daily_bars(candidate, days=45)
            if daily is None or daily.empty:
                daily = yf.download(candidate, period='2mo', interval='1d')
            if daily is None or daily.empty:
                try:
                    daily = yfin.download(candidate, period='2mo', interval='1d')
                except Exception:
                    daily = None
            if daily is not None and not daily.empty:
                selected_symbol = candidate
                break

        if daily is None or daily.empty:
            return {'error': f"Données introuvables pour {symbol}"}
        symbol = selected_symbol or symbol
        if not isinstance(daily.columns, pd.MultiIndex):
            daily = daily.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
            })

        close_series = None
        if isinstance(daily.columns, pd.MultiIndex):
            level0 = daily.columns.get_level_values(0)
            if 'Close' in level0:
                close_series = daily['Close']
                if isinstance(close_series, pd.DataFrame):
                    close_series = close_series.iloc[:, 0] if not close_series.empty else None
            elif 'close' in level0:
                close_series = daily['close']
                if isinstance(close_series, pd.DataFrame):
                    close_series = close_series.iloc[:, 0] if not close_series.empty else None
            elif 'Close' in daily.columns.get_level_values(-1):
                try:
                    close_series = daily.xs('Close', axis=1, level=-1)
                    if isinstance(close_series, pd.DataFrame):
                        close_series = close_series.iloc[:, 0] if not close_series.empty else None
                except Exception:
                    close_series = None
        elif 'close' in daily:
            close_series = daily['close']
        elif 'Close' in daily:
            close_series = daily['Close']
        elif 'Adj Close' in daily:
            close_series = daily['Adj Close']
        else:
            close_series = _extract_close_series(daily)

        if close_series is None or close_series.empty:
            if isinstance(daily.columns, pd.MultiIndex):
                try:
                    flat = daily.copy()
                    flat.columns = [col[0] for col in flat.columns]
                    flat_close = flat.get('Close')
                    if flat_close is None:
                        flat_close = flat.get('close')
                    close_series = flat_close
                except Exception:
                    close_series = None
        if close_series is None or close_series.empty:
            return {'error': f"Données introuvables pour {symbol}"}
        last_price = float(close_series.iloc[-1])

        is_bluechip = last_price >= 5
        universe = 'BLUECHIP' if is_bluechip else 'PENNY'
        model_path = get_model_path(universe)
        fusion = DataFusionEngine(symbol)
        fusion_df = fusion.fuse_all()
        payload = load_or_train_model(fusion_df, model_path=model_path) if fusion_df is not None else None

        confidence_score = 0.5
        if payload and payload.get('model') and fusion_df is not None and not fusion_df.empty:
            last_row = fusion_df.tail(1).copy()
            feature_list = payload.get('features') or FEATURE_COLUMNS
            for col in feature_list:
                if col not in last_row.columns:
                    last_row[col] = 0.0
            features = last_row[feature_list].fillna(0).values
            try:
                confidence_score = float(payload['model'].predict_proba(features)[0][1])
            except Exception:
                confidence_score = 0.5

        news_sentiment, titles = _news_sentiment_score(symbol, days=2)
        sentiment_score = max(0.0, min(1.0, (float(news_sentiment) + 1.0) / 2.0))

        atr = None
        if isinstance(daily.columns, pd.MultiIndex):
            level0 = daily.columns.get_level_values(0)
            if {'High', 'Low', 'Close'}.issubset(set(level0)):
                high = daily['High']
                low = daily['Low']
                close = daily['Close']
                if isinstance(high, pd.DataFrame):
                    high = high.iloc[:, 0] if not high.empty else None
                if isinstance(low, pd.DataFrame):
                    low = low.iloc[:, 0] if not low.empty else None
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0] if not close.empty else None
            elif {'high', 'low', 'close'}.issubset(set(level0)):
                high = daily['high']
                low = daily['low']
                close = daily['close']
                if isinstance(high, pd.DataFrame):
                    high = high.iloc[:, 0] if not high.empty else None
                if isinstance(low, pd.DataFrame):
                    low = low.iloc[:, 0] if not low.empty else None
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0] if not close.empty else None
            elif {'High', 'Low', 'Close'}.issubset(set(daily.columns.get_level_values(-1))):
                try:
                    high = daily.xs('High', axis=1, level=-1)
                    low = daily.xs('Low', axis=1, level=-1)
                    close = daily.xs('Close', axis=1, level=-1)
                    if isinstance(high, pd.DataFrame):
                        high = high.iloc[:, 0] if not high.empty else None
                    if isinstance(low, pd.DataFrame):
                        low = low.iloc[:, 0] if not low.empty else None
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0] if not close.empty else None
                except Exception:
                    high = low = close = None
            else:
                high = low = close = None
        elif {'high', 'low', 'close'}.issubset(set(daily.columns)):
            high = daily['high']
            low = daily['low']
            close = daily['close']
        elif {'High', 'Low', 'Close'}.issubset(set(daily.columns)):
            high = daily['High']
            low = daily['Low']
            close = daily['Close']
        else:
            high = low = close = None
        if high is not None and low is not None and close is not None:
            tr = pd.concat([
                (high - low).abs(),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else None

        allocation = _risk_manager_allocation(confidence_score * 100, sentiment_score, last_price, atr)

        earnings_date = None
        days_to_earnings = None
        pe_ratio = None
        eps = None
        debt_to_equity = None
        try:
            ticker = yf.Ticker(symbol)
            earnings_date = _get_earnings_date(ticker)
            if earnings_date:
                days_to_earnings = (earnings_date - timezone.now().date()).days
            info = ticker.info or {}
            pe_ratio = float(info.get('trailingPE') or 0) if info.get('trailingPE') else None
            eps = float(info.get('trailingEps') or 0) if info.get('trailingEps') else None
            debt_to_equity = float(info.get('debtToEquity') or 0) if info.get('debtToEquity') else None
        except Exception:
            earnings_date = None

        gemini_summary = None
        if getattr(settings, 'GEMINI_AI_API_KEY', None) and genai is not None:
            try:
                client = genai.Client(api_key=settings.GEMINI_AI_API_KEY)
                prompt = (
                    f"Analyse {symbol} au prix {last_price:.2f}. "
                    f"Confidence ML: {confidence_score:.2f}, Sentiment News: {sentiment_score:.2f}. "
                    f"P/E: {pe_ratio}, EPS: {eps}, Debt/Equity: {debt_to_equity}. "
                    f"Jours avant earnings: {days_to_earnings}. "
                    "Explique en 3 lignes max la synthèse de l'analyste. "
                    "Si earnings dans <7 jours, ajoute ⚠️ PRUDENCE : Résultats imminents. "
                    "Termine par un verdict ACHETER, VENDRE ou ATTENDRE."
                )
                response = client.models.generate_content(model='gemini-1.5-flash', contents=prompt)
                gemini_summary = (getattr(response, 'text', None) or '').strip() or None
            except Exception:
                gemini_summary = None

        if not gemini_summary:
            trend = "HAUSSIÈRE" if confidence_score > 0.5 else "BAISSIÈRE"
            strength = "forte" if confidence_score > 0.8 else "modérée"
            suggested_amount = allocation
            gemini_summary = (
                f"Analyse Algorithmique : Le modèle détecte une tendance {trend} {strength}. "
                f"Score de confiance : {confidence_score * 100:.1f}%. "
                f"La mise a été ajustée à {suggested_amount}$ conformément aux paramètres de risque (Max Drawdown 8%)."
            )
            if days_to_earnings and days_to_earnings < 7:
                gemini_summary += " ⚠️ RISQUE ÉLEVÉ : Résultats financiers imminents."

        target_price = round(last_price * (1 + (confidence_score / 5)), 2)
        stop_loss = round(last_price * 0.95, 2)

        result = {
            'symbol': symbol,
            'price': round(last_price, 4),
            'confidence': round(confidence_score * 100, 2),
            'sentiment': 'Positif' if news_sentiment > 0.2 else 'Négatif' if news_sentiment < -0.2 else 'Neutre',
            'suggested_investment': allocation,
            'earnings_risk': 'HAUT' if days_to_earnings is not None and days_to_earnings < 7 else 'BAS',
            'summary': gemini_summary,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'news_titles': titles[:5],
            'universe': universe,
            'sandbox': 'AI_PENNY' if universe == 'PENNY' else 'AI_BLUECHIP',
        }
        cache.set(f"ai_analysis:{symbol}", result, timeout=60 * 10)
        return result
    except Exception as exc:
        return {'error': str(exc)}


def _analyze_bluechip_rebounds() -> dict[str, Any]:
    symbols = _fetch_sp500_symbols(limit=50)
    if not symbols:
        return {'selection': [], 'count': 0}
    tickers = symbols + ['SPY']
    data = yf.download(
        tickers=" ".join(tickers),
        period='1y',
        interval='1d',
        group_by='ticker',
        threads=True,
        auto_adjust=False,
    )
    spy = data['SPY'] if isinstance(data.columns, pd.MultiIndex) and 'SPY' in data else None
    if spy is None or spy.empty:
        return {'selection': [], 'count': 0}
    spy_close = spy['Close'] if 'Close' in spy else None
    if spy_close is None or spy_close.empty:
        return {'selection': [], 'count': 0}

    candidates: list[dict[str, Any]] = []
    for symbol in symbols:
        if symbol not in data:
            continue
        frame = data[symbol].dropna()
        if frame is None or frame.empty or 'Close' not in frame:
            continue
        close = frame['Close']
        if len(close) < 220:
            continue
        sma50 = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1]
        last_close = float(close.iloc[-1])
        if not sma50 or not sma200:
            continue
        rs = close / spy_close.reindex(close.index).fillna(method='ffill')
        rs_mean = rs.rolling(50).mean().iloc[-1]
        rs_score = 1 if rs.iloc[-1] >= rs_mean else 0
        rebound = last_close >= sma50 * 0.99 or last_close >= sma200 * 0.99
        if not rebound:
            continue
        sentiment, titles = _news_sentiment_score(symbol, days=14)
        score = (0.4 if last_close >= sma50 else 0) + (0.3 if last_close >= sma200 else 0) + (0.3 * rs_score)
        score = score * (1 + sentiment * 0.2)
        candidates.append({
            'symbol': symbol,
            'last_close': round(last_close, 2),
            'sma50': round(float(sma50), 2),
            'sma200': round(float(sma200), 2),
            'sentiment': round(sentiment, 3),
            'score': round(score, 3),
            'news': titles[:3],
        })

    candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:3]
    return {'selection': candidates, 'count': len(candidates)}


@shared_task
def economic_calendar_module() -> dict[str, Any]:
    log = _task_log_start('economic_calendar_module')
    url = os.getenv('ECONOMIC_CALENDAR_URL', 'https://nfs.faireconomy.media/ff_calendar_thisweek.json')
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        payload = resp.json() or []
        events: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            event_dt = _parse_economic_event_datetime(item)
            if not event_dt:
                continue
            ny_dt = event_dt.astimezone(ZoneInfo('America/New_York'))
            events.append({
                'title': item.get('title') or item.get('event') or '',
                'country': item.get('country') or '',
                'currency': item.get('currency') or '',
                'impact': item.get('impact') or '',
                'datetime_utc': event_dt.isoformat(),
                'datetime_ny': ny_dt.isoformat(),
                'forecast': item.get('forecast'),
                'actual': item.get('actual'),
                'previous': item.get('previous'),
            })
        cache.set(ECONOMIC_CALENDAR_CACHE_KEY, events, timeout=60 * 60 * 24 * 7)

        for event in events:
            title = str(event.get('title') or '').lower()
            if 'cpi' not in title:
                continue
            actual = _safe_float(event.get('actual'))
            forecast = _safe_float(event.get('forecast'))
            if actual is None or forecast is None:
                continue
            if actual < forecast:
                cache.set('bluechip_aggressive_multiplier', 1.05, timeout=60 * 60 * 6)
                break

        result = {'status': 'ok', 'events': len(events)}
        _task_log_finish(log, 'SUCCESS', result)
        return result
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


@shared_task
def weekend_deep_research() -> dict[str, Any]:
    log = _task_log_start('weekend_deep_research')
    try:
        penny = _analyze_penny_breakouts()
        bluechip = _analyze_bluechip_rebounds()
        payload = {
            'as_of': timezone.now().isoformat(),
            'penny': penny,
            'bluechip': bluechip,
        }
        cache.set('weekend_deep_research', payload, timeout=60 * 60 * 24 * 7)
        _task_log_finish(log, 'SUCCESS', payload)
        return payload
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


@shared_task
def sunday_evening_briefing() -> dict[str, Any]:
    log = _task_log_start('sunday_evening_briefing')
    try:
        research = cache.get('weekend_deep_research') or {}
        calendar_events = _economic_calendar_events()
        high_events = [e for e in calendar_events if _is_high_impact_event(e)]

        lines = [
            "🏛️ *RÉSUMÉ STRATÉGIQUE DE LA SEMAINE*",
        ]
        if high_events:
            lines.append("\n📅 *CALENDRIER ÉCONOMIQUE (⚠️ Risque)*")
            for event in high_events[:6]:
                dt_ny = event.get('datetime_ny')
                try:
                    when = pd.to_datetime(dt_ny, utc=True, errors='coerce')
                    if pd.isna(when):
                        when = None
                    else:
                        when = when.to_pydatetime().astimezone(ZoneInfo('America/New_York'))
                except Exception:
                    when = None
                when_txt = when.strftime('%a %H:%M') if when else 'n/a'
                lines.append(f"• {when_txt} : {event.get('title')} (Impact: {event.get('impact')})")
            lines.append("Note : l'IA passera en mode observation autour des annonces high impact.")

        penny = (research.get('penny') or {}).get('watchlist') or []
        if penny:
            lines.append("\n🚀 *WATCHLIST PENNY STOCKS*")
            for item in penny[:5]:
                catalyst = f" | News: {item['catalysts'][0]}" if item.get('catalysts') else ''
                lines.append(
                    f"${item['symbol']} : Accumulation | Vol {item['volume_ratio']}x | Score {item['score']}{catalyst}"
                )

        bluechip = (research.get('bluechip') or {}).get('selection') or []
        if bluechip:
            lines.append("\n💎 *SÉLECTION BLUE CHIPS*")
            for item in bluechip:
                lines.append(
                    f"${item['symbol']} : Rebond SMA | Score {item['score']} | Sentiment {item['sentiment']}"
                )

        message = "\n".join(lines).strip()
        if message:
            _send_telegram_alert(message, allow_during_blackout=True, category='briefing')

        payload = {'status': 'sent', 'penny': len(penny), 'bluechip': len(bluechip), 'events': len(high_events)}
        _task_log_finish(log, 'SUCCESS', payload)
        return payload
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


@shared_task
def morning_status_check() -> dict[str, Any]:
    log = _task_log_start('morning_status_check')
    try:
        price, change_pct = _spy_premarket_snapshot()
        eco_note = _economic_calendar_note_for_today()
        watchlist = _top_watchlist_symbols(limit=3)

        lines = ["☕ *Morning Brief*", ""]
        if price is not None:
            change_txt = f" ({change_pct:+.2f}%)" if change_pct is not None else ''
            lines.append(f"SPY pré-marché: {price:.2f}$" + change_txt)
        else:
            lines.append("SPY pré-marché: n/a")
        if eco_note:
            lines.append(eco_note)
        if watchlist:
            lines.append("Top 3 watchlist: " + ", ".join(watchlist))

        message = "\n".join(lines).strip()
        if message:
            _send_telegram_alert(message, allow_during_blackout=True, category='briefing')

        payload = {'status': 'sent', 'watchlist': watchlist}
        _task_log_finish(log, 'SUCCESS', payload)
        return payload
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


def _fear_greed_from_vix(vix_level: float | None) -> tuple[str, int]:
    if vix_level is None:
        return 'n/a', 50
    index = int(round(max(0.0, min(100.0, 100 - (float(vix_level) - 10) * 4))))
    if index <= 10:
        label = 'Extreme Fear'
    elif index <= 30:
        label = 'Fear'
    elif index <= 55:
        label = 'Neutral'
    elif index <= 75:
        label = 'Greed'
    else:
        label = 'Extreme Greed'
    return label, index


def _format_diag_price(value: float | None) -> str:
    if value is None:
        return 'n/a'
    price = float(value)
    if price >= 1:
        return f"{price:.2f}$"
    if price >= 0.1:
        return f"{price:.3f}$"
    return f"{price:.4f}$"


@shared_task
def nightly_intraday_retrain() -> dict[str, Any]:
    log = _task_log_start('nightly_intraday_retrain')
    try:
        enabled = os.getenv('INTRADAY_TRAINING_ENABLED', 'true').lower() in {'1', 'true', 'yes', 'y'}
        if not enabled:
            result = {'status': 'disabled'}
            _task_log_finish(log, 'SUCCESS', result)
            return result

        from .ml_engine.intraday_training import (
            train_voting_ensemble,
            train_xgboost_model,
            save_intraday_model,
            build_dataset,
        )

        bluechip = _watchlist_symbols(['AI_BLUECHIP', 'WATCHLIST'], limit=50)
        penny = _watchlist_symbols(['AI_PENNY'], limit=50)
        days = int(os.getenv('INTRADAY_TRAINING_DAYS', '180'))

        if bluechip and penny:
            result = train_voting_ensemble(bluechip, penny, market_symbol='QQQ', days=days)
            path = save_intraday_model(result, 'ensemble')
        else:
            dataset, labels, features = build_dataset(bluechip or penny, market_symbol='QQQ', days=days)
            result = train_xgboost_model(dataset, labels, features)
            path = save_intraday_model(result, 'single')

        payload = {
            'status': 'ok',
            'model_path': path,
            'scores': result.scores,
            'features': result.features,
        }
        _task_log_finish(log, 'SUCCESS', payload)
        return payload
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


@shared_task
def nightly_summary_report() -> dict[str, Any]:
    log = _task_log_start('nightly_summary_report')
    try:
        top_news = _overnight_positive_news(limit=3)
        spy_change = _percent_change('SPY')
        gold_change = _percent_change('GC=F')
        oil_change = _percent_change('CL=F')
        spy_price, spy_status = _latest_price_with_status('SPY')
        gold_price, gold_status = _latest_price_with_status('GC=F')
        oil_price, oil_status = _latest_price_with_status('CL=F')
        sentiment, meta = get_market_sentiment()
        vix = meta.get('vix')

        rec = 'Reste sur la touche'
        if sentiment == 'BULLISH' and (vix is None or float(vix) < float(os.getenv('VIX_STRESS_THRESHOLD', '30'))):
            rec = "Aujourd'hui, sois agressif"

        def _format_line(label: str, change: float | None, price: float | None, status: str) -> str:
            if change is not None:
                if price is not None:
                    return f"{label}: {change:+.2f}% ({price:.2f}$, {status})"
                return f"{label}: {change:+.2f}%"
            if price is not None:
                return f"{label}: {price:.2f}$ ({status})"
            return f"{label}: n/a"

        lines = [
            "🌙 *Nightly Summary*",
            _format_line("SPY", spy_change, spy_price, spy_status),
            _format_line("Or (GC=F)", gold_change, gold_price, gold_status),
            _format_line("Pétrole (CL=F)", oil_change, oil_price, oil_status),
            f"Régime: {sentiment}",
            "",
            "Top 3 news positives overnight:",
        ]
        if top_news:
            for item in top_news:
                headline = f" — {item['headline']}" if item.get('headline') else ''
                lines.append(f"• {item['symbol']}: {item['avg_sent']:+.2f} ({item['count']} news){headline}")
        else:
            lines.append('— Aucune news marquante')
        lines.append("")
        lines.append(f"🧭 Recommandation: {rec}")

        message = "\n".join(lines).strip()
        _send_telegram_alert(message, allow_during_blackout=True, category='report')
        payload = {'status': 'sent', 'top_news': top_news, 'sentiment': sentiment, 'vix': vix}
        _task_log_finish(log, 'SUCCESS', payload)
        return payload
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


@shared_task
def send_daily_bot_journal() -> dict[str, Any]:
    log = _task_log_start('send_daily_bot_journal')
    try:
        as_of_date = timezone.localdate()
        date_label = _format_french_date(as_of_date)

        lines = [
            f"📜 JOURNAL DE BORD DU BOT | {date_label}",
            "🕒 EXÉCUTION DES TÂCHES (LOGS)",
        ]
        lines.extend(_journal_task_logs(as_of_date))

        lines.append("")
        lines.append("🕵️ CHASSE AUX DEALS (AUTO SCANNER)")
        intro, details = _journal_scanner_sections()
        lines.append(intro)
        lines.extend(details)

        lines.append("")
        lines.append("🧪 PERFORMANCE PAPER TRADE (ALPACA)")
        closed_lines, open_lines, theoretical_pnl = _journal_paper_trades(as_of_date)

        lines.append("Positions Closes :")
        if closed_lines:
            lines.extend(closed_lines)
        else:
            lines.append("— Aucune position clôturée.")

        lines.append("")
        lines.append("Positions Actives :")
        if open_lines:
            lines.extend(open_lines)
        else:
            lines.append("— Aucune position active.")

        lines.append("")
        lines.append(f"Profit Théorique du jour : {theoretical_pnl:+.2f}$")

        lines.append("")
        lines.append("💰 RÉSUMÉ DU PORTFOLIO RÉEL")
        capital_text, positions_text, status_text = _journal_portfolio_summary()
        lines.append(f"Capital Total : {capital_text}")
        lines.append(f"Positions Ouvertes : {positions_text}")
        lines.append(f"Statut : {status_text}")

        lines.append("")
        lines.append("🛠️ CONFIGURATION "
                     "PATTERN REPLICATION" )
        enabled = os.getenv('PATTERN_REPLICATOR_ENABLED', 'true').lower() in {'1', 'true', 'yes', 'y'}
        volumez = os.getenv('PATTERN_REPLICATOR_VOLUMEZ_MIN', '2.0')
        rule = os.getenv('PATTERN_REPLICATOR_RULE', 'Price_Crossing_MA20')
        lines.append(f"Statut : {'Activé' if enabled else 'Désactivé'}")
        lines.append(f"Cible : Stocks avec VolumeZ > {volumez} et {rule}.")
        lines.append("Alerte : Priorité maximale sur Telegram si un clone est détecté.")

        message = "\n".join(lines).strip()
        _send_telegram_alert(message, allow_during_blackout=True, category='report')
        payload = {'status': 'sent', 'as_of': as_of_date.isoformat()}
        _task_log_finish(log, 'SUCCESS', payload)
        return payload
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


@shared_task
def send_telegram_report(mode: str = 'premarket') -> dict[str, Any]:
    log = _task_log_start(f'send_telegram_report:{mode}')
    try:
        mode = (mode or 'premarket').lower().strip()
        watchlist = _watchlist_symbols(['WATCHLIST', 'AI_BLUECHIP', 'AI_PENNY'], limit=15)
        min_conf = float(os.getenv('MIN_CONFIDENCE', '0.7'))
        min_rvol = float(os.getenv('REPORT_MIN_RVOL', '1.5'))

        if mode == 'premarket':
            spy_change = _percent_change('SPY')
            gold_change = _percent_change('GC=F')
            oil_change = _percent_change('CL=F')
            news_scores = []
            for symbol in watchlist:
                sent, _ = _news_sentiment_score(symbol, days=1)
                news_scores.append((symbol, sent))
            news_scores = sorted(news_scores, key=lambda x: x[1], reverse=True)[:5]

            gap_watch: list[tuple[str, float]] = []
            for symbol in watchlist:
                try:
                    hist = yf.Ticker(symbol).history(period='2d', interval='1d', timeout=10)
                    if hist is None or hist.empty or 'Close' not in hist or len(hist) < 2:
                        continue
                    prev_close = float(hist['Close'].iloc[-2])
                    last_price = _latest_price_snapshot(symbol)
                    if not _is_valid_price(last_price) or prev_close <= 0:
                        continue
                    gap_pct = ((last_price - prev_close) / prev_close) * 100
                    gap_watch.append((symbol, gap_pct))
                except Exception:
                    continue
            gap_watch = sorted(gap_watch, key=lambda x: abs(x[1]), reverse=True)[:5]

            lines = [
                "☀️ *Réveil Stratégique (Pré-Market)*",
                f"SPY: {spy_change:+.2f}%" if spy_change is not None else "SPY: n/a",
                f"Or (GC=F): {gold_change:+.2f}%" if gold_change is not None else "Or (GC=F): n/a",
                f"Pétrole (CL=F): {oil_change:+.2f}%" if oil_change is not None else "Pétrole (CL=F): n/a",
                "",
                "Watchlist — sentiment overnight:",
            ]
            if news_scores:
                for symbol, sent in news_scores:
                    lines.append(f"• {symbol}: {sent:+.2f}")
            else:
                lines.append('— n/a')

            lines.append("")
            lines.append("Gap Watch (pré-ouverture):")
            if gap_watch:
                for symbol, gap in gap_watch:
                    lines.append(f"• {symbol}: {gap:+.2f}%")
            else:
                lines.append('— n/a')

        elif mode == 'validation':
            lines = ["✅ *Validation de Tendance (09:45)*", "Bougie 15m confirmée:"]
            picks: list[str] = []
            for symbol in watchlist:
                ctx_15m = _intraday_context_for_timeframe(
                    symbol,
                    minutes=int(os.getenv('ALPACA_INTRADAY_MINUTES', '390')),
                    timeframe=15,
                    rvol_window=int(os.getenv('ALPACA_RVOL_WINDOW', '20')),
                )
                if not ctx_15m:
                    continue
                rvol = float(ctx_15m.get('rvol') or 0.0)
                if rvol < min_rvol:
                    continue
                signal_payload = _model_signal(symbol, 'BLUECHIP', get_model_path('BLUECHIP'), use_alpaca=False)
                conf = float(signal_payload.get('signal') or 0.0) if signal_payload else 0.0
                if conf < min_conf:
                    continue
                picks.append(f"• {symbol}: Conf {conf:.2f}, RVOL {rvol:.2f}")
                if len(picks) >= 3:
                    break
            if not picks:
                lines.append('— Aucun signal high conviction')
            else:
                lines.extend(picks)

        else:
            lines = ["🏦 *Institutional Flow (10:30)*"]
            flow: list[str] = []
            for symbol in watchlist:
                ctx_5m = _intraday_context_for_timeframe(
                    symbol,
                    minutes=int(os.getenv('ALPACA_INTRADAY_MINUTES', '390')),
                    timeframe=5,
                    rvol_window=int(os.getenv('ALPACA_RVOL_WINDOW', '20')),
                )
                if not ctx_5m:
                    continue
                rvol = float(ctx_5m.get('rvol') or 0.0)
                if rvol < min_rvol:
                    continue
                flow.append(f"• {symbol}: RVOL {rvol:.2f}")
                if len(flow) >= 3:
                    break
            if not flow:
                lines.append('— Flux institutionnel faible ou absent')
            else:
                lines.extend(flow)

        message = "\n".join(lines).strip()
        _send_telegram_alert(message, allow_during_blackout=True, category='report')
        payload = {'status': 'sent', 'mode': mode}
        _task_log_finish(log, 'SUCCESS', payload)
        return payload
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


def _penny_diagnostic_item(symbol: str) -> dict[str, Any] | None:
    symbol = (symbol or '').strip().upper()
    if not symbol:
        return None

    snapshot = (
        PennyStockSnapshot.objects.select_related('stock')
        .filter(stock__symbol__iexact=symbol)
        .order_by('-as_of')
        .first()
    )

    price = None
    rsi = None
    ai_score = None
    if snapshot:
        price = _safe_float(snapshot.price)
        rsi = _safe_float(snapshot.rsi)
        ai_score = _safe_float(snapshot.ai_score)

    if price is None or rsi is None:
        try:
            hist = yf.Ticker(symbol).history(period='30d', interval='1d', timeout=10)
            close = _extract_close_series(hist)
            if close is not None and not close.empty:
                if price is None:
                    price = _safe_float(close.iloc[-1])
                if rsi is None:
                    rsi = _compute_rsi(close, 14)
        except Exception:
            pass

    if price is None:
        return None

    buy_rsi = float(os.getenv('PENNY_DIAG_BUY_RSI', '32'))
    wait_rsi = float(os.getenv('PENNY_DIAG_WAIT_RSI', '40'))
    min_ai_buy = float(os.getenv('PENNY_DIAG_MIN_AI', '0.55'))

    action = '⏳ WAIT'
    if rsi is not None and rsi <= buy_rsi and (ai_score is None or ai_score >= min_ai_buy):
        action = '🚀 BUY'
    elif rsi is not None and rsi <= wait_rsi:
        action = '⏳ WAIT'
    elif ai_score is not None and ai_score >= min_ai_buy:
        action = '⏳ WAIT'
    else:
        action = '⚠️ HOLD'

    base_target = float(os.getenv('PENNY_DIAG_TARGET_BASE_PCT', '0.12'))
    ai_mult = float(os.getenv('PENNY_DIAG_TARGET_AI_MULT', '0.6'))
    score = ai_score if ai_score is not None else 0.5
    target_pct = base_target + (score * ai_mult)
    if rsi is not None and rsi < 30:
        target_pct += 0.1
    target_pct = max(0.05, min(0.8, target_pct))

    stop_pct = float(os.getenv('PENNY_DIAG_STOP_PCT', '0.12'))
    if rsi is not None and rsi < 30:
        stop_pct = float(os.getenv('PENNY_DIAG_STOP_PCT_OVERSOLD', '0.16'))

    target_price = price * (1 + target_pct)
    stop_price = price * (1 - stop_pct)
    roi_pct = target_pct * 100

    bncd_threshold = float(os.getenv('PENNY_DIAG_BNCD_THRESHOLD', '0.4'))
    bncd_txt = f" 🎯 BNCD: {_format_diag_price(target_price)}" if target_pct >= bncd_threshold else ''

    rsi_txt = f"{rsi:.1f}" if rsi is not None else 'N/A'
    return {
        'symbol': symbol,
        'price': price,
        'action': action,
        'rsi': rsi,
        'roi_pct': roi_pct,
        'target_price': target_price,
        'stop_price': stop_price,
        'bncd_txt': bncd_txt,
        'rsi_txt': rsi_txt,
    }


@shared_task
def send_penny_rebound_diagnostic() -> dict[str, Any]:
    log = _task_log_start('send_penny_rebound_diagnostic')
    try:
        limit = int(os.getenv('PENNY_DIAGNOSTIC_LIMIT', '5'))
        vix_level = _get_vix_level()
        label, index = _fear_greed_from_vix(vix_level)

        symbols: list[str] = []
        watch = SandboxWatchlist.objects.filter(sandbox='AI_PENNY').first()
        if watch and watch.symbols:
            symbols = [str(s).strip().upper() for s in watch.symbols if str(s).strip()]

        if not symbols:
            snapshots = PennyStockSnapshot.objects.select_related('stock').order_by('-as_of', '-ai_score')
            if snapshots.exists():
                latest_date = snapshots.first().as_of
                snapshots = snapshots.filter(as_of=latest_date)
            symbols = [s.stock.symbol for s in snapshots[:limit]]

        hive_symbol = os.getenv('HIVE_REBOUND_SYMBOL', 'HIVE.V').strip().upper()
        if hive_symbol:
            symbols = [s for s in symbols if str(s).strip().upper() != hive_symbol]
            symbols.insert(0, hive_symbol)

        diagnostics: list[dict[str, Any]] = []
        for symbol in symbols:
            item = _penny_diagnostic_item(symbol)
            if item:
                diagnostics.append(item)
            if len(diagnostics) >= limit:
                break

        lines = [
            "DIAGNOSTIC GLOBAL",
            f"🌍 Sentiment: {label} ({index}/100)",
            "━━━━━━━━━━━━━━━",
        ]

        if diagnostics:
            for item in diagnostics:
                lines.append(f"{item['symbol']}: {_format_diag_price(item['price'])} ({item['action']})")
                lines.append(f"└ RSI: {item['rsi_txt']} | ROI: {item['roi_pct']:.1f}%")
                lines.append(
                    f"└ Obj: {_format_diag_price(item['target_price'])}{item['bncd_txt']}"
                )
                lines.append(f"└ 🛡️ SL: {_format_diag_price(item['stop_price'])}")
                lines.append("")
        else:
            lines.append("Aucune donnée penny disponible.")

        message = "\n".join(lines).strip()
        _send_telegram_alert(message, allow_during_blackout=True, category='report')

        payload = {'status': 'sent', 'count': len(diagnostics), 'sentiment_index': index}
        _task_log_finish(log, 'SUCCESS', payload)
        return payload
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


def _opening_price_today(symbol: str) -> float | None:
    symbol = (symbol or '').strip().upper()
    if not symbol:
        return None
    df = get_intraday_bars(symbol, minutes=120)
    if df is None or df.empty:
        return None
    ts = None
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    elif df.index is not None:
        ts = pd.to_datetime(df.index, utc=True, errors='coerce')
    if ts is None:
        return None
    try:
        ts_ny = ts.tz_convert('America/New_York')
    except Exception:
        return None
    df = df.copy()
    df['ts_ny'] = ts_ny
    today = _ny_time_now().date()
    df_today = df[df['ts_ny'].dt.date == today].sort_values('ts_ny')
    if df_today.empty:
        return None
    first = df_today.iloc[0]
    for key in ['open', 'Open', 'close', 'Close']:
        if key in first:
            try:
                value = float(first.get(key) or 0)
            except Exception:
                value = 0.0
            if value > 0:
                return value
    return None


@shared_task
def rebound_rsi_alert(symbols: list[str] | None = None, rsi_threshold: float = 30.0) -> dict[str, Any]:
    log = _task_log_start('rebound_rsi_alert')
    try:
        default_hive = os.getenv('HIVE_REBOUND_SYMBOL', 'HIVE.V')
        symbols = [s.strip().upper() for s in (symbols or [default_hive, 'ARRY']) if s and str(s).strip()]
        results: dict[str, Any] = {}
        for symbol in symbols:
            hist = yf.Ticker(symbol).history(period='90d', interval='1d')
            close = _extract_close_series(hist)
            if close is None or close.empty:
                results[symbol] = {'status': 'no_data'}
                continue
            rsi = _compute_rsi(close, 14)
            results[symbol] = {'rsi': rsi}
            if rsi is not None and rsi < rsi_threshold:
                currency = _symbol_currency(symbol)
                _send_telegram_alert(
                    f"🚀 ACHAT DE REBOND : ${symbol} ({currency}) | RSI {rsi:.2f} (< {rsi_threshold:.0f})",
                    allow_during_blackout=True,
                    category='signal',
                )
                results[symbol]['alerted'] = True
            else:
                results[symbol]['alerted'] = False
        payload = {'status': 'ok', 'results': results}
        _task_log_finish(log, 'SUCCESS', payload)
        return payload
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


@shared_task
def hive_opening_rebound_alert() -> dict[str, Any]:
    log = _task_log_start('hive_opening_rebound_alert')
    try:
        symbol = os.getenv('HIVE_REBOUND_SYMBOL', 'HIVE.V').strip().upper() or 'HIVE.V'
        hist = yf.Ticker(symbol).history(period='90d', interval='1d')
        close = _extract_close_series(hist)
        rsi = _compute_rsi(close, 14) if close is not None and not close.empty else None
        open_price = _opening_price_today(symbol)
        if open_price is None:
            open_price = get_latest_trade_price(symbol)
        status = 'no_open'
        currency = _symbol_currency(symbol)
        message = f"🚀 HIVE Rebound Alert: prix d'ouverture indisponible."

        if open_price is not None:
            status = 'evaluated'
            if rsi is not None and rsi < 30 and 2.05 <= open_price <= 2.15:
                message = (
                    f"🚀 SIGNAL BUY CONFIRMÉ : HIVE ouvre à {open_price:.2f} {currency} | RSI {rsi:.2f}"
                )
            elif open_price < 2.00:
                message = (
                    f"⛔ SIGNAL ANNULÉ - TROP RISQUÉ : HIVE ouvre à {open_price:.2f} {currency}"
                )
            elif rsi is not None and rsi < 30:
                message = (
                    f"⚠️ SIGNAL REBOND : HIVE ouvre à {open_price:.2f} {currency} | RSI {rsi:.2f} (hors plage 2.05–2.15)"
                )
            else:
                rsi_txt = f"{rsi:.2f}" if rsi is not None else 'n/a'
                message = (
                    f"ℹ️ HIVE : pas de signal rebond | Open {open_price:.2f} {currency} | RSI {rsi_txt}"
                )

        _send_telegram_alert(message, allow_during_blackout=True, category='signal')

        payload = {
            'status': status,
            'open_price': open_price,
            'rsi': rsi,
        }
        _task_log_finish(log, 'SUCCESS', payload)
        return payload
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


@shared_task
def monitor_hive_trade() -> dict[str, Any]:
    log = _task_log_start('monitor_hive_trade')
    try:
        symbol = os.getenv('HIVE_MONITOR_SYMBOL', 'HIVE.V').strip().upper() or 'HIVE.V'
        limit_price = float(os.getenv('HIVE_ENTRY_LIMIT', '2.95'))
        target1_pct = float(os.getenv('HIVE_TARGET1_PCT', '0.10'))
        target2_pct = float(os.getenv('HIVE_TARGET2_PCT', '0.15'))
        stop_pct = float(os.getenv('HIVE_STOP_PCT', '0.05'))
        rsi_sell = float(os.getenv('HIVE_RSI_SELL', '60'))
        currency = _symbol_currency(symbol)
        stock = Stock.objects.filter(symbol__iexact=symbol).first()
        filled_cooldown = int(os.getenv('HIVE_FILLED_COOLDOWN_MIN', '480'))
        rsi_cooldown = int(os.getenv('HIVE_RSI_COOLDOWN_MIN', '120'))
        nofill_cooldown = int(os.getenv('HIVE_NOFILL_COOLDOWN_MIN', '360'))
        target_cooldown = int(os.getenv('HIVE_TARGET_COOLDOWN_MIN', '480'))
        stop_cooldown = int(os.getenv('HIVE_STOP_COOLDOWN_MIN', '480'))

        def _send_hive_alert(category: str, message: str, cooldown_min: int) -> bool:
            cutoff = timezone.now() - timedelta(minutes=cooldown_min)
            if AlertEvent.objects.filter(category=category, stock=stock, created_at__gte=cutoff).exists():
                return False
            AlertEvent.objects.create(category=category, message=message, stock=stock)
            _send_telegram_alert(message, allow_during_blackout=True, category='signal')
            return True
        now_ny = _ny_time_now()
        if now_ny.weekday() >= 5:
            _task_log_finish(log, 'SUCCESS', {'status': 'market_closed'})
            return {'status': 'market_closed'}
        if not (dt_time(9, 30) <= now_ny.time() <= dt_time(16, 0)):
            _task_log_finish(log, 'SUCCESS', {'status': 'outside_session'})
            return {'status': 'outside_session'}

        latest_price = get_latest_trade_price(symbol)
        if latest_price is None:
            latest_price = yf.Ticker(symbol).history(period='1d', interval='1m')
            if latest_price is not None and not latest_price.empty:
                latest_price = float(_extract_close_series(latest_price).iloc[-1])

        if latest_price is None:
            _task_log_finish(log, 'SUCCESS', {'status': 'no_price'})
            return {'status': 'no_price'}

        filled_key = _cache_key(symbol, 'filled')
        filled = cache.get(filled_key, False)
        if not filled and latest_price >= limit_price:
            cache.set(filled_key, True, timeout=60 * 60 * 8)
            _send_hive_alert(
                'HIVE_FILLED',
                f"✅ ORDRE PROBABLEMENT FILLED : {symbol} à {latest_price:.2f} {currency} (limite {limit_price:.2f}).",
                filled_cooldown,
            )
            filled = True

        stop_price = limit_price * (1 - stop_pct)
        stop_key = _cache_key(symbol, 'stop')
        if filled and latest_price <= stop_price and not cache.get(stop_key, False):
            cache.set(stop_key, True, timeout=60 * 60 * 8)
            _send_hive_alert(
                'HIVE_STOP',
                f"⛔ STOP-LOSS ATTEINT : {symbol} à {latest_price:.2f} {currency} (stop {stop_price:.2f}).",
                stop_cooldown,
            )

        target1 = limit_price * (1 + target1_pct)
        target2 = limit_price * (1 + target2_pct)
        target1_key = _cache_key(symbol, 'target10')
        target2_key = _cache_key(symbol, 'target15')
        if filled and latest_price >= target1 and not cache.get(target1_key, False):
            cache.set(target1_key, True, timeout=60 * 60 * 8)
            _send_hive_alert(
                'HIVE_TARGET_10',
                f"✅ PALIER 10% ATTEINT : {symbol} à {latest_price:.2f} {currency} (cible {target1:.2f}).",
                target_cooldown,
            )
        if filled and latest_price >= target2 and not cache.get(target2_key, False):
            cache.set(target2_key, True, timeout=60 * 60 * 8)
            _send_hive_alert(
                'HIVE_TARGET_15',
                f"🔥 CIBLE 15% ATTEINTE : {symbol} à {latest_price:.2f} {currency} (cible {target2:.2f}).",
                target_cooldown,
            )

        intraday = yf.Ticker(symbol).history(period='5d', interval='15m')
        close_15m = _extract_close_series(intraday)
        rsi_15m = _compute_rsi(close_15m, 14) if close_15m is not None and not close_15m.empty else None
        rsi_key = _cache_key(symbol, 'rsi')
        rsi_alerts_enabled = os.getenv('HIVE_RSI_ALERTS', 'false').lower() in {'1', 'true', 'yes', 'y'}
        if rsi_alerts_enabled and filled and rsi_15m is not None and rsi_15m >= rsi_sell and not cache.get(rsi_key, False):
            cache.set(rsi_key, True, timeout=60 * 60 * 8)
            _send_hive_alert(
                'HIVE_RSI_HOT',
                f"⚠️ RSI CHAUD : {symbol} RSI15m {rsi_15m:.1f} ≥ {rsi_sell:.0f}. Recommandation : vendre.",
                rsi_cooldown,
            )

        if now_ny.time() >= dt_time(10, 0) and not filled:
            advice_key = _cache_key(symbol, '10am')
            if not cache.get(advice_key, False):
                cache.set(advice_key, True, timeout=60 * 60 * 6)
                bid, ask = _latest_bid_ask(symbol)
                if bid is not None and ask is not None and ask > 0:
                    spread = (ask - bid) / ask
                    if bid < limit_price and spread > 0.01:
                        message = (
                            f"⏰ 10:00 — ORDRE NON REMPLI. Bid {bid:.2f} / Ask {ask:.2f} {currency}. "
                            "Recommandation : monter légèrement la limite ou attendre confirmation."
                        )
                    else:
                        message = (
                            f"⏰ 10:00 — ORDRE NON REMPLI. Bid {bid:.2f} / Ask {ask:.2f} {currency}. "
                            "Signal en attente, pas d'ajustement nécessaire."
                        )
                else:
                    message = (
                        "⏰ 10:00 — ORDRE NON REMPLI. Carnet indisponible. "
                        "Recommandation : ajuster la limite si le prix reste sous 2.95."
                    )
                _send_hive_alert('HIVE_NOFILL_10AM', message, nofill_cooldown)

        payload = {
            'status': 'ok',
            'symbol': symbol,
            'price': latest_price,
            'filled': bool(filled),
            'rsi_15m': rsi_15m,
            'target1': target1,
            'target2': target2,
        }
        _task_log_finish(log, 'SUCCESS', payload)
        return payload
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


@shared_task
def daily_profit_tracker() -> dict[str, Any]:
    log = _task_log_start('daily_profit_tracker')
    try:
        investment = float(os.getenv('PROFIT_TRACKER_INVESTMENT', '200'))
        today = timezone.localdate()
        signals = ActiveSignal.objects.filter(closed_at__date=today)

        wins = 0
        losses = 0
        profit_total = 0.0
        win_details: list[str] = []
        loss_details: list[str] = []
        for signal in signals:
            entry = float(signal.entry_price or 0)
            exit_price = float(signal.closed_price or 0)
            if entry <= 0 or exit_price <= 0:
                continue
            ret_pct = (exit_price - entry) / entry
            pnl = investment * ret_pct
            profit_total += pnl
            if pnl >= 0:
                wins += 1
                win_details.append(
                    f"• {signal.ticker}: {entry:.2f} → {exit_price:.2f} | +{pnl:.2f}$ (+{ret_pct * 100:.2f}%)"
                )
            else:
                losses += 1
                loss_details.append(
                    f"• {signal.ticker}: {entry:.2f} → {exit_price:.2f} | {pnl:.2f}$ ({ret_pct * 100:.2f}%)"
                )

        wallet = _load_profit_wallet()
        week_start = _week_start_date(timezone.now()).isoformat()
        if wallet.get('week_start') != week_start:
            wallet = {'week_start': week_start, 'weekly_total': 0.0}
        wallet['weekly_total'] = float(wallet.get('weekly_total') or 0) + profit_total
        wallet['updated_at'] = timezone.now().isoformat()
        _save_profit_wallet(wallet)

        day_pct = (profit_total / investment * 100) if investment else 0.0
        lines = [
            "🏦 *BILAN DE TA JOURNÉE (Swing Mode)*",
            f"✅ Trades Gagnants : {wins}",
            f"❌ Trades Perdants : {losses}",
        ]
        if win_details or loss_details:
            lines.append("")
            if win_details:
                lines.append("✅ Détails gagnants:")
                lines.extend(win_details[:5])
            else:
                lines.append("✅ Détails gagnants: Aucun")
            if loss_details:
                lines.append("❌ Détails perdants:")
                lines.extend(loss_details[:5])
            else:
                lines.append("❌ Détails perdants: Aucun")
        lines.extend(
            [
                "",
                f"💰 Profit du jour : {profit_total:+.2f}$",
                f"📈 Rendement : {day_pct:+.2f}% (base {investment:.0f}$ par trade)",
                "",
                f"📅 CAGNOTTE DE LA SEMAINE : {wallet['weekly_total']:+.2f}$",
                "Objectif Vendredi : 100$",
                "",
                "🧠 Note de l'IA : Sentiment news stable aujourd'hui.",
                "Le modèle sera réentraîné ce soir. Prêt pour demain !",
            ]
        )
        _send_telegram_alert("\n".join(lines).strip(), allow_during_blackout=True, category='report')

        payload = {
            'status': 'sent',
            'wins': wins,
            'losses': losses,
            'profit_total': round(profit_total, 2),
            'weekly_total': round(float(wallet['weekly_total']), 2),
        }
        _task_log_finish(log, 'SUCCESS', payload)
        return payload
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


def _portfolio_news_sentiment(symbol: str, hours: int = 24) -> float:
    cutoff = timezone.now() - timedelta(hours=hours)
    avg = (
        StockNews.objects.filter(stock__symbol__iexact=symbol, published_at__gte=cutoff)
        .aggregate(avg=models.Avg('sentiment'))
        .get('avg')
    )
    if avg is None:
        sentiment, _ = _news_sentiment_score(symbol, days=1)
        return float(sentiment)
    return float(avg)


def _normalize_yf_symbol(symbol: str) -> str:
    sym = (symbol or '').strip().upper()
    if sym.startswith('$'):
        sym = sym[1:]
    if '.' not in sym:
        force_v = [s.strip().upper() for s in os.getenv('FORCE_V_SUFFIX_SYMBOLS', '').split(',') if s.strip()]
        force_to = [s.strip().upper() for s in os.getenv('FORCE_TO_SUFFIX_SYMBOLS', '').split(',') if s.strip()]
        if sym in force_v:
            sym = f"{sym}.V"
        elif sym in force_to:
            sym = f"{sym}.TO"
    return sym


def _pct_change_from_open(symbol: str) -> float | None:
    try:
        hist = yf.download(tickers=_normalize_yf_symbol(symbol), period='1d', interval='1m')
    except Exception:
        hist = None
    if hist is None or hist.empty:
        return None
    if 'Open' not in hist.columns or 'Close' not in hist.columns:
        return None
    first_open = float(hist['Open'].iloc[0]) if float(hist['Open'].iloc[0]) else None
    last_close = float(hist['Close'].iloc[-1]) if float(hist['Close'].iloc[-1]) else None
    if not first_open or not last_close:
        return None
    return (last_close - first_open) / first_open


def _intraday_pct_change(symbol: str, minutes: int = 20) -> float | None:
    try:
        hist = yf.download(tickers=_normalize_yf_symbol(symbol), period='1d', interval='1m')
    except Exception:
        hist = None
    if hist is None or hist.empty or 'Close' not in hist.columns:
        return None
    close = hist['Close'].dropna()
    if close.empty:
        return None
    if minutes <= 1 or len(close) < minutes:
        minutes = min(len(close), max(2, minutes))
    start_price = float(close.iloc[-minutes]) if float(close.iloc[-minutes]) else None
    end_price = float(close.iloc[-1]) if float(close.iloc[-1]) else None
    if not start_price or not end_price:
        return None
    return (end_price - start_price) / start_price


def _is_market_dump() -> bool:
    dump_pct = float(os.getenv('PENNY_SNIPER_MARKET_DUMP_PCT', '-0.007'))
    parent = os.getenv('PENNY_SNIPER_MARKET_PARENT', 'SPY').strip().upper() or 'SPY'
    change = _intraday_pct_change(parent, minutes=int(os.getenv('PENNY_SNIPER_MARKET_MINUTES', '30')))
    if change is None:
        return False
    return change <= dump_pct


def _intraday_volume(symbol: str) -> float | None:
    try:
        hist = yf.download(tickers=_normalize_yf_symbol(symbol), period='1d', interval='1m')
    except Exception:
        hist = None
    if hist is None or hist.empty or 'Volume' not in hist.columns:
        return None
    return float(hist['Volume'].sum())


def _daily_history(symbol: str, days: int = 365) -> pd.DataFrame:
    try:
        hist = yf.download(tickers=_normalize_yf_symbol(symbol), period=f'{days}d', interval='1d')
    except Exception:
        return pd.DataFrame()
    return hist if isinstance(hist, pd.DataFrame) else pd.DataFrame()


CORRELATION_MAP: dict[str, str] = {
    # CRYPTO
    'HIVE': 'BTC-USD',
    'HUT': 'BTC-USD',
    'BITF': 'BTC-USD',
    'MARA': 'BTC-USD',
    'RIOT': 'BTC-USD',
    'CLSK': 'BTC-USD',
    # MINING / METALS
    'TECK.TO': 'HG=F',
    'ABX.TO': 'GC=F',
    'FNV.TO': 'GC=F',
    # BANKS (Canada)
    'RY.TO': 'ZEB.TO',
    'TD.TO': 'ZEB.TO',
    # ENERGY
    'SU.TO': 'CL=F',
    'CNQ.TO': 'CL=F',
    # RETAIL / CONSUMER
    'ATD.TO': 'CADUSD=X',
}

CRYPTO_TICKERS = {'HIVE', 'HUT', 'BITF', 'MARA', 'RIOT', 'CLSK'}


def _intraday_ohlc_5m(symbol: str, minutes: int = 120) -> pd.DataFrame:
    bars = get_intraday_bars(symbol, minutes=minutes)
    if bars is None or bars.empty:
        return pd.DataFrame()
    frame = bars.copy()
    if 'timestamp' in frame.columns:
        frame['timestamp'] = pd.to_datetime(frame['timestamp'], errors='coerce', utc=True)
        frame = frame.dropna(subset=['timestamp']).set_index('timestamp')
    if not isinstance(frame.index, pd.DatetimeIndex):
        return pd.DataFrame()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in frame.columns:
            frame[col] = pd.NA
    ohlc = frame.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    })
    ohlc = ohlc.dropna(subset=['open', 'high', 'low', 'close'])
    return ohlc


def analyze_with_gemini(
    ticker: str,
    df_ohlc: pd.DataFrame,
    news_sentiment: float | None,
    correlation_data: dict[str, Any] | None,
) -> str | None:
    api_key = getattr(settings, 'GEMINI_AI_API_KEY', None)
    if not api_key or genai is None:
        return None
    try:
        client = genai.Client(api_key=api_key)
        ohlc_text = 'n/a'
        if df_ohlc is not None and not df_ohlc.empty:
            ohlc_text = df_ohlc.tail(10).to_string(index=False)
        prompt = (
            "Tu es un expert en Day Trading de Penny Stocks. Analyse les données suivantes pour "
            f"{ticker}. Données Techniques (5min) : {ohlc_text}. Sentiment des News : {news_sentiment}. "
            f"Corrélation avec actif parent : {correlation_data}. Donne un verdict clair : ACHAT, ATTENTE ou DANGER. "
            "Inclus un prix d'entrée, un objectif de profit (+10%) et un stop-loss (-4%). Sois concis pour un message Telegram."
        )
        response = client.models.generate_content(model='gemini-1.5-flash', contents=prompt)
        return (getattr(response, 'text', None) or '').strip() or None
    except Exception as exc:
        message = str(exc).lower()
        if 'quota' in message or 'resource_exhausted' in message or '429' in message:
            return "Analyse Gemini indisponible (quota gratuit dépassé)."
        return f"Analyse Gemini indisponible ({str(exc)})"


def _penny_sniper_candidate(symbol: str) -> dict[str, Any] | None:
    symbol = (symbol or '').strip().upper()
    if not symbol:
        return None

    parent = CORRELATION_MAP.get(symbol) or os.getenv('PENNY_SNIPER_DEFAULT_PARENT', 'SPY').strip().upper() or 'SPY'
    corr_minutes = int(os.getenv('PENNY_SNIPER_CORR_MINUTES', '20'))
    parent_change = _intraday_pct_change(parent, minutes=corr_minutes)
    require_parent_up = True
    if symbol in CRYPTO_TICKERS:
        require_parent_up = True
    if require_parent_up and parent_change is not None and parent_change <= 0:
        return None

    hist = _daily_history(symbol, days=365)
    if hist.empty or 'Close' not in hist.columns:
        return None

    close = hist['Close'].dropna()
    if close.empty:
        return None
    price = float(close.iloc[-1])
    min_price = float(os.getenv('PENNY_SNIPER_MIN_PRICE', '0.5'))
    max_price = float(os.getenv('PENNY_SNIPER_MAX_PRICE', '4.5'))
    if price < min_price or price > max_price:
        return None

    avg_volume = float(hist['Volume'].tail(90).mean()) if 'Volume' in hist.columns else 0.0
    min_volume = float(os.getenv('PENNY_SNIPER_MIN_VOLUME', '500000'))
    if avg_volume < min_volume:
        return None

    rsi = _compute_rsi(close, 14)
    if rsi is None:
        return None
    max_rsi = float(os.getenv('PENNY_SNIPER_MAX_RSI', '30'))
    if symbol.endswith('.TO') and symbol not in CRYPTO_TICKERS:
        max_rsi = float(os.getenv('PENNY_SNIPER_MAX_RSI_BLUECHIP', '40'))
    if rsi > max_rsi:
        return None

    high_52w = float(close.tail(252).max()) if len(close) >= 252 else float(close.max())
    if high_52w <= 0:
        return None
    dist_from_high = (price - high_52w) / high_52w
    if dist_from_high > -0.70:
        return None

    change_from_open = _pct_change_from_open(symbol)
    if change_from_open is None or change_from_open <= 0:
        return None

    intraday_vol = _intraday_volume(symbol)
    if intraday_vol is None or avg_volume <= 0:
        return None
    rvol = intraday_vol / avg_volume if avg_volume else 0.0
    if rvol < float(os.getenv('PENNY_SNIPER_MIN_RVOL', '2.0')):
        return None

    target_pct = float(os.getenv('PENNY_SNIPER_TARGET_PCT', '0.14'))
    stop_pct = float(os.getenv('PENNY_SNIPER_STOP_PCT', '0.05'))
    if _is_market_dump():
        stop_pct = float(os.getenv('PENNY_SNIPER_STOP_PCT_DUMP', '0.03'))
    target = price * (1 + target_pct)
    stop = price * (1 - stop_pct)

    return {
        'symbol': symbol,
        'price': price,
        'rsi': rsi,
        'rvol': rvol,
        'change_from_open': change_from_open,
        'parent': parent,
        'parent_change': parent_change,
        'target': target,
        'stop': stop,
        'roi_pct': target_pct * 100,
    }


@shared_task
def penny_sniper_alert() -> dict[str, Any]:
    log = _task_log_start('penny_sniper_alert')
    try:
        enabled = os.getenv('PENNY_SNIPER_ENABLED', 'true').lower() in {'1', 'true', 'yes', 'y'}
        if not enabled:
            return {'status': 'disabled'}

        watch = SandboxWatchlist.objects.filter(sandbox='WATCHLIST').first()
        symbols = [str(s).strip().upper() for s in (watch.symbols if watch else []) if str(s).strip()]
        if not symbols:
            return {'status': 'no_symbols'}

        alerts = []
        for symbol in symbols:
            candidate = _penny_sniper_candidate(symbol)
            if not candidate:
                continue
            cache_key = _cache_key(symbol, 'penny_sniper')
            if cache.get(cache_key):
                continue
            cache.set(cache_key, True, timeout=60 * 20)

            parent_txt = 'n/a'
            if candidate.get('parent'):
                change = candidate.get('parent_change')
                if change is not None:
                    parent_txt = f"{candidate['parent']} {change * 100:+.1f}%"
                else:
                    parent_txt = f"{candidate['parent']} n/a"

            news_sentiment, _ = _news_sentiment_score(symbol, days=1)
            correlation_data = {
                'parent': candidate.get('parent'),
                'parent_change_pct': round(float(candidate.get('parent_change') or 0) * 100, 2),
            }
            df_ohlc = _intraday_ohlc_5m(symbol, minutes=120)
            gemini_note = analyze_with_gemini(symbol, df_ohlc, news_sentiment, correlation_data)

            message = (
                f"🎯 Opportunité à Rabais : {symbol}\n\n"
                f"Prix : {candidate['price']:.2f}$ (RSI: {candidate['rsi']:.1f})\n"
                f"Signal : Volume {candidate['rvol']:.1f}x · Corrélation {parent_txt}\n"
                f"Plan : In @ {candidate['price']:.2f}$ / Out @ {candidate['target']:.2f}$ (+{candidate['roi_pct']:.1f}%) "
                f"/ Stop @ {candidate['stop']:.2f}$"
            )
            if gemini_note:
                message = f"{message}\n\n🧠 Gemini : {gemini_note}"
            _send_telegram_alert(message, allow_during_blackout=True, category='signal')
            alerts.append(symbol)

        payload = {'status': 'ok', 'alerts': alerts}
        _task_log_finish(log, 'SUCCESS', payload)
        return payload
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


def _latest_news_headline(symbol: str) -> str | None:
    symbol = (symbol or '').strip().upper()
    if not symbol:
        return None
    stock = Stock.objects.filter(symbol__iexact=symbol).first()
    if not stock:
        return None
    news = StockNews.objects.filter(stock=stock).order_by('-published_at', '-fetched_at').first()
    if news and news.headline:
        return str(news.headline)
    return None


@shared_task
def task_audit_portfolio_complet(is_close: bool = False) -> dict[str, Any]:
    log = _task_log_start('task_audit_portfolio_complet')
    try:
        client = get_trading_client()
        if client is None:
            payload = {'status': 'no_client'}
            _task_log_finish(log, 'SUCCESS', payload)
            return payload

        positions = []
        try:
            positions = client.get_all_positions() or []
        except Exception:
            try:
                positions = client.list_positions() or []
            except Exception:
                positions = []

        if not positions:
            payload = {'status': 'empty'}
            _task_log_finish(log, 'SUCCESS', payload)
            return payload

        rows = []
        for pos in positions:
            symbol = str(getattr(pos, 'symbol', '') or getattr(pos, 'ticker', '')).strip().upper()
            entry_price = float(getattr(pos, 'avg_entry_price', 0) or getattr(pos, 'entry_price', 0) or 0)
            current_price = float(getattr(pos, 'current_price', 0) or getattr(pos, 'market_value', 0) or 0)
            pnl_pct = None
            unrealized_plpc = getattr(pos, 'unrealized_plpc', None)
            if unrealized_plpc is not None:
                try:
                    pnl_pct = float(unrealized_plpc) * 100
                except Exception:
                    pnl_pct = None
            if pnl_pct is None and entry_price:
                pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price else 0.0
            news_title = _latest_news_headline(symbol) or ''
            rows.append({
                'ticker': symbol,
                'entry_price': round(entry_price, 4),
                'current_price': round(current_price, 4),
                'pnl_pct': round(float(pnl_pct or 0), 2),
                'news': news_title,
            })

        api_key = getattr(settings, 'GEMINI_AI_API_KEY', None)
        if not api_key or genai is None:
            payload = {'status': 'no_api_key', 'count': len(rows)}
            _task_log_finish(log, 'SUCCESS', payload)
            return payload

        client = genai.Client(api_key=api_key)
        prompt = (
            f"Voici mon portfolio actuel : {rows}. En tant qu'expert en gestion de risque, analyse chaque ligne. "
            "Pour chacune, réponds brièvement : 1. Statut (HOLD/SELL/TAKE PROFIT), 2. Raison rapide (ex: News négative, "
            "Bitcoin chute, ou Momentum haussier), et 3. Urgence (Bas/Moyen/Haut). Termine par un conseil sur mon exposition totale."
        )
        try:
            response = client.models.generate_content(model='gemini-1.5-flash', contents=prompt)
            analysis = (getattr(response, 'text', None) or '').strip()
        except Exception as exc:
            message = str(exc).lower()
            if 'quota' in message or 'resource_exhausted' in message or '429' in message:
                analysis = "Analyse Gemini indisponible (quota gratuit dépassé)."
            else:
                analysis = f"Analyse Gemini indisponible ({str(exc)})"

        if analysis:
            prefix = "⚠️ DÉCISION DE CLÔTURE REQUISE\n" if is_close else ""
            timestamp = timezone.localtime().strftime('%Y-%m-%d %H:%M')
            message = f"{prefix}{analysis}\n\n⏱️ {timestamp}"
            _send_telegram_alert(message, category='portfolio_audit')

        payload = {'status': 'ok', 'count': len(rows)}
        _task_log_finish(log, 'SUCCESS', payload)
        return payload
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


def _portfolio_symbols_from_env() -> tuple[list[str], list[str]]:
    core = [s.strip().upper() for s in os.getenv('MY_PORTFOLIO_CORE', 'RY.TO,ATD.TO,TEC.TO').split(',') if s.strip()]
    moon = [s.strip().upper() for s in os.getenv('MY_PORTFOLIO_MOONSHOTS', 'PDN.TO,MN.V,ONCY').split(',') if s.strip()]
    return core, moon


@shared_task
def monitor_my_portfolio() -> dict[str, Any]:
    log = _task_log_start('monitor_my_portfolio')
    try:
        core_symbols, moon_symbols = _portfolio_symbols_from_env()
        symbols = list(dict.fromkeys(core_symbols + moon_symbols))
        if not symbols:
            _task_log_finish(log, 'SUCCESS', {'status': 'empty'})
            return {'status': 'empty'}

        alerts_sent = 0

        for symbol in symbols:
            is_core = symbol in core_symbols
            interval = '60m' if is_core else '15m'
            period = '60d' if is_core else '10d'
            hist = yf.Ticker(symbol).history(period=period, interval=interval)
            if hist is None or hist.empty or 'Close' not in hist:
                continue
            close = hist['Close']
            price = float(close.iloc[-1])
            rsi = _compute_rsi(close, 14)
            macd_line, macd_signal, macd_hist = _compute_macd(close)
            macd_series = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
            macd_hist_series = macd_series - macd_series.ewm(span=9, adjust=False).mean()
            macd_weak = _macd_weakening(macd_hist_series) if len(macd_hist_series) >= 3 else False

            daily = yf.Ticker(symbol).history(period='1y', interval='1d')
            sma200 = None
            prev_close = None
            if daily is not None and not daily.empty and 'Close' in daily:
                series = daily['Close']
                sma200 = float(series.rolling(200).mean().iloc[-1]) if len(series) >= 200 else None
                prev_close = float(series.iloc[-2]) if len(series) >= 2 else None

            _, _, bb_lower = _compute_bollinger(daily['Close'] if daily is not None and 'Close' in daily else close, 20)
            news_sent = _portfolio_news_sentiment(symbol, hours=24)
            negative_news = news_sent < -0.2

            if rsi is not None and rsi >= 75:
                _send_telegram_alert(
                    f"⚠️ SÉCURISATION : ${symbol} en surchauffe (RSI {rsi:.1f}). "
                    "Je suggère de vendre 50% pour sécuriser les gains.",
                    allow_during_blackout=True,
                    category='portfolio_guard',
                )
                alerts_sent += 1
                continue

            if macd_hist is not None and macd_hist < 0 and macd_weak:
                _send_telegram_alert(
                    f"⚠️ SÉCURISATION : ${symbol} montre un essoufflement MACD. "
                    "Je suggère de vendre 50%.",
                    allow_during_blackout=True,
                    category='portfolio_guard',
                )
                alerts_sent += 1
                continue

            if prev_close and prev_close > 0:
                drop_pct = ((price - prev_close) / prev_close) * 100
                near_support = False
                if sma200:
                    near_support = abs(price - sma200) / sma200 <= 0.015
                if bb_lower:
                    near_support = near_support or price <= bb_lower * 1.01
                if drop_pct <= -3 and near_support and not negative_news:
                    _send_telegram_alert(
                        f"📉 BUY THE DIP : ${symbol} est sur un support majeur. "
                        "Bon moment pour accumuler.",
                        allow_during_blackout=True,
                        category='portfolio_guard',
                    )
                    alerts_sent += 1

            if not is_core:
                vol = hist['Volume'] if 'Volume' in hist else None
                if vol is not None and len(vol) >= 20:
                    rvol = float(vol.iloc[-1]) / float(vol.tail(20).mean()) if float(vol.tail(20).mean()) else 0.0
                    if rvol >= 2 and not negative_news:
                        _send_telegram_alert(
                            f"🚀 Volume anormal sur ${symbol} (RVOL {rvol:.2f}). "
                            "News positives détectées.",
                            allow_during_blackout=True,
                            category='portfolio_guard',
                        )
                        alerts_sent += 1

            if symbol.upper() == 'TEC.TO':
                spy_price, spy_change = _spy_premarket_snapshot()
                if spy_change is not None and spy_change <= -1.0:
                    _send_telegram_alert(
                        "⚠️ Nasdaq en baisse >1%. TEC.TO pourrait subir une pression à la baisse.",
                        allow_during_blackout=True,
                        category='portfolio_guard',
                    )
                    alerts_sent += 1

        result = {'status': 'ok', 'alerts': alerts_sent}
        _task_log_finish(log, 'SUCCESS', result)
        return result
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


@shared_task
def portfolio_news_brief() -> dict[str, Any]:
    log = _task_log_start('portfolio_news_brief')
    try:
        core_symbols, moon_symbols = _portfolio_symbols_from_env()
        symbols = list(dict.fromkeys(core_symbols + moon_symbols))
        if not symbols:
            _task_log_finish(log, 'SUCCESS', {'status': 'empty'})
            return {'status': 'empty'}
        lines = ["📰 *News Portfolio*", ""]
        for symbol in symbols:
            titles = _google_news_titles(symbol, days=1, limit=2)
            if not titles:
                continue
            lines.append(f"${symbol}:")
            for title in titles:
                lines.append(f"• {title}")
        message = "\n".join(lines).strip()
        if message:
            _send_telegram_alert(message, allow_during_blackout=True, category='portfolio_guard')
        _task_log_finish(log, 'SUCCESS', {'status': 'sent'})
        return {'status': 'sent'}
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


@shared_task
def generate_penny_signals(days: int = 7) -> dict[str, int]:
    """Generate penny stock signals from Reddit sentiment + pattern matching."""
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT', 'pennystock-bot')

    has_reddit = bool(client_id and client_secret)

    subreddits = os.getenv('PENNY_SUBREDDITS', 'pennystocks+wallstreetbets').split('+')
    max_symbols = int(os.getenv('PENNY_MAX_SYMBOLS', '20'))

    analyzer = SentimentIntensityAnalyzer()
    mention_map: dict[str, dict] = {}

    if has_reddit:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )

        for sub in subreddits:
            for post in reddit.subreddit(sub).hot(limit=200):
                text = f"{post.title} {post.selftext}".upper()
                tickers = set(re.findall(r"\b\$?[A-Z]{2,5}\b", text))
                for t in tickers:
                    symbol = t.replace('$', '')
                    entry = mention_map.setdefault(symbol, {
                        'mentions': 0,
                        'sentiment': 0.0,
                        'hype': 0.0,
                    })
                    entry['mentions'] += 1
                    entry['sentiment'] += analyzer.polarity_scores(text).get('compound', 0)
                    entry['hype'] += (post.score or 0) + (post.num_comments or 0)
    else:
        cutoff = timezone.now() - timedelta(days=days)
        for stock in Stock.objects.all().order_by('symbol'):
            qs = StockNews.objects.filter(stock=stock, fetched_at__gte=cutoff)
            count = qs.count()
            if count == 0:
                continue
            avg_sentiment = (
                qs.aggregate(avg=models.Avg('sentiment')).get('avg')
            ) or 0
            mention_map[stock.symbol] = {
                'mentions': count,
                'sentiment': avg_sentiment * count,
                'hype': count * 25,
            }

    # Rank by mentions + hype
    ranked = sorted(
        mention_map.items(),
        key=lambda x: (x[1]['mentions'], x[1]['hype']),
        reverse=True,
    )[:max_symbols]

    created = 0
    seen = 0
    today = timezone.now().date()

    for symbol, stats in ranked:
        seen += 1
        # Fetch recent prices
        data = yf.Ticker(symbol).history(period=f"{days}d")
        if data is None or data.empty or 'Close' not in data or 'Volume' not in data:
            continue

        last_price = float(data['Close'].iloc[-1])
        if last_price > 0.5:
            continue  # not a sub-$0.50 penny stock

        avg_volume = float(data['Volume'].rolling(20).mean().iloc[-1]) if len(data) >= 20 else float(data['Volume'].mean())
        last_volume = float(data['Volume'].iloc[-1])
        prev_close = float(data['Close'].iloc[-2]) if len(data) > 1 else last_price
        gap = (last_price - prev_close) / prev_close if prev_close else 0

        volume_spike = (last_volume / avg_volume) if avg_volume else 0
        pattern_score = max(0.0, min(1.0, (gap * 5) + (volume_spike / 10)))

        sentiment_score = stats['sentiment'] / max(stats['mentions'], 1)
        hype_score = min(1.0, stats['hype'] / 10000)
        liquidity_score = min(1.0, avg_volume / 1_000_000)
        combined = (pattern_score * 0.5) + (sentiment_score * 0.3) + (hype_score * 0.2)

        _, was_created = PennySignal.objects.update_or_create(
            symbol=symbol,
            as_of=today,
            defaults={
                'pattern_score': pattern_score,
                'sentiment_score': sentiment_score,
                'hype_score': hype_score,
                'liquidity_score': liquidity_score,
                'combined_score': combined,
                'last_price': last_price,
                'avg_volume': avg_volume,
                'mentions': stats['mentions'],
                'data': {
                    'gap': gap,
                    'volume_spike': volume_spike,
                },
            },
        )

        if was_created:
            created += 1

    return {'created': created, 'seen': seen}


@shared_task
def scan_long_goal_candidates(limit: int = 15) -> dict[str, Any]:
    """Scan bluechips for long-goal discounts (weekly RSI < 30 + earnings growth + below MA200)."""
    log = _task_log_start('scan_long_goal_candidates')
    symbols = [
        s.strip().upper()
        for s in os.getenv('LONG_GOAL_SYMBOLS', 'AAPL,AMZN,MSFT,NVDA,TD.TO,CNQ.TO,RY.TO,ENB.TO').split(',')
        if s.strip()
    ]
    results: list[dict[str, Any]] = []
    for symbol in symbols:
        try:
            weekly = yf.Ticker(symbol).history(period='2y', interval='1wk', timeout=10)
            if weekly is None or weekly.empty or 'Close' not in weekly:
                continue
            close = weekly['Close']
            delta = close.diff().fillna(0.0)
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, pd.NA)
            rsi = 100 - (100 / (1 + rs))
            rsi_val = float(rsi.iloc[-1]) if len(rsi) else 0.0
            if rsi_val >= 30:
                continue

            daily = yf.Ticker(symbol).history(period='1y', interval='1d', timeout=10)
            if daily is None or daily.empty or 'Close' not in daily:
                continue
            ma200 = daily['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
            last = float(daily['Close'].iloc[-1])
            if not (last < ma200):
                continue

            info = yf.Ticker(symbol).info or {}
            earnings_growth = info.get('earningsGrowth')
            if earnings_growth is None or float(earnings_growth) <= 0:
                continue

            results.append({
                'symbol': symbol,
                'weekly_rsi': round(rsi_val, 2),
                'price': round(last, 2),
                'ma200': round(float(ma200), 2),
                'earnings_growth': float(earnings_growth),
            })
        except Exception:
            continue
    results = sorted(results, key=lambda x: x['weekly_rsi'])[:limit]
    if results and os.getenv('LONG_GOAL_NOTIFY', 'false').lower() in {'1', 'true', 'yes', 'y'}:
        lines = ["🏁 Long Goal Candidates", ""]
        for row in results:
            lines.append(
                f"{row['symbol']}: RSIw {row['weekly_rsi']}, Price {row['price']} < MA200 {row['ma200']}"
            )
        _send_telegram_alert("\n".join(lines), allow_during_blackout=True, category='report')
    payload = {'status': 'ok', 'count': len(results), 'results': results}
    _task_log_finish(log, 'SUCCESS', payload)
    return payload


@shared_task
def run_morning_ai_analysis() -> dict[str, Any]:
    log = _task_log_start('run_morning_ai_analysis')
    try:
        def _parse_symbols(env_key: str) -> list[str]:
            raw = os.getenv(env_key, '')
            return [s.strip().upper() for s in raw.split(',') if s.strip()]

        growth = _parse_symbols('OPTIMIZER_GROWTH_SYMBOLS')
        spec = _parse_symbols('OPTIMIZER_SPEC_SYMBOLS')
        blue = _parse_symbols('OPTIMIZER_BLUECHIP_SYMBOLS')
        watchlist = list(dict.fromkeys(growth + spec + blue))
        if not watchlist:
            _task_log_finish(log, 'SUCCESS', {'status': 'empty'})
            return {'status': 'empty'}

        advisor = DeepSeekAdvisor()
        router = DanasMLRouter()
        alerts = []
        for symbol in watchlist:
            context = get_intraday_context(symbol)
            if not context:
                continue
            bars = context.get('bars')
            if bars is None or getattr(bars, 'empty', True):
                continue
            try:
                price = float(bars.iloc[-1]['close'])
            except Exception:
                price = None
            try:
                rvol = float(context.get('rvol') or 0)
            except Exception:
                rvol = 0.0
            try:
                day_change = float(context.get('day_change_pct') or 0)
            except Exception:
                day_change = 0.0

            is_penny = price is not None and price < 7
            passes = False
            if is_penny:
                passes = rvol >= 2.0 and day_change <= -7.0
            else:
                passes = rvol >= 1.2 and day_change <= -3.0
            if not passes:
                continue

            prompt = "Analyse l'ouverture 9h45 et propose un plan d'action complet."
            chunks = []
            for item in advisor.stream_answer(symbol, prompt):
                try:
                    payload = json.loads(item)
                except Exception:
                    payload = {'text': str(item)}
                text = payload.get('text') or ''
                chunks.append(text)
                if payload.get('done'):
                    break

            ai_text = "".join(chunks).strip()
            ml_conf = router.predict(symbol).get('confidence')
            if should_send_to_telegram(ml_conf, ai_text):
                message = (
                    f"🎯 *SIGNAL VALIDÉ : {symbol}*\n"
                    f"💰 *Prix:* {price} | *Drop:* {day_change:.2f}%\n"
                    f"📊 *RVOL:* {rvol:.2f}\n\n"
                    f"🧠 *ANALYSE DANAS :*\n{ai_text[:400]}...\n\n"
                    f"🔗 [Wealthsimple](https://my.wealthsimple.com/app/invest/search?query={symbol})"
                )
                _send_telegram_alert(message, allow_during_blackout=True, category='report')
                alerts.append(symbol)
            else:
                _queue_telegram_candidate({
                    'ticker': symbol,
                    'score': ml_conf or 0,
                    'diagnostic': ai_text[:200],
                    'ts': timezone.now().isoformat(),
                })
            sleep(5)

        result = {'status': 'ok', 'alerts': alerts}
        _task_log_finish(log, 'SUCCESS', result)
        return result
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


@shared_task
def cache_optimizer_snapshot(fast: bool = True, portfolio_id: int | None = None) -> dict[str, Any]:
    log = _task_log_start('cache_optimizer_snapshot')
    try:
        from types import SimpleNamespace
        from portfolio.views import PortfolioOptimizerView

        params: dict[str, str] = {'fast': '1' if fast else '0'}
        if portfolio_id:
            params['portfolio_id'] = str(portfolio_id)
        request = SimpleNamespace(query_params=params, user=None)

        response = PortfolioOptimizerView().get(request)
        payload = getattr(response, 'data', None)
        if isinstance(payload, dict):
            cache_key = f"optimizer:scheduled:{portfolio_id or 'default'}:{'fast' if fast else 'slow'}"
            ttl_min = int(os.getenv('OPTIMIZER_SCHEDULE_TTL_MIN', '360'))
            cache.set(cache_key, payload, timeout=max(60, ttl_min * 60))
        _task_log_finish(log, 'SUCCESS', {'status': 'ok'})
        return {'status': 'ok'}
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}


@shared_task
def cache_ai_center_snapshot() -> dict[str, Any]:
    log = _task_log_start('cache_ai_center_snapshot')
    try:
        from types import SimpleNamespace
        from portfolio.views import AICenterView

        request = SimpleNamespace(query_params={}, user=None)
        response = AICenterView().get(request)
        payload = getattr(response, 'data', None)
        if isinstance(payload, dict):
            ttl_min = int(os.getenv('AI_CENTER_SCHEDULE_TTL_MIN', '360'))
            cache.set('ai_center:scheduled', payload, timeout=max(60, ttl_min * 60))
        _task_log_finish(log, 'SUCCESS', {'status': 'ok'})
        return {'status': 'ok'}
    except Exception as exc:
        _task_log_finish(log, 'FAILED', error=str(exc))
        return {'status': 'failed', 'error': str(exc)}
