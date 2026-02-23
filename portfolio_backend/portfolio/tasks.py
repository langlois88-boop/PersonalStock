from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from datetime import date, datetime, timedelta, time as dt_time
from math import isfinite
from typing import Any

from . import market_data as yf
import requests
import random
import json
import pandas as pd
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
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai
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
    Stock,
    StockNews,
    TaskRunLog,
    UserPreference,
)
from .ai_module import run_predictions
from .alpaca_data import (
    get_daily_bars,
    get_intraday_context,
    get_intraday_bars,
    get_latest_trade_price,
    get_stock_snapshots,
    get_trading_client,
    get_tradable_symbols,
)
from .patterns import enrich_bars_with_patterns
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


def _ny_time_now() -> datetime:
    try:
        return timezone.now().astimezone(ZoneInfo('America/New_York'))
    except Exception:
        return timezone.now()


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
    force_list = {'RY'}
    force_list.update({
        s.strip().upper()
        for s in str(os.getenv('FORCE_CAD_TICKERS', '')).split(',')
        if s.strip()
    })
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
    updated = 0

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
                url = a.get('url')
                if not url:
                    continue

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

                _, was_created = StockNews.objects.get_or_create(
                    url=url,
                    defaults={
                        'stock': stock,
                        'headline': headline,
                        'source': ((a.get('source') or {}).get('name') or '').strip()[:100],
                        'published_at': published_at,
                        'sentiment': sentiment,
                        'raw': a,
                    },
                )
                if was_created:
                    created += 1
                else:
                    updated += 1

        _task_log_finish(log, 'SUCCESS', {'created': created, 'updated': updated})
        return {'created': created, 'updated': updated}
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
                url = item.get('url')
                if not url:
                    continue

                headline = (item.get('headline') or '').strip()[:300] or url
                summary = (item.get('summary') or '').strip()
                text_for_sentiment = f"{headline}. {summary}".strip()
                sentiment = analyzer.polarity_scores(text_for_sentiment).get('compound')

                published_at = None
                if item.get('datetime'):
                    published_at = datetime.fromtimestamp(item['datetime'], tz=timezone.UTC)

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

        _task_log_finish(log, 'SUCCESS', {'created': created, 'seen': seen})
        return {'created': created, 'seen': seen}
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
                link = entry.get('link')
                if not link:
                    continue

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

        _task_log_finish(log, 'SUCCESS', {'created': created, 'seen': seen})
        return {'created': created, 'seen': seen}
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
                link = entry.get('link')
                if not link:
                    continue

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


def _get_vix_level() -> float | None:
    try:
        hist = yf.Ticker('^VIX').history(period='5d', interval='1d', timeout=10)
        if hist is not None and not hist.empty and 'Close' in hist:
            return _safe_float(hist['Close'].iloc[-1])
    except Exception:
        return None
    return None


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

    if sandbox == 'AI_BLUECHIP':
        buy_threshold = 0.85
        sell_threshold = 0.65
        trail_pct = 0.05
        min_volume_z = max(min_volume_z, 0.5)

    capital = initial_capital + closed_pnl
    available = max(0.0, capital - open_value)

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
            return float(price) if price else None
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

    def _signal(symbol: str) -> dict[str, Any] | None:
        return _model_signal(symbol, universe, model_path, use_alpaca=use_alpaca)

    created = 0
    closed = 0

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
        signal_payload = _signal(symbol)
        signal = signal_payload['signal'] if signal_payload else None
        for trade in trades:
            entry_price = float(trade.entry_price or 0)
            profit_pct = (price - entry_price) / entry_price if entry_price else 0.0
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
            break_even_stop = entry_price if entry_price and price >= (entry_price * (1 + break_even_pct)) else 0.0
            atr = _atr(symbol)
            dynamic_atr_stop = 0.0
            if profit_pct >= trail_profit_trigger and atr > 0:
                dynamic_atr_stop = price - (trail_atr_mult * atr)
            new_stop = max(
                float(trade.stop_loss),
                price * (1 - trail_pct),
                break_even_stop,
                dynamic_atr_stop,
            )
            if intraday_ctx:
                pattern_signal = float(intraday_ctx.get('pattern_signal') or 0)
                rvol = float(intraday_ctx.get('rvol') or 0)
                if pattern_signal < 0 and rvol >= 2:
                    new_stop = max(new_stop, price * (1 - (trail_pct * 0.5)))
                    trade.notes = (trade.notes or '') + ' | Pattern baissier + RVOL élevé: stop resserré.'
            trade.stop_loss = new_stop
            stop_hit = price <= new_stop
            signal_exit = signal is not None and signal < sell_threshold
            should_close = stop_hit or signal_exit
            if should_close:
                trade.status = 'CLOSED'
                trade.exit_price = price
                trade.exit_date = timezone.now()
                trade.pnl = float(price - float(trade.entry_price)) * float(trade.quantity)
                trade.outcome = 'WIN' if float(trade.pnl or 0) > 0 else 'LOSS'
                volume_z = _safe_float((signal_payload or {}).get('features', {}).get('VolumeZ'))
                exit_reason = 'Stop Loss' if stop_hit else 'Signal IA'
                volume_note = f" VolumeZ {volume_z:.2f}." if volume_z is not None else ''
                trade.notes = (
                    f"Trade fermé à cause de {exit_reason}. Cause probable : Volume trop faible lors de l'entrée." + volume_note
                )
                closed += 1
            trade.save(update_fields=['stop_loss', 'status', 'exit_price', 'exit_date', 'pnl', 'outcome', 'notes'])

    for symbol in watchlist:
        existing_trades = open_trades_by_symbol.get(symbol, [])
        signal_payload = _signal(symbol)
        signal = signal_payload['signal'] if signal_payload else None
        intraday_ctx = None
        if use_alpaca:
            intraday_ctx = get_intraday_context(
                symbol,
                minutes=int(os.getenv('ALPACA_INTRADAY_MINUTES', '390')),
                rvol_window=int(os.getenv('ALPACA_RVOL_WINDOW', '20')),
            )
            if intraday_ctx:
                pattern_signal = float(intraday_ctx.get('pattern_signal') or 0)
                rvol = float(intraday_ctx.get('rvol') or 0)
                if pattern_signal < 0 and rvol >= 2:
                    continue
                if signal is not None and pattern_signal > 0 and rvol >= 2:
                    signal = min(1.0, float(signal) * 1.02)
        if signal is not None and sandbox == 'AI_BLUECHIP':
            multiplier = _bluechip_aggressive_multiplier()
            if multiplier > 1.0:
                signal = min(1.0, float(signal) * multiplier)
        if signal is None or signal < buy_threshold:
            continue
        if existing_trades and signal < reinforce_min_score:
            continue
        if sandbox == 'AI_PENNY' and _penny_blocked():
            continue
        volume_z = _safe_float((signal_payload or {}).get('features', {}).get('VolumeZ'))
        if volume_z is not None and volume_z < min_volume_z:
            ai_score = float(signal or 0) * 100
            message = (
                f"Non-Trade [{sandbox}]: {symbol} signal {ai_score:.2f}% "
                f"volume_z {float(volume_z):.2f} < {min_volume_z:.2f}"
            )
            stock = Stock.objects.filter(symbol__iexact=symbol).first()
            AlertEvent.objects.create(category='PAPER_NON_TRADE', message=message, stock=stock)
            continue
        if sandbox == 'AI_PENNY':
            altman_z = _safe_float(
                (signal_payload or {}).get('features', {}).get('AltmanZ')
                or (signal_payload or {}).get('features', {}).get('AltmanZScore')
            )
            if altman_z is None or altman_z <= min_altman_z:
                continue
        price = _latest_price(symbol)
        if price is None:
            continue
        atr = _atr(symbol)
        stop_distance = max(price * trail_pct, atr_mult * atr, price * 0.01)
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
            continue
        position_value = min(available, position_cap, risk_budget / (stop_distance / price)) if price else 0.0
        position_value *= confidence_factor
        quantity = int(position_value / price) if price else 0
        if quantity <= 0:
            continue
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
            })
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
            stop_loss=round(stop_loss, 2),
            status='OPEN',
            pnl=0,
            notes=f"Signal {signal:.2f} / ATR {atr:.2f}",
        )
        created += 1
        available = max(0.0, available - (quantity * price))

    return {'created': created, 'closed': closed, 'available': round(available, 2)}


@shared_task
def execute_paper_trades() -> dict[str, Any]:
    return _execute_paper_trades_for_sandbox('WATCHLIST', 'PAPER')


@shared_task
def execute_paper_trades_ai_bluechip() -> dict[str, Any]:
    return _execute_paper_trades_for_sandbox('AI_BLUECHIP', 'AI_BLUECHIP')


@shared_task
def execute_paper_trades_ai_penny() -> dict[str, Any]:
    return _execute_paper_trades_for_sandbox('AI_PENNY', 'AI_PENNY')


@shared_task
def market_scanner_task() -> dict[str, Any]:
    """Scan market for high-momentum candidates and cache results."""
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
    if symbols_env:
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
                _send_telegram_alert(
                    f"⚠️ ALERTE DE CHUTE : ${symbol} RVOL {rvol:.2f} mais pattern négatif.",
                    category='alert',
                )
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

    if notify and results:
        top = results[0]
        entry_price = float(top['price'])
        buy_limit = entry_price * (1 + buy_limit_buffer)
        target_pct = swing_target_pct
        if rvol > 3:
            target_pct = max(target_pct, swing_min_rvol_target_pct)
        target_price = buy_limit * (1 + target_pct)
        stop_price = buy_limit * (1 - swing_stop_pct)
        title = (top.get('news_titles') or [''])[0] or None
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
        _send_telegram_alert(message, category='signal')
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

    return {'status': 'ok', 'count': len(results)}


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
        if not trades.exists():
            return {'sandbox': selected_sandbox, 'status': 'no_trades', 'samples': 0}

        samples: list[dict[str, Any]] = []
        for trade in trades:
            symbol = (trade.ticker or '').strip().upper()
            if not symbol:
                continue
            entry_date = trade.entry_date or trade.exit_date
            if not entry_date:
                continue
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

        payload = train_fusion_model_from_labels(samples_df.drop(columns=['entry_date']), model_path=model_path, save=False)
        if not payload:
            return {'sandbox': selected_sandbox, 'status': 'failed', 'samples': int(len(train_df)), 'trained': False}

        model_version = get_model_version(payload, model_path)

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
                        'holdout_ok': holdout_ok,
                        'holdout_days': holdout_days,
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
            _send_telegram_alert(
                f"✅ ORDRE PROBABLEMENT FILLED : {symbol} à {latest_price:.2f} {currency} (limite {limit_price:.2f}).",
                allow_during_blackout=True,
                category='signal',
            )
            filled = True

        stop_price = limit_price * (1 - stop_pct)
        stop_key = _cache_key(symbol, 'stop')
        if filled and latest_price <= stop_price and not cache.get(stop_key, False):
            cache.set(stop_key, True, timeout=60 * 60 * 8)
            _send_telegram_alert(
                f"⛔ STOP-LOSS ATTEINT : {symbol} à {latest_price:.2f} {currency} (stop {stop_price:.2f}).",
                allow_during_blackout=True,
                category='signal',
            )

        target1 = limit_price * (1 + target1_pct)
        target2 = limit_price * (1 + target2_pct)
        target1_key = _cache_key(symbol, 'target10')
        target2_key = _cache_key(symbol, 'target15')
        if filled and latest_price >= target1 and not cache.get(target1_key, False):
            cache.set(target1_key, True, timeout=60 * 60 * 8)
            _send_telegram_alert(
                f"✅ PALIER 10% ATTEINT : {symbol} à {latest_price:.2f} {currency} (cible {target1:.2f}).",
                allow_during_blackout=True,
                category='signal',
            )
        if filled and latest_price >= target2 and not cache.get(target2_key, False):
            cache.set(target2_key, True, timeout=60 * 60 * 8)
            _send_telegram_alert(
                f"🔥 CIBLE 15% ATTEINTE : {symbol} à {latest_price:.2f} {currency} (cible {target2:.2f}).",
                allow_during_blackout=True,
                category='signal',
            )

        intraday = yf.Ticker(symbol).history(period='5d', interval='15m')
        close_15m = _extract_close_series(intraday)
        rsi_15m = _compute_rsi(close_15m, 14) if close_15m is not None and not close_15m.empty else None
        rsi_key = _cache_key(symbol, 'rsi')
        if filled and rsi_15m is not None and rsi_15m >= rsi_sell and not cache.get(rsi_key, False):
            cache.set(rsi_key, True, timeout=60 * 60 * 8)
            _send_telegram_alert(
                f"⚠️ RSI CHAUD : {symbol} RSI15m {rsi_15m:.1f} ≥ {rsi_sell:.0f}. Recommandation : vendre.",
                allow_during_blackout=True,
                category='signal',
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
                _send_telegram_alert(message, allow_during_blackout=True, category='signal')

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


def _pct_change_from_open(symbol: str) -> float | None:
    try:
        hist = yf.download(tickers=symbol, period='1d', interval='1m')
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
        hist = yf.download(tickers=symbol, period='1d', interval='1m')
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
        hist = yf.download(tickers=symbol, period='1d', interval='1m')
    except Exception:
        hist = None
    if hist is None or hist.empty or 'Volume' not in hist.columns:
        return None
    return float(hist['Volume'].sum())


def _daily_history(symbol: str, days: int = 365) -> pd.DataFrame:
    try:
        hist = yf.download(tickers=symbol, period=f'{days}d', interval='1d')
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
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        ohlc_text = 'n/a'
        if df_ohlc is not None and not df_ohlc.empty:
            ohlc_text = df_ohlc.tail(10).to_string(index=False)
        prompt = (
            "Tu es un expert en Day Trading de Penny Stocks. Analyse les données suivantes pour "
            f"{ticker}. Données Techniques (5min) : {ohlc_text}. Sentiment des News : {news_sentiment}. "
            f"Corrélation avec actif parent : {correlation_data}. Donne un verdict clair : ACHAT, ATTENTE ou DANGER. "
            "Inclus un prix d'entrée, un objectif de profit (+10%) et un stop-loss (-4%). Sois concis pour un message Telegram."
        )
        response = model.generate_content(prompt)
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
        if not api_key:
            payload = {'status': 'no_api_key', 'count': len(rows)}
            _task_log_finish(log, 'SUCCESS', payload)
            return payload

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (
            f"Voici mon portfolio actuel : {rows}. En tant qu'expert en gestion de risque, analyse chaque ligne. "
            "Pour chacune, réponds brièvement : 1. Statut (HOLD/SELL/TAKE PROFIT), 2. Raison rapide (ex: News négative, "
            "Bitcoin chute, ou Momentum haussier), et 3. Urgence (Bas/Moyen/Haut). Termine par un conseil sur mon exposition totale."
        )
        try:
            response = model.generate_content(prompt)
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
