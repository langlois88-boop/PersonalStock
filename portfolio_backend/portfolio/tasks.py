from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from datetime import date, datetime, timedelta
from math import isfinite
from typing import Any

import yfinance as yf
import requests
import random
import json
import pandas as pd
from celery import shared_task
from django.conf import settings
from django.core.mail import send_mail
from django.core.management import call_command
from django.db import models
from django.utils import timezone
import finnhub
import feedparser
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
import re
import praw
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
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
    try:
        return float(getattr(settings, 'USD_CAD_RATE', os.getenv('USD_CAD_RATE', '1.36')))
    except (TypeError, ValueError):
        return 1.36


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


@shared_task
def fetch_prices_hourly() -> dict[str, float]:
    log = _task_log_start('fetch_prices_hourly')
    prices: dict[str, float] = {}
    try:
        for stock in Stock.objects.all().order_by('symbol'):
            if not _is_valid_symbol(stock.symbol):
                continue
            ticker = yf.Ticker(stock.symbol)
            try:
                data = ticker.history(period='1mo', interval='1d', timeout=10)
            except Exception:
                data = None
            if data is None or data.empty or 'Close' not in data:
                if _backfill_latest_price(stock):
                    prices[stock.symbol] = float(stock.latest_price or 0)
                continue
            last_row = data.iloc[-1]
            price = float(last_row['Close'])
            if not _is_valid_price(price):
                if _backfill_latest_price(stock):
                    prices[stock.symbol] = float(stock.latest_price or 0)
                continue
            day_low = float(last_row['Low']) if 'Low' in data else None
            day_high = float(last_row['High']) if 'High' in data else None

            info: dict[str, Any] = {}
            try:
                info = ticker.info or {}
            except Exception:
                info = {}

            sector = (info.get('sector') or '').strip()
            div_yield = info.get('dividendYield')
            if div_yield is None:
                div_yield = info.get('trailingAnnualDividendYield')
            div_yield = float(div_yield) if div_yield is not None else None
            if stock.symbol.upper() == 'AVGO':
                dividend_rate = info.get('dividendRate') or info.get('trailingAnnualDividendRate')
                if dividend_rate is not None:
                    try:
                        div_yield = float(dividend_rate) / 332.54
                    except Exception:
                        pass
            if stock.symbol.upper() == 'TEC.TO':
                dividend_rate = info.get('dividendRate') or info.get('trailingAnnualDividendRate')
                if dividend_rate is not None and price:
                    try:
                        div_yield = float(dividend_rate) / float(price)
                    except Exception:
                        pass
                if div_yield is not None:
                    div_yield = min(float(div_yield), 0.02)

            price = _to_cad_price(stock.symbol, price, info)
            day_low = _to_cad_price(stock.symbol, day_low, info)
            day_high = _to_cad_price(stock.symbol, day_high, info)

            stock.latest_price = price
            stock.day_low = day_low
            stock.day_high = day_high
            if (not stock.sector) or stock.sector.lower() == 'unknown':
                if sector:
                    stock.sector = sector
            if stock.symbol.upper() in {'AVGO', 'TEC.TO'} and div_yield is not None:
                stock.dividend_yield = div_yield
            elif not stock.dividend_yield or float(stock.dividend_yield or 0) == 0:
                if div_yield is not None:
                    stock.dividend_yield = div_yield
            stock.latest_price_updated_at = timezone.now()
            stock.save(update_fields=['latest_price', 'day_low', 'day_high', 'sector', 'dividend_yield', 'latest_price_updated_at'])

            for dt, row in data.iterrows():
                close_price = float(row['Close'])
                close_price = _to_cad_price(stock.symbol, close_price, info)
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
        _task_log_finish(log, 'FAILED', error=str(exc))
        _send_alert('Task failed: fetch_finnhub_news_daily', str(exc))
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
                f"?q={query}&hl=en-US&gl=US&ceid=US:en"
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
                f"?q={query}&hl=en-US&gl=US&ceid=US:en"
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
    lookback_days = int(os.getenv('BACKTEST_LOOKBACK_DAYS', '90'))
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
        if isinstance(value, (list, tuple, pd.Series)) and value:
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


def _model_signal(symbol: str, universe: str, model_path: str | Path) -> dict[str, Any] | None:
    try:
        fusion = DataFusionEngine(symbol)
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
        try:
            hist = yf.Ticker(symbol).history(period='5d', interval='1d', timeout=10)
            if hist is not None and not hist.empty and 'Close' in hist:
                return float(hist['Close'].iloc[-1])
        except Exception:
            return None
        return None

    def _atr(symbol: str) -> float:
        try:
            hist = yf.Ticker(symbol).history(period='20d', interval='1d', timeout=10)
            if hist is None or hist.empty or not {'High', 'Low', 'Close'}.issubset(hist.columns):
                return 0.0
            tr = pd.concat([
                (hist['High'] - hist['Low']).abs(),
                (hist['High'] - hist['Close'].shift(1)).abs(),
                (hist['Low'] - hist['Close'].shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            return float(atr) if atr is not None else 0.0
        except Exception:
            return 0.0

    def _volume_z_negative_streak(symbol: str, days: int) -> tuple[bool, float | None]:
        if days <= 0:
            return False, None
        try:
            fusion = DataFusionEngine(symbol)
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
        return _model_signal(symbol, universe, model_path)

    created = 0
    closed = 0

    for symbol, trades in open_trades_by_symbol.items():
        price = _latest_price(symbol)
        if price is None:
            continue
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
        risk_budget = capital * risk_pct
        position_cap = capital * position_cap_pct
        position_value = min(available, position_cap, risk_budget / (stop_distance / price)) if price else 0.0
        quantity = int(position_value / price) if price else 0
        if quantity <= 0:
            continue
        stop_loss = price - stop_distance
        PaperTrade.objects.create(
            ticker=symbol,
            sandbox=sandbox,
            entry_price=round(price, 2),
            quantity=quantity,
            entry_signal=signal,
            entry_features=(signal_payload or {}).get('features'),
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
def send_morning_scout_report() -> dict[str, Any]:
    log = _task_log_start('send_morning_scout_report')
    try:
        email_to = settings.ALERT_EMAIL_TO
        if not email_to:
            result = {'status': 'skipped', 'reason': 'no alert email configured'}
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

        lines: list[str] = []
        today = timezone.now().date()
        lines.append(f"Rapport du matin - {today}")
        if market_stress:
            lines.append(
                f"⚠️ Market Stress High - Trading Restricted (VIX {float(vix_level or 0):.2f} >= {vix_threshold:.2f})"
            )
        lines.append('')

        payload_summary: dict[str, Any] = {}

        for sandbox, prefix, universe, default_watchlist in sandboxes:
            watchlist = _get_watchlist(sandbox, prefix, default_watchlist)
            model_path = get_model_path(universe)
            buy_threshold = _env_float(prefix, 'BUY_THRESHOLD', '0.82')
            min_volume_z = _env_float(prefix, 'VOLUME_ZSCORE_MIN', '0.5')

            validated: list[dict[str, Any]] = []
            excluded: list[tuple[str, str]] = []

            for symbol in watchlist:
                if not _is_valid_symbol(symbol):
                    continue
                signal_payload = _model_signal(symbol, universe, model_path)
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

            lines.append(f"=== {sandbox} ===")
            lines.append('Validées:')
            if validated:
                for entry in validated:
                    extra = f" | {entry['reason']}" if entry['reason'] else ''
                    vol_txt = (
                        f" | VolumeZ {entry['volume_z']:.2f}" if entry['volume_z'] is not None else ''
                    )
                    lines.append(
                        f"- {entry['symbol']} | signal {entry['signal']:.2f}{vol_txt}{extra}"
                    )
            else:
                lines.append('- Aucune')

            lines.append('Exclues aujourd’hui:')
            if excluded:
                for symbol, reason in excluded:
                    lines.append(f"- {symbol}: {reason}")
            else:
                lines.append('- Aucune')

            lines.append('')

            payload_summary[sandbox] = {
                'validated': len(validated),
                'excluded': len(excluded),
                'watchlist': len(watchlist),
            }

        # Portfolio details
        portfolios = Portfolio.objects.all()
        if portfolios.exists():
            lines.append('=== Portefeuilles ===')
            for portfolio in portfolios:
                lines.append(f"{portfolio.name}:")
                holdings = (
                    PortfolioHolding.objects.select_related('stock')
                    .filter(portfolio=portfolio)
                    .order_by('stock__symbol')
                )
                if not holdings:
                    lines.append('- Aucun titre')
                    lines.append('')
                    continue

                table_rows: list[list[str]] = []
                news_map: dict[str, list[str]] = {}
                for holding in holdings:
                    stock = holding.stock
                    shares = float(holding.shares or 0)
                    if shares <= 0:
                        continue

                    latest_price = (
                        float(stock.latest_price)
                        if stock.latest_price is not None
                        else None
                    )
                    avg_cost = _avg_cost_for_stock(portfolio, stock)
                    value_now = latest_price * shares if latest_price is not None else None
                    cost_now = avg_cost * shares if avg_cost is not None else None
                    pnl = (value_now - cost_now) if value_now is not None and cost_now is not None else None

                    prediction = (
                        Prediction.objects.filter(stock=stock)
                        .order_by('-date')
                        .first()
                    )
                    projection = (
                        _fmt_money(prediction.predicted_price)
                        if prediction else 'n/a'
                    )
                    reco = prediction.recommendation if prediction else 'n/a'

                    open_price = 'n/a'
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
                        .order_by('-published_at')[:2]
                    )
                    if news_items:
                        news_map[stock.symbol] = [
                            f"{item.headline} ({item.url})" for item in news_items
                        ]

                headers = ['Symbole', 'Qté', 'Prix', 'Achat', 'Valeur', 'PnL', 'Jour', 'Projection', 'Reco']
                lines.extend(_format_table(headers, table_rows))
                if news_map:
                    for symbol, headlines in news_map.items():
                        for headline in headlines:
                            lines.append(f"  • {symbol}: {headline}")
                lines.append('')

        # Account details
        accounts = Account.objects.all().order_by('name')
        if accounts.exists():
            lines.append('=== Comptes ===')
            for account in accounts:
                lines.append(f"{account.name} ({account.account_type}):")
                positions = _positions_from_account(account)
                if not positions:
                    lines.append('- Aucun titre')
                    lines.append('')
                    continue

                table_rows = []
                news_map: dict[str, list[str]] = {}
                for pos in positions:
                    stock = pos['stock']
                    shares = float(pos['shares'] or 0)
                    avg_cost = pos.get('avg_cost')
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
                    projection = (
                        _fmt_money(prediction.predicted_price)
                        if prediction else 'n/a'
                    )
                    reco = prediction.recommendation if prediction else 'n/a'

                    open_price = 'n/a'
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
                        .order_by('-published_at')[:2]
                    )
                    if news_items:
                        news_map[stock.symbol] = [
                            f"{item.headline} ({item.url})" for item in news_items
                        ]

                headers = ['Symbole', 'Qté', 'Prix', 'Achat', 'Valeur', 'PnL', 'Jour', 'Projection', 'Reco']
                lines.extend(_format_table(headers, table_rows))
                if news_map:
                    for symbol, headlines in news_map.items():
                        for headline in headlines:
                            lines.append(f"  • {symbol}: {headline}")
                lines.append('')

        lines.append('Notes: simulation uniquement. Les recommandations sont indicatives.')

        subject = f"Daily AI Scout Report - {today}"
        send_mail(
            subject=subject,
            message='\n'.join(lines).strip(),
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
        lookback_days = int(os.getenv('BACKTEST_LOOKBACK_DAYS', '90'))
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
