from __future__ import annotations

import os
from functools import lru_cache
from datetime import datetime, timedelta
from typing import Any

import yfinance as yf
import requests
import random
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

from .models import (
    Account,
    AccountTransaction,
    AlertEvent,
    DividendHistory,
    DripSnapshot,
    Portfolio,
    PortfolioDigest,
    PennyStockSnapshot,
    PennyStockUniverse,
    PennySignal,
    PriceHistory,
    Prediction,
    PaperTrade,
    Stock,
    StockNews,
    UserPreference,
)
from .ai_module import run_predictions
from .ml_engine.engine.data_fusion import DataFusionEngine
from .ml_engine.backtester import (
    AIBacktester,
    FEATURE_COLUMNS,
    apply_feature_weighting_to_signal,
    load_or_train_model,
    save_model_payload,
    train_fusion_model,
    train_fusion_model_from_labels,
)


DEFAULT_YIELD = 0.02


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
    prices: dict[str, float] = {}
    for stock in Stock.objects.all().order_by('symbol'):
        if not _is_valid_symbol(stock.symbol):
            continue
        ticker = yf.Ticker(stock.symbol)
        try:
            data = ticker.history(period='1mo', interval='1d', timeout=10)
        except Exception:
            continue
        if data is None or data.empty or 'Close' not in data:
            continue
        last_row = data.iloc[-1]
        price = float(last_row['Close'])
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

        stock.latest_price = price
        stock.day_low = day_low
        stock.day_high = day_high
        if (not stock.sector) or stock.sector.lower() == 'unknown':
            if sector:
                stock.sector = sector
        if not stock.dividend_yield or float(stock.dividend_yield or 0) == 0:
            if div_yield is not None:
                stock.dividend_yield = div_yield
        stock.latest_price_updated_at = timezone.now()
        stock.save(update_fields=['latest_price', 'day_low', 'day_high', 'sector', 'dividend_yield', 'latest_price_updated_at'])

        for dt, row in data.iterrows():
            PriceHistory.objects.update_or_create(
                stock=stock,
                date=dt.date(),
                defaults={'close_price': float(row['Close'])},
            )

        prices[stock.symbol] = price
    return prices


@shared_task
def fetch_news_daily(days: int = 1, page_size: int = 10, language: str = 'en') -> dict[str, int]:
    api_key = os.getenv('NEWSAPI_KEY')
    if not api_key:
        return {'created': 0, 'seen': 0}

    newsapi = NewsApiClient(api_key=api_key)
    analyzer = SentimentIntensityAnalyzer()
    from_dt = timezone.now() - timedelta(days=days)
    from_iso = from_dt.strftime('%Y-%m-%d')

    created = 0
    seen = 0

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

        return {'created': created, 'updated': updated}


@shared_task
def fetch_finnhub_news_daily(days: int = 1) -> dict[str, int]:
    api_key = os.getenv('FINNHUB_API_KEY')
    if not api_key:
        return {'created': 0, 'seen': 0}

    client = finnhub.Client(api_key=api_key)
    analyzer = SentimentIntensityAnalyzer()

    to_dt = timezone.now().date()
    from_dt = to_dt - timedelta(days=days)

    created = 0
    seen = 0

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

    return {'created': created, 'seen': seen}


@shared_task
def fetch_google_news_daily(days: int = 1) -> dict[str, int]:
    analyzer = SentimentIntensityAnalyzer()
    cutoff = timezone.now() - timedelta(days=days)

    created = 0
    seen = 0

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

    return {'created': created, 'seen': seen}


@shared_task
def fetch_macro_daily(start: str = '2025-01-01') -> dict[str, int]:
    try:
        call_command('fetch_macro', start=start)
    except Exception:
        return {'ok': 0}
    return {'ok': 1}


@shared_task
def fetch_press_releases_hourly(hours: int = 24) -> dict[str, int]:
    cutoff = timezone.now() - timedelta(hours=hours)
    finbert = _finbert_pipeline()

    created = 0
    seen = 0

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

    return {'created': created, 'seen': seen}


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

    hits = 0

    def send_alert(subject: str, message: str, category: str, stock: Stock | None = None,
                   portfolio: Portfolio | None = None):
        nonlocal hits
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
        if stock.latest_price is not None and stock.latest_price >= price_threshold:
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

    return {'created': created, 'seen': seen}


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

    model = load_or_train_model(data)
    backtester = AIBacktester(data, model, symbol=symbol)
    result = backtester.run_simulation(lookback_days=lookback_days)

    retrained = False
    if result.win_rate < min_win_rate:
        model = train_fusion_model(data)
        retrained = model is not None

    return {
        'status': 'ok',
        'symbol': symbol,
        'lookback_days': lookback_days,
        'win_rate': result.win_rate,
        'min_win_rate': min_win_rate,
        'retrained': retrained,
    }


@shared_task
def execute_paper_trades() -> dict[str, Any]:
    """Execute live paper trades using model signals and risk rules."""
    watchlist = os.getenv('PAPER_WATCHLIST', 'SPY,AAPL,MSFT,NVDA,AMZN').split(',')
    watchlist = [s.strip().upper() for s in watchlist if s.strip()]
    buy_threshold = float(os.getenv('PAPER_BUY_THRESHOLD', '0.82'))
    sell_threshold = float(os.getenv('PAPER_SELL_THRESHOLD', '0.4'))
    trail_pct = float(os.getenv('PAPER_TRAIL_PCT', '0.04'))
    atr_mult = float(os.getenv('PAPER_ATR_MULT', '1.5'))
    risk_pct = float(os.getenv('PAPER_RISK_PCT', '0.015'))
    position_cap_pct = float(os.getenv('PAPER_POSITION_CAP_PCT', '0.10'))
    initial_capital = float(os.getenv('PAPER_CAPITAL', '10000'))

    closed_trades = PaperTrade.objects.filter(status='CLOSED')
    closed_pnl = float(sum([float(t.pnl or 0) for t in closed_trades]))
    open_trades = {t.ticker: t for t in PaperTrade.objects.filter(status='OPEN')}
    open_value = sum([float(t.entry_price) * float(t.quantity) for t in open_trades.values()])
    capital = initial_capital + closed_pnl
    available = max(0.0, capital - open_value)

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

    def _signal(symbol: str) -> dict[str, Any] | None:
        try:
            fusion = DataFusionEngine(symbol)
            fusion_df = fusion.fuse_all()
            if fusion_df is None or fusion_df.empty:
                return None
            payload = load_or_train_model(fusion_df)
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
            return {'signal': signal, 'features': feature_snapshot}
        except Exception:
            return None

    created = 0
    closed = 0

    for symbol, trade in open_trades.items():
        price = _latest_price(symbol)
        if price is None:
            continue
        signal_payload = _signal(symbol)
        signal = signal_payload['signal'] if signal_payload else None
        new_stop = max(float(trade.stop_loss), price * (1 - trail_pct))
        trade.stop_loss = new_stop
        should_close = price <= new_stop or (signal is not None and signal < sell_threshold)
        if should_close:
            trade.status = 'CLOSED'
            trade.exit_price = price
            trade.exit_date = timezone.now()
            trade.pnl = float(price - float(trade.entry_price)) * float(trade.quantity)
            closed += 1
        trade.save(update_fields=['stop_loss', 'status', 'exit_price', 'exit_date', 'pnl'])

    for symbol in watchlist:
        if symbol in open_trades:
            continue
        signal_payload = _signal(symbol)
        signal = signal_payload['signal'] if signal_payload else None
        if signal is None or signal < buy_threshold:
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
            entry_price=round(price, 2),
            quantity=quantity,
            entry_signal=signal,
            entry_features=(signal_payload or {}).get('features'),
            stop_loss=round(stop_loss, 2),
            status='OPEN',
            pnl=0,
            notes=f"Signal {signal:.2f} / ATR {atr:.2f}",
        )
        created += 1
        available = max(0.0, available - (quantity * price))

    return {'created': created, 'closed': closed, 'available': round(available, 2)}


@shared_task
def retrain_from_paper_trades_daily() -> dict[str, Any]:
    """Retrain the fusion model daily from closed paper trades + fresh API/news data."""
    learn_days = int(os.getenv('PAPER_TRADE_LEARN_DAYS', '90'))
    min_samples = int(os.getenv('PAPER_TRADE_MIN_SAMPLES', '20'))
    min_win_improve = float(os.getenv('PAPER_TRADE_MIN_WIN_IMPROVEMENT', '0.5'))
    cutoff = timezone.now() - timedelta(days=learn_days)

    trades = (
        PaperTrade.objects.filter(status='CLOSED', entry_date__gte=cutoff)
        .exclude(exit_date__isnull=True)
        .order_by('-entry_date')
    )

    if not trades.exists():
        return {'status': 'no_trades', 'samples': 0}

    samples: list[dict[str, float]] = []

    for trade in trades:
        symbol = (trade.ticker or '').strip().upper()
        if not symbol:
            continue
        if trade.entry_features:
            sample = {col: float((trade.entry_features or {}).get(col, 0.0)) for col in FEATURE_COLUMNS}
        else:
            fusion = DataFusionEngine(symbol)
            fusion_df = fusion.fuse_all()
            if fusion_df is None or fusion_df.empty:
                continue
            df = fusion_df.copy().sort_index()
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            entry_ts = pd.to_datetime(trade.entry_date)
            row = df[df.index <= entry_ts].tail(1)
            if row.empty:
                continue
            sample = {col: float(row.iloc[0].get(col, 0.0)) for col in FEATURE_COLUMNS}
        sample['label'] = 1 if float(trade.pnl or 0) > 0 else 0
        samples.append(sample)

    if len(samples) < min_samples:
        return {'status': 'insufficient_samples', 'samples': len(samples), 'min_required': min_samples}

    samples_df = pd.DataFrame(samples)
    payload = train_fusion_model_from_labels(samples_df, save=False)
    if not payload:
        return {'status': 'failed', 'samples': len(samples), 'trained': False}

    symbol = os.getenv('BACKTEST_SYMBOL', 'SPY').strip().upper()
    lookback_days = int(os.getenv('BACKTEST_LOOKBACK_DAYS', '90'))
    engine = DataFusionEngine(symbol)
    data = engine.fuse_all()
    if data is None or data.empty:
        return {'status': 'no_data', 'samples': len(samples), 'trained': False}

    current = load_or_train_model(data)
    current_result = AIBacktester(data, current).run_simulation(lookback_days=lookback_days)
    candidate_result = AIBacktester(data, payload).run_simulation(lookback_days=lookback_days)

    improved = (
        candidate_result.win_rate >= current_result.win_rate + min_win_improve
        or candidate_result.sharpe_ratio > current_result.sharpe_ratio
    )

    if improved:
        save_model_payload(payload)

    return {
        'status': 'ok' if improved else 'skipped',
        'samples': len(samples),
        'trained': bool(improved),
        'current_win_rate': current_result.win_rate,
        'candidate_win_rate': candidate_result.win_rate,
    }


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
