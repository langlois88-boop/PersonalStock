import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from django.utils import timezone as dj_timezone

from portfolio.models import SandboxWatchlist, StockNews
from portfolio.alpaca_data import get_intraday_bars_range
from portfolio import market_data as yf


def _setup_django() -> None:
    import django

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'portfolio_backend.settings')
    django.setup()


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


def _finbert_score(pipe: Any, text: str) -> float:
    if pipe is None or not text:
        return 0.0
    try:
        result = pipe(text)[0]
        label = str(result.get('label') or '').upper()
        score = float(result.get('score') or 0)
        if label == 'POSITIVE':
            return score
        if label == 'NEGATIVE':
            return -score
        return 0.0
    except Exception:
        return 0.0


def _collect_watchlist_symbols(limit: int = 50) -> list[str]:
    watch = SandboxWatchlist.objects.filter(sandbox='WATCHLIST').first()
    symbols = [str(s).strip().upper() for s in (watch.symbols if watch else []) if str(s).strip()]
    if not symbols:
        fallback = os.getenv('WATCHLIST_SYMBOLS', os.getenv('WATCHLIST', '')).strip()
        if fallback:
            symbols = [s.strip().upper() for s in fallback.split(',') if s.strip()]
    return symbols[:limit]


def _finnhub_news(symbol: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
    key = os.getenv('FINNHUB_API_KEY') or os.getenv('FINNHUB_KEY')
    if not key:
        return []
    try:
        import finnhub
        client = finnhub.Client(api_key=key)
        items = client.company_news(symbol, _from=start.strftime('%Y-%m-%d'), to=end.strftime('%Y-%m-%d')) or []
        return [i for i in items if isinstance(i, dict)]
    except Exception:
        return []


def _google_news_from_db(symbol: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
    qs = (
        StockNews.objects.filter(stock__symbol__iexact=symbol, published_at__gte=start, published_at__lte=end)
        .order_by('published_at')
        .values('headline', 'summary', 'published_at')
    )
    results = []
    for item in qs:
        published_at = item.get('published_at')
        if not published_at:
            continue
        results.append({
            'headline': item.get('headline') or '',
            'summary': item.get('summary') or '',
            'datetime': published_at,
        })
    return results


def _intraday_price_move(symbol: str, news_dt: datetime, hours: int = 4) -> tuple[float | None, float | None, float | None]:
    start = news_dt.astimezone(timezone.utc)
    end = start + timedelta(hours=hours)
    df = get_intraday_bars_range(symbol, start=start, end=end)
    if df is None or df.empty:
        return None, None, None
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df = df.dropna(subset=['timestamp']).sort_values('timestamp')
    if df.empty or 'close' not in df.columns:
        return None, None, None
    first = df.iloc[0]
    base = float(first.get('close') or 0)
    if base <= 0:
        return None, None, None
    max_price = float(df['close'].max())
    min_price = float(df['close'].min())
    volume = float(first.get('volume') or 0)
    move = max(abs(max_price - base), abs(min_price - base)) / base
    return move, base, volume


def train_sentiment_impact() -> dict[str, Any]:
    pipe = _finbert_pipeline()
    symbols = _collect_watchlist_symbols(limit=int(os.getenv('SENTIMENT_IMPACT_SYMBOL_LIMIT', '50')))
    if not symbols:
        return {'status': 'empty'}

    end = dj_timezone.now()
    start = end - timedelta(days=365 * 2)

    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        news_items = _google_news_from_db(symbol, start=start, end=end)
        news_items += _finnhub_news(symbol, start=start, end=end)
        seen = set()
        for item in news_items:
            headline = item.get('headline') or item.get('title') or ''
            summary = item.get('summary') or ''
            published = item.get('published_at') or item.get('datetime')
            if not headline or not published:
                continue
            if isinstance(published, (int, float)):
                published = datetime.fromtimestamp(published, tz=timezone.utc)
            if isinstance(published, str):
                try:
                    published = datetime.fromisoformat(published)
                except Exception:
                    continue
            if not isinstance(published, datetime):
                continue
            key = f"{symbol}:{headline}:{published.isoformat()}"
            if key in seen:
                continue
            seen.add(key)

            move, base, volume = _intraday_price_move(symbol, published)
            if move is None:
                continue

            score = _finbert_score(pipe, f"{headline}. {summary}")
            rows.append({
                'symbol': symbol,
                'finbert_score': score,
                'initial_volume': volume or 0.0,
                'hour_of_day': published.astimezone(timezone.utc).hour,
                'day_of_week': published.weekday(),
                'volatility_4h': float(move),
            })

    if not rows:
        return {'status': 'no_samples'}

    dataset = pd.DataFrame(rows).fillna(0.0)
    X = dataset[['finbert_score', 'initial_volume', 'hour_of_day', 'day_of_week']]
    y = dataset['volatility_4h']

    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42,
    )
    model.fit(X, y)

    model_path = Path(__file__).resolve().parents[1] / 'ml_engine' / 'sentiment_impact_model.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump({'model': model, 'features': list(X.columns)}, model_path)
    return {'status': 'ok', 'samples': len(dataset), 'model_path': str(model_path)}


if __name__ == '__main__':
    _setup_django()
    result = train_sentiment_impact()
    print(result)
