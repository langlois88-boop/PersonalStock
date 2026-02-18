from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import yfinance as yf
from django.db import models
from django.utils import timezone
from openai import OpenAI
from openai import APIError, RateLimitError, AuthenticationError

from .models import PriceHistory, Stock, StockNews, AlertEvent
from .ml_engine.engine.data_fusion import DataFusionEngine
from .ml_engine.backtester import AIBacktester, load_or_train_model, get_model_path


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _get_recent_prices(symbol: str, stock: Stock | None) -> list[float]:
    if stock:
        prices = list(PriceHistory.objects.filter(stock=stock).order_by("-date")[:180])
        series = [float(p.close_price or 0) for p in prices if p.close_price]
        if series:
            return series

    try:
        hist = yf.Ticker(symbol).history(period="6mo", interval="1d")
        if not hist.empty and "Close" in hist:
            return [float(v) for v in hist["Close"].tolist()][::-1]
    except Exception:
        return []

    return []


def _price_change(series: list[float], days: int) -> float | None:
    if len(series) <= days:
        return None
    latest = series[0]
    past = series[days]
    if past == 0:
        return None
    return (latest - past) / past


def _lite_summary(payload: dict[str, Any]) -> str:
    lines = []
    price = payload.get("price")
    news = payload.get("news_sentiment")
    rev = payload.get("revenue_growth")
    earn = payload.get("earnings_growth")
    margins = payload.get("profit_margins")
    roe = payload.get("roe")
    sector = payload.get("sector")
    sector_roe_avg = payload.get("sector_roe_avg")
    beta = payload.get("beta")
    change_1m = payload.get("price_change_1m")
    change_3m = payload.get("price_change_3m")

    if price is not None:
        lines.append(f"Current price: ${float(price):.2f}.")
    if change_1m is not None:
        lines.append(f"1M performance: {change_1m * 100:.2f}%." )
    if change_3m is not None:
        lines.append(f"3M performance: {change_3m * 100:.2f}%." )

    if rev is not None:
        lines.append(f"Revenue growth: {rev * 100:.2f}%.")
    if earn is not None:
        lines.append(f"Earnings growth: {earn * 100:.2f}%.")
    if margins is not None:
        lines.append(f"Profit margin: {margins * 100:.2f}%.")
    if roe is not None:
        lines.append(f"ROE: {roe * 100:.2f}%.")
    if sector and sector_roe_avg is not None:
        lines.append(
            f"Sector ROE avg ({sector}): {sector_roe_avg * 100:.2f}%."
        )

    if beta is not None:
        lines.append(f"Beta: {float(beta):.2f} (lower is more stable).")

    if news is not None:
        sentiment_label = "positive" if news > 0.2 else "negative" if news < -0.2 else "neutral"
        lines.append(f"News sentiment: {sentiment_label} ({float(news):.2f}).")

    score = 0
    if rev and rev > 0:
        score += 1
    if earn and earn > 0:
        score += 1
    if margins and margins > 0:
        score += 1
    if roe and roe > 0:
        score += 1
    if sector_roe_avg is not None and roe is not None:
        if roe > sector_roe_avg:
            score += 1
        elif roe < sector_roe_avg * 0.5:
            score -= 1
    if beta and beta < 1:
        score += 1
    if news and news > 0.2:
        score += 1
    if change_3m and change_3m > 0:
        score += 1

    if score >= 5:
        tag = "BUY"
    elif score >= 3:
        tag = "HOLD"
    else:
        tag = "SELL"

    lines.append(f"Action tag: {tag}.")
    return "\n".join(lines)


def build_scout_summary(symbol: str, allow_llm: bool = True) -> dict[str, Any]:
    symbol = symbol.upper().strip()
    stock = Stock.objects.filter(symbol__iexact=symbol).first()

    info: dict[str, Any] = {}
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:
        info = {}

    cutoff = timezone.now() - timedelta(days=7)
    news_qs = StockNews.objects.filter(stock=stock, fetched_at__gte=cutoff) if stock else None
    news_items = []
    news_sentiment = None

    if news_qs is not None:
        news_items = list(news_qs.order_by("-published_at")[:5])
        news_sentiment = news_qs.aggregate(avg=models.Avg("sentiment")).get("avg")

    if news_sentiment is None or news_sentiment == 0 or not news_items:
        try:
            news_items = (yf.Ticker(symbol).news or [])[:5]
        except Exception:
            news_items = []

        if news_items:
            scores = []
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            analyzer = SentimentIntensityAnalyzer()
            for item in news_items:
                headline = (item.get("title") or "")
                summary = (item.get("summary") or "")
                text = f"{headline}. {summary}".strip()
                if text:
                    scores.append(analyzer.polarity_scores(text).get("compound", 0))
            news_sentiment = float(np.mean(scores)) if scores else 0

    news_sentiment = float(news_sentiment or 0)

    prices = _get_recent_prices(symbol, stock)
    change_1m = _price_change(prices, 20)
    change_3m = _price_change(prices, 60)
    change_6m = _price_change(prices, 120)

    sector_value = (stock.sector if stock else None) or info.get("sector")
    if sector_value and str(sector_value).strip().lower() == "unknown":
        sector_value = info.get("sector")

    backtest_win_rate = None
    try:
        fusion = DataFusionEngine(symbol)
        df = fusion.fuse_all()
        if df is not None and not df.empty:
            payload_model = load_or_train_model(df, model_path=get_model_path('BLUECHIP'))
            if payload_model and payload_model.get('model'):
                result = AIBacktester(df, payload_model, symbol=symbol).run_simulation(lookback_days=180)
                backtest_win_rate = float(result.win_rate)
    except Exception:
        backtest_win_rate = None

    if backtest_win_rate is not None and backtest_win_rate >= 60:
        cutoff = timezone.now() - timedelta(hours=12)
        existing = AlertEvent.objects.filter(
            category='SCOUT_HIGH_PROB',
            message__icontains=symbol,
            created_at__gte=cutoff,
        ).exists()
        if not existing:
            AlertEvent.objects.create(
                category='SCOUT_HIGH_PROB',
                message=f"🔥 Signal de Haute Probabilité détecté : {symbol} (Backtest {backtest_win_rate:.1f}%)",
            )

    payload = {
        "symbol": symbol,
        "name": (stock.name if stock else info.get("shortName") or info.get("longName")),
        "sector": sector_value,
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
        "beta": info.get("beta"),
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "profit_margins": info.get("profitMargins"),
        "roe": info.get("returnOnEquity"),
        "price": info.get("currentPrice") or (stock.latest_price if stock else None),
        "price_change_1m": change_1m,
        "price_change_3m": change_3m,
        "price_change_6m": change_6m,
        "news_sentiment": news_sentiment,
        "backtest_win_rate": backtest_win_rate,
        "headlines": [
            (n.headline if hasattr(n, "headline") else n.get("title"))
            for n in news_items
            if (n.headline if hasattr(n, "headline") else n.get("title"))
        ],
    }

    sector_roe_samples = []
    sector_name = payload.get("sector")
    if sector_name and str(sector_name).strip().lower() != "unknown":
        for peer in Stock.objects.all().order_by("symbol")[:50]:
            try:
                peer_sector = peer.sector
                if not peer_sector or str(peer_sector).strip().lower() == "unknown":
                    peer_info = yf.Ticker(peer.symbol).info or {}
                    peer_sector = peer_info.get("sector")
                if not peer_sector or str(peer_sector).strip().lower() == "unknown":
                    continue
                if str(peer_sector).strip().lower() != str(sector_name).strip().lower():
                    continue
                peer_info = yf.Ticker(peer.symbol).info or {}
                peer_roe = peer_info.get("returnOnEquity")
                if peer_roe is not None:
                    sector_roe_samples.append(float(peer_roe))
                if len(sector_roe_samples) >= 30:
                    break
            except Exception:
                continue

    if sector_roe_samples:
        payload["sector_roe_avg"] = float(np.mean(sector_roe_samples))
    else:
        payload["sector_roe_avg"] = None

    summary_text = ""
    if allow_llm and os.getenv("OPENAI_API_KEY"):
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = (
                "You are a financial analyst assistant. Summarize the stock health and risk in 5-7 bullets, "
                "using only the provided data. End with a cautious action tag: BUY, HOLD, or SELL. "
                "Be conservative and avoid hype.\n\n"
                f"Data: {payload}"
            )

            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You provide concise, conservative equity summaries."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=450,
            )

            summary_text = response.choices[0].message.content.strip()
        except (RateLimitError, APIError, AuthenticationError):
            summary_text = ""

    if not summary_text:
        summary_text = _lite_summary(payload)

    return {
        "symbol": symbol,
        "summary": summary_text,
        "data": payload,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
