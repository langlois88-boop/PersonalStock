from __future__ import annotations

import json
import os
from django.utils import timezone
from dataclasses import dataclass
from typing import Any

import pandas as pd
from openai import OpenAI
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ..alpaca_data import get_intraday_context
from ..models import AccountTransaction, Portfolio, PortfolioHolding, Stock, StockNews
from ..ml_engine.engine.data_fusion import DataFusionEngine
from ..ml_engine.backtester import (
    apply_feature_weighting_to_signal,
    get_model_path,
    load_or_train_model,
)


@dataclass
class MarketContext:
    symbol: str
    price: float | None
    rsi14: float | None
    patterns: list[str]
    rvol: float | None
    ml_signal: float | None
    ml_confidence: float | None
    sentiment: float | None
    portfolio_holdings: list[dict[str, Any]]


class DanasMLRouter:
    def __init__(self) -> None:
        self.default_universe = os.getenv("DANAS_DEFAULT_UNIVERSE", "BLUECHIP")

    def infer_universe(self, symbol: str, price: float | None, market_type: str | None) -> str:
        if market_type:
            return market_type
        if price is not None and price < 7:
            return "PENNY"
        return self.default_universe

    def predict(self, symbol: str, market_type: str | None = None) -> dict[str, float | None]:
        try:
            engine = DataFusionEngine(symbol)
            frame = engine.fuse_all()
            if frame is None or frame.empty:
                return {"signal": None, "confidence": None}

            try:
                price = float(frame.tail(1).iloc[0].get("Close") or 0.0)
            except Exception:
                price = None

            universe = self.infer_universe(symbol, price, market_type)
            payload = load_or_train_model(frame, model_path=get_model_path(universe))
            if not payload or not payload.get("model"):
                return {"signal": None, "confidence": None}

            last_row = frame.tail(1).copy()
            feature_list = payload.get("features") or []
            for col in feature_list:
                if col not in last_row.columns:
                    last_row[col] = 0.0
            features = last_row[feature_list].fillna(0).values
            try:
                proba = float(payload["model"].predict_proba(features)[0][1])
                signal = 1.0 if proba >= 0.5 else 0.0
                confidence = max(proba, 1.0 - proba)
            except Exception:
                raw = float(payload["model"].predict(features)[0]) if hasattr(payload["model"], "predict") else 0.0
                signal = 1.0 if raw >= 0.5 else 0.0
                confidence = abs(raw - 0.5) + 0.5

            weighted = apply_feature_weighting_to_signal(float(signal), last_row.iloc[0], symbol)
            return {
                "signal": round(weighted, 4),
                "confidence": round(float(confidence) * 100, 2) if confidence is not None else None,
            }
        except Exception:
            return {"signal": None, "confidence": None}


class DanasBroker:
    def __init__(self, user: Any | None = None) -> None:
        self.user = user
        base_url = (
            os.getenv("DANAS_CHAT_BASE_URL")
            or os.getenv("OLLAMA_CHAT_BASE_URL")
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:8090/v1"
        ).strip().rstrip("/")
        self.model = os.getenv("DANAS_MODEL", os.getenv("OLLAMA_MODEL", "deepseek-r1")).strip()
        api_key = os.getenv("DANAS_API_KEY", "sk-no-key-required")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.analyzer = SentimentIntensityAnalyzer()
        self.ml_engine = DanasMLRouter()

    def _portfolio_snapshot(self) -> list[dict[str, Any]]:
        portfolio = None
        if self.user is not None:
            portfolio = Portfolio.objects.filter(user=self.user).first()
        if portfolio is None:
            portfolio = Portfolio.objects.first()
        if not portfolio:
            return []
        holdings = (
            PortfolioHolding.objects.select_related("stock")
            .filter(portfolio=portfolio)
            .order_by("-id")[:25]
        )
        items = []
        for holding in holdings:
            stock = holding.stock
            items.append(
                {
                    "symbol": (stock.symbol or "").strip().upper(),
                    "shares": float(holding.shares or 0),
                    "avg_cost": float(holding.avg_cost or 0),
                    "latest_price": float(stock.latest_price or 0),
                }
            )
        if items:
            return items

        positions: dict[int, dict[str, Any]] = {}
        account_qs = AccountTransaction.objects.select_related("stock", "account").filter(
            account__account_type__in=["TFSA", "CRI", "CASH"]
        )
        if self.user is not None:
            account_qs = account_qs.filter(account__user=self.user)
        for tx in account_qs:
            if not tx.stock or not tx.stock.symbol:
                continue
            if tx.type == "DIVIDEND":
                continue
            sign = 1 if tx.type == "BUY" else -1
            entry = positions.setdefault(
                tx.stock_id,
                {"stock": tx.stock, "shares": 0.0, "buy_qty": 0.0, "buy_cost": 0.0},
            )
            qty = float(tx.quantity or 0)
            entry["shares"] += qty * sign
            if tx.type == "BUY":
                entry["buy_qty"] += qty
                entry["buy_cost"] += qty * float(tx.price or 0)

        fallback_items = []
        for entry in positions.values():
            stock = entry.get("stock")
            shares = float(entry.get("shares") or 0)
            if not stock or shares <= 0:
                continue
            buy_qty = float(entry.get("buy_qty") or 0)
            buy_cost = float(entry.get("buy_cost") or 0)
            avg_cost = (buy_cost / buy_qty) if buy_qty else 0.0
            fallback_items.append(
                {
                    "symbol": (stock.symbol or "").strip().upper(),
                    "shares": shares,
                    "avg_cost": avg_cost,
                    "latest_price": float(stock.latest_price or 0),
                }
            )
        return fallback_items

    def _news_sentiment(self, symbol: str) -> float | None:
        if not symbol:
            return None
        stock = Stock.objects.filter(symbol__iexact=symbol).first()
        qs = StockNews.objects.filter(stock=stock) if stock else StockNews.objects.filter(stock__symbol__iexact=symbol)
        latest = list(qs.order_by("-published_at")[:8])
        if not latest:
            return None
        scores = []
        for item in latest:
            headline = (item.headline or "").strip()
            if not headline:
                continue
            scores.append(self.analyzer.polarity_scores(headline)["compound"])
        if not scores:
            return None
        return float(sum(scores) / len(scores))

    def get_market_context(self, symbol: str, market_type: str | None = None) -> MarketContext:
        context = get_intraday_context(symbol)
        price = None
        rsi14 = None
        patterns: list[str] = []
        rvol = None
        if context:
            bars = context.get("bars")
            if isinstance(bars, pd.DataFrame) and not bars.empty:
                try:
                    price = float(bars.iloc[-1]["close"])
                except Exception:
                    price = None
            rsi14 = context.get("rsi14")
            try:
                rsi14 = float(rsi14) if rsi14 is not None else None
            except Exception:
                rsi14 = None
            patterns = list(context.get("patterns") or [])
            try:
                rvol = float(context.get("rvol")) if context.get("rvol") is not None else None
            except Exception:
                rvol = None

        if price is None:
            try:
                fusion = DataFusionEngine(symbol, fast_mode=True)
                frame = fusion.fuse_all()
                if frame is not None and not frame.empty:
                    price = float(frame.tail(1).iloc[0].get("Close") or 0.0)
            except Exception:
                price = None

        ml_result = self.ml_engine.predict(symbol, market_type)
        sentiment = self._news_sentiment(symbol)
        holdings = self._portfolio_snapshot()

        return MarketContext(
            symbol=(symbol or "").upper(),
            price=price,
            rsi14=rsi14,
            patterns=patterns,
            rvol=rvol,
            ml_signal=ml_result.get("signal"),
            ml_confidence=ml_result.get("confidence"),
            sentiment=sentiment,
            portfolio_holdings=holdings,
        )

    def prepare_portfolio_payload(self, portfolio_holdings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not portfolio_holdings:
            return []
        total_value = 0.0
        for item in portfolio_holdings:
            shares = float(item.get("shares") or 0)
            price = float(item.get("latest_price") or 0)
            total_value += shares * price
        total_value = max(total_value, 1.0)
        cleaned = []
        for item in portfolio_holdings:
            shares = float(item.get("shares") or 0)
            price = float(item.get("latest_price") or 0)
            value = shares * price
            cleaned.append(
                {
                    "symbol": item.get("symbol"),
                    "allocation": round(value / total_value * 100, 2),
                    "avg_cost": float(item.get("avg_cost") or 0),
                    "price": price,
                }
            )
        return cleaned

    def _build_prompt(self, ctx: MarketContext, user_query: str) -> str:
        portfolio = self.prepare_portfolio_payload(ctx.portfolio_holdings)
        return (
            "Tu es Danas, un expert Quant. Réponds uniquement en français. "
            "Analyse technique + news + ML, et propose un plan Entry/Stop/TP.\n"
            f"Ticker: {ctx.symbol}\n"
            f"Prix: {ctx.price}\n"
            f"RSI14: {ctx.rsi14}\n"
            f"Patterns: {ctx.patterns}\n"
            f"RVOL: {ctx.rvol}\n"
            f"ML signal: {ctx.ml_signal} | ML confiance: {ctx.ml_confidence}%\n"
            f"Sentiment: {ctx.sentiment}\n"
            f"Portfolio: {json.dumps(portfolio, ensure_ascii=False)}\n"
            f"Question: {user_query}\n"
        )

    def ask_danas(self, user_query: str, symbol: str | None = None, market_type: str | None = None) -> str:
        now = timezone.now().strftime('%d/%m/%Y %H:%M')
        if not symbol:
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"Tu es Danas. Nous sommes le {now}. "
                        "Ignore ta date interne et utilise uniquement les données live fournies. "
                        "Réponds en français, concis et actionnable."
                    ),
                },
                {"role": "user", "content": user_query},
            ]
        else:
            ctx = self.get_market_context(symbol, market_type)
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"Tu es Danas. Nous sommes le {now}. "
                        "Ignore ta date interne et utilise uniquement les données live fournies. "
                        "Réponds en français, concis et actionnable."
                    ),
                },
                {"role": "user", "content": self._build_prompt(ctx, user_query)},
            ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
        )
        return (response.choices[0].message.content or "").strip()
