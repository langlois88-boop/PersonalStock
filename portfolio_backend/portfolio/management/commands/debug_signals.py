from __future__ import annotations

import os
from typing import Any

from django.core.management.base import BaseCommand
from django.utils import timezone

from portfolio.models import SandboxWatchlist
from portfolio.tasks import (
    _model_signal,
    _news_sentiment_score,
    _latest_bid_ask,
    _symbol_currency,
    _ny_time_now,
    _is_valid_price,
    _latest_price_snapshot,
    get_order_book_imbalance,
    get_latest_bid_ask_spread_pct,
    get_model_path,
)


class Command(BaseCommand):
    help = "Debug Alpaca signal filters for selected tickers."

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbols",
            type=str,
            default=os.getenv("DEBUG_SIGNALS_SYMBOLS", "SPY,BTO.TO,ONCY,SLQT"),
            help="Comma-separated list of symbols to diagnose.",
        )

    def handle(self, *args, **options):
        symbols_arg = options.get("symbols") or ""
        symbols = [s.strip().upper() for s in symbols_arg.split(",") if s.strip()]
        if not symbols:
            self.stdout.write("No symbols to diagnose.")
            return

        now_ny = _ny_time_now()
        if now_ny.weekday() >= 5:
            self.stdout.write("Market Closed - weekend (data may be frozen).")
        else:
            if not (timezone.datetime(now_ny.year, now_ny.month, now_ny.day, 9, 30, tzinfo=now_ny.tzinfo).time() <= now_ny.time() <=
                    timezone.datetime(now_ny.year, now_ny.month, now_ny.day, 16, 0, tzinfo=now_ny.tzinfo).time()):
                self.stdout.write("Market Closed - outside regular hours (data may be frozen).")

        min_conf = float(os.getenv("ALPACA_MIN_CONFIDENCE", "0.7"))
        min_sent = float(os.getenv("ALPACA_MIN_SENTIMENT", "0.0"))
        min_imb = float(os.getenv("ALPACA_MIN_IMBALANCE", "1.0"))
        max_cost_pct = float(os.getenv("ALPACA_MAX_COST_PCT", "0.01"))
        commission_pct = float(os.getenv("ALPACA_COMMISSION_PCT", "0.0"))

        penny_watch = SandboxWatchlist.objects.filter(sandbox="AI_PENNY").first()
        penny_symbols = set([str(s).strip().upper() for s in (penny_watch.symbols if penny_watch else []) if str(s).strip()])

        self.stdout.write("--- 🔍 DIAGNOSTIC ALGO-TRADING START ---")
        for symbol in symbols:
            self.stdout.write(f"\nAnalyse de {symbol}...")
            universe = "PENNY" if symbol in penny_symbols else "BLUECHIP"
            use_alpaca = _symbol_currency(symbol) != "CAD"

            payload = _model_signal(symbol, universe, get_model_path(universe), use_alpaca=use_alpaca)
            confidence = float(payload.get("signal") or 0.0) if payload else 0.0
            status_ml = "✅" if confidence >= min_conf else "❌"
            self.stdout.write(f"{status_ml} Confidence: {confidence:.2f} (Besoin de: {min_conf:.2f})")

            feature_snapshot = (payload or {}).get("features") or {}
            sentiment = feature_snapshot.get("sentiment_score")
            if sentiment is None:
                sentiment, _ = _news_sentiment_score(symbol, days=1)
            sentiment = float(sentiment or 0.0)
            status_sent = "✅" if sentiment >= min_sent else "❌"
            self.stdout.write(f"{status_sent} Sentiment: {sentiment:.2f} (Besoin de: {min_sent:.2f})")

            imbalance = get_order_book_imbalance(symbol)
            if imbalance is None:
                self.stdout.write("⚠️ Imbalance: n/a (market closed or data unavailable)")
            else:
                status_imb = "✅" if imbalance >= min_imb else "❌"
                self.stdout.write(f"{status_imb} Imbalance: {imbalance:.2f} (Besoin de: {min_imb:.2f})")

            spread_pct = get_latest_bid_ask_spread_pct(symbol)
            if spread_pct is None:
                bid, ask = _latest_bid_ask(symbol)
                if bid and ask and ask > 0:
                    spread_pct = float((ask - bid) / ((ask + bid) / 2))
            if spread_pct is None:
                self.stdout.write("⚠️ Spread: n/a")
            else:
                cost_pct = float(spread_pct) + commission_pct
                status_cost = "✅" if cost_pct <= max_cost_pct else "❌"
                self.stdout.write(
                    f"{status_cost} Spread+Fees: {cost_pct:.4f} (Max {max_cost_pct:.4f})"
                )

            price = _latest_price_snapshot(symbol)
            if _is_valid_price(price):
                self.stdout.write(f"Prix actuel: {float(price):.2f}")

        self.stdout.write("\n--- DIAGNOSTIC FINI ---")
