from __future__ import annotations

import os
import json
import time
import threading
from collections import deque
from typing import Any

from django.core.management.base import BaseCommand
from portfolio.models import ActiveSignal
from portfolio.alpaca_data import get_latest_trade_price
from portfolio.tasks import _send_telegram_alert

try:
    from alpaca.data.live import StockDataStream
    from alpaca.data.enums import DataFeed
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except Exception:  # pragma: no cover
    StockDataStream = None
    DataFeed = None
    TradingClient = None
    MarketOrderRequest = None
    OrderSide = None
    TimeInForce = None

try:
    from google import genai
except Exception:  # pragma: no cover
    genai = None


class Command(BaseCommand):
    help = "Realtime tracker for manual ActiveSignal positions (Alpaca stream + Gemini)."

    def handle(self, *args, **options):
        if StockDataStream is None or TradingClient is None:
            self.stdout.write(self.style.ERROR("alpaca-py not installed."))
            return

        alpaca_key = os.getenv("ALPACA_API_KEY")
        alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
        alpaca_base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        alpaca_data_feed = os.getenv("ALPACA_DATA_FEED", "iex")

        if not alpaca_key or not alpaca_secret:
            self.stdout.write(self.style.ERROR("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY."))
            return

        gemini_key = os.getenv("GEMINI_AI_API_KEY")
        gemini_model = os.getenv("GEMINI_AI_MODEL", "models/gemini-2.5-flash")
        gemini_enabled = os.getenv("TRACKER_GEMINI", "true").lower() in {"1", "true", "yes", "y"}

        allowed_symbols_raw = os.getenv("TRACKER_SYMBOLS", "").strip()
        allowed_symbols = {
            s.strip().upper()
            for s in allowed_symbols_raw.replace(";", ",").replace(" ", ",").split(",")
            if s.strip()
        }

        auto_sell = os.getenv("TRACKER_AUTO_SELL", "false").lower() in {"1", "true", "yes", "y"}
        min_bars = int(os.getenv("TRACKER_MIN_BARS", "10"))
        trend_cooldown = int(os.getenv("TRACKER_TREND_COOLDOWN_SEC", "180"))
        heartbeat_minutes = int(os.getenv("TRACKER_HEARTBEAT_MINUTES", "15"))

        feed_value = (alpaca_data_feed or "iex").strip().lower()
        if DataFeed is None:
            self.stdout.write(self.style.ERROR("alpaca-py DataFeed not available."))
            return
        feed_enum = DataFeed.SIP if feed_value == "sip" else DataFeed.IEX

        stream = StockDataStream(
            alpaca_key,
            alpaca_secret,
            feed=feed_enum,
        )

        trading_client = TradingClient(
            alpaca_key,
            alpaca_secret,
            paper="paper" in alpaca_base_url,
        )

        candles: dict[str, deque[dict[str, Any]]] = {}
        last_trend_ts: dict[str, float] = {}
        last_status_ts: dict[str, float] = {}
        last_heartbeat_ts: dict[str, float] = {}
        first_bar_sent: set[str] = set()
        first_trade_sent: set[str] = set()
        first_poll_sent: set[str] = set()
        subscribed: set[str] = set()

        def _now_ts() -> float:
            return time.time()

        def _cooldown_ok(key: dict[str, float], symbol: str, seconds: int) -> bool:
            last = key.get(symbol, 0)
            if _now_ts() - last < seconds:
                return False
            key[symbol] = _now_ts()
            return True

        def _predict_trend(symbol: str, bars: list[dict[str, Any]]) -> tuple[str, str]:
            if not gemini_enabled or not gemini_key or genai is None or len(bars) < min_bars:
                return "NEUTRAL", "Gemini indisponible."
            client = genai.Client(api_key=gemini_key)
            payload = [
                {"t": b["t"], "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"]}
                for b in bars[-min_bars:]
            ]
            prompt = (
                "Analyse ces 10 dernières bougies 1m et donne une prédiction courte: "
                "UP/DOWN/NEUTRAL + une phrase de justification.\n"
                f"Data: {json.dumps(payload, ensure_ascii=False)}"
            )
            try:
                resp = client.models.generate_content(model=gemini_model, contents=prompt)
                text = (getattr(resp, "text", "") or "").strip()
                if text.upper().startswith("UP"):
                    return "UP", text
                if text.upper().startswith("DOWN"):
                    return "DOWN", text
                return "NEUTRAL", text or "NEUTRAL"
            except Exception:
                return "NEUTRAL", "Erreur Gemini."

        async def on_bar(bar):
            symbol = bar.symbol
            if symbol not in candles:
                candles[symbol] = deque(maxlen=max(10, min_bars))
            candles[symbol].append({
                "t": bar.timestamp.isoformat(),
                "o": float(bar.open),
                "h": float(bar.high),
                "l": float(bar.low),
                "c": float(bar.close),
                "v": float(bar.volume),
            })

            if symbol not in first_bar_sent:
                first_bar_sent.add(symbol)
                _send_telegram_alert(
                    f"📡 Flux live {symbol} OK. Dernier prix {float(bar.close):.4f}.",
                    allow_during_blackout=True,
                    category="tracker",
                )

            trend, note = _predict_trend(symbol, list(candles[symbol]))
            if trend == "UP" and _cooldown_ok(last_trend_ts, symbol, trend_cooldown):
                _send_telegram_alert(
                    f"🚀 {symbol} tendance forte. {note}",
                    allow_during_blackout=True,
                    category="tracker",
                )
            elif trend == "DOWN" and _cooldown_ok(last_trend_ts, symbol, trend_cooldown):
                _send_telegram_alert(
                    f"⚠️ {symbol} danger de chute. {note}",
                    allow_during_blackout=True,
                    category="tracker",
                )
            elif trend == "NEUTRAL" and heartbeat_minutes > 0:
                if _cooldown_ok(last_heartbeat_ts, symbol, heartbeat_minutes * 60):
                    _send_telegram_alert(
                        f"📊 {symbol} tendance neutre. {note} (prix {float(bar.close):.4f})",
                        allow_during_blackout=True,
                        category="tracker",
                    )

            _check_targets(symbol, float(bar.close))

        async def on_trade(trade):
            symbol = trade.symbol
            if symbol not in first_trade_sent:
                first_trade_sent.add(symbol)
                _send_telegram_alert(
                    f"📡 Trade live {symbol} OK. Dernier prix {float(trade.price):.4f}.",
                    allow_during_blackout=True,
                    category="tracker",
                )
            if heartbeat_minutes > 0:
                if _cooldown_ok(last_heartbeat_ts, f"{symbol}:trade", heartbeat_minutes * 60):
                    _send_telegram_alert(
                        f"📊 {symbol} live trade {float(trade.price):.4f}",
                        allow_during_blackout=True,
                        category="tracker",
                    )
            _check_targets(symbol, float(trade.price))

        def _check_targets(symbol: str, price: float):
            if symbol not in signals:
                return
            entry = signals[symbol]["entry"]
            target = signals[symbol]["target"]
            stop = signals[symbol]["stop"]
            qty = signals[symbol]["qty"]

            if price >= target and _cooldown_ok(last_status_ts, f"{symbol}:target", 300):
                _send_telegram_alert(
                    f"🚀 TARGET touchée {symbol} @ {price:.2f}. Cible {target:.2f}.",
                    allow_during_blackout=True,
                    category="tracker",
                )
            if price <= stop and _cooldown_ok(last_status_ts, f"{symbol}:stop", 300):
                msg = f"⚠️ STOP touché {symbol} @ {price:.2f}. Stop {stop:.2f}."
                if auto_sell:
                    try:
                        order = MarketOrderRequest(
                            symbol=symbol,
                            qty=qty,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY,
                        )
                        trading_client.submit_order(order)
                        msg += " Ordre market SELL envoyé."
                    except Exception as exc:
                        msg += f" Erreur ordre: {exc}"
                _send_telegram_alert(msg, allow_during_blackout=True, category="tracker")

        def _load_signals() -> dict[str, dict[str, Any]]:
            open_signals = ActiveSignal.objects.filter(status="OPEN", meta__daytrade=True)
            updated = {}
            for sig in open_signals:
                symbol = (sig.ticker or "").strip().upper()
                if not symbol:
                    continue
                if allowed_symbols and symbol not in allowed_symbols:
                    continue
                updated[symbol] = {
                    "entry": float(sig.entry_price or 0),
                    "target": float(sig.target_price or 0),
                    "stop": float(sig.stop_loss or 0),
                    "qty": int((sig.meta or {}).get("qty") or 0) or 1,
                }
                if symbol not in subscribed:
                    stream.subscribe_trades(on_trade, symbol)
                    stream.subscribe_bars(on_bar, symbol)
                    subscribed.add(symbol)
            return updated

        signals = _load_signals()

        if not signals:
            self.stdout.write(self.style.WARNING("No OPEN ActiveSignal to track."))

        def refresh_subscriptions():
            while True:
                time.sleep(60)
                updated = _load_signals()
                signals.clear()
                signals.update(updated)

        def poll_prices():
            while True:
                time.sleep(60)
                for symbol in list(signals.keys()):
                    try:
                        price = get_latest_trade_price(symbol)
                        if price is None:
                            continue
                        price = float(price)
                    except Exception:
                        continue
                    if symbol not in first_poll_sent:
                        first_poll_sent.add(symbol)
                        _send_telegram_alert(
                            f"📡 Poll live {symbol} OK. Dernier prix {price:.4f}.",
                            allow_during_blackout=True,
                            category="tracker",
                        )
                    if heartbeat_minutes > 0:
                        if _cooldown_ok(last_heartbeat_ts, f"{symbol}:poll", heartbeat_minutes * 60):
                            _send_telegram_alert(
                                f"📊 {symbol} poll {price:.4f}",
                                allow_during_blackout=True,
                                category="tracker",
                            )
                    _check_targets(symbol, price)

        thread = threading.Thread(target=refresh_subscriptions, daemon=True)
        thread.start()

        poll_thread = threading.Thread(target=poll_prices, daemon=True)
        poll_thread.start()

        self.stdout.write(self.style.SUCCESS("Realtime tracker started."))
        stream.run()
