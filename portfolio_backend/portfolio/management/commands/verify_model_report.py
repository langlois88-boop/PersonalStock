from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from ...alpaca_data import get_intraday_context, get_latest_bid_ask_spread_pct
from ...ml_engine.backtester import (
    FEATURE_COLUMNS,
    TRIPLE_BARRIER_ATR_MULT_DOWN,
    TRIPLE_BARRIER_ATR_MULT_UP,
    TRIPLE_BARRIER_USE_ATR,
    get_model_path,
)
from ...ml_engine.engine.data_fusion import DataFusionEngine


class Command(BaseCommand):
    help = "Generate a verification report for model features, ATR barriers, and bid-ask spread availability."

    def add_arguments(self, parser):
        parser.add_argument("--symbol", type=str, default=os.getenv("VERIFY_REPORT_SYMBOL", "SPY"))
        parser.add_argument("--universe", type=str, default=os.getenv("VERIFY_REPORT_UNIVERSE", "BLUECHIP"))

    def handle(self, *args, **options):
        symbol = (options.get("symbol") or "SPY").strip().upper()
        universe = (options.get("universe") or "BLUECHIP").strip().upper()

        report = {
            "timestamp": timezone.now().isoformat(),
            "symbol": symbol,
            "universe": universe,
            "feature_columns": list(FEATURE_COLUMNS),
            "atr_barriers": {
                "enabled": TRIPLE_BARRIER_USE_ATR,
                "atr_mult_up": TRIPLE_BARRIER_ATR_MULT_UP,
                "atr_mult_down": TRIPLE_BARRIER_ATR_MULT_DOWN,
            },
            "data_fusion": {},
            "intraday": {},
            "model": {},
        }

        try:
            engine = DataFusionEngine(symbol)
            df = engine.fuse_all()
            if df is None or df.empty:
                report["data_fusion"] = {"status": "empty"}
            else:
                last_row = df.tail(1).iloc[0]
                report["data_fusion"] = {
                    "status": "ok",
                    "rows": int(len(df)),
                    "has_bid_ask_spread_pct": "bid_ask_spread_pct" in df.columns,
                    "last_bid_ask_spread_pct": float(last_row.get("bid_ask_spread_pct") or 0.0),
                }
        except Exception as exc:
            report["data_fusion"] = {"status": "error", "error": str(exc)}

        try:
            intraday_ctx = get_intraday_context(symbol, minutes=60, rvol_window=20)
            report["intraday"] = {
                "status": "ok" if intraday_ctx else "empty",
                "has_bid_ask_spread_pct": bool(intraday_ctx and "bid_ask_spread_pct" in intraday_ctx),
                "bid_ask_spread_pct": float((intraday_ctx or {}).get("bid_ask_spread_pct") or 0.0),
            }
        except Exception as exc:
            report["intraday"] = {"status": "error", "error": str(exc)}

        try:
            model_path = get_model_path(universe)
            payload = None
            if model_path.exists():
                payload = joblib.load(model_path)
            report["model"] = {
                "path": str(model_path),
                "exists": bool(model_path.exists()),
                "has_payload": bool(payload),
                "features": payload.get("features") if payload else [],
                "has_bid_ask_spread_pct": bool(payload and "bid_ask_spread_pct" in (payload.get("features") or [])),
            }
        except Exception as exc:
            report["model"] = {"status": "error", "error": str(exc)}

        try:
            spread = get_latest_bid_ask_spread_pct(symbol)
            report["bid_ask_snapshot"] = {
                "status": "ok" if spread is not None else "empty",
                "bid_ask_spread_pct": float(spread or 0.0),
            }
        except Exception as exc:
            report["bid_ask_snapshot"] = {"status": "error", "error": str(exc)}

        report_dir = Path(os.getenv("TRADING_JOURNAL_DIR", os.path.join(str(getattr(settings, "BASE_DIR", "")), "reports")))
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"verification_report_{symbol}_{timezone.now().date().isoformat()}.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))

        self.stdout.write(str(report_path))
