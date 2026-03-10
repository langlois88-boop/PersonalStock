from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as rt
from django.core.management.base import BaseCommand

from portfolio.ml_engine.data.market import fetch_history
from portfolio.ml_engine.backtest_engine import _prepare_stable_features, _prepare_penny_features
from portfolio.ml_engine.pipelines.stable_pipeline import STABLE_FEATURES_CORE
from portfolio.ml_engine.pipelines.penny_pipeline import PENNY_FEATURES_CORE
from portfolio.tasks import _send_telegram_message


class Command(BaseCommand):
    help = "Scan canadian_universe.csv for dip anomalies and log ML signals."

    def handle(self, *args, **options):
        universe_path = _resolve_universe_path()
        if not universe_path:
            self.stdout.write("Universe CSV not found.")
            return

        symbols = _load_symbols(universe_path)
        if not symbols:
            self.stdout.write("No symbols to scan.")
            return

        limit = int(os.getenv("SCANNER_LIMIT", "0"))
        if limit > 0:
            symbols = symbols[:limit]

        stable_session = _load_session(os.getenv("STABLE_ONNX_PATH", "/app/portfolio/ml_engine/stable_brain_v1.onnx"))
        penny_session = _load_session(os.getenv("PENNY_ONNX_PATH", "/app/portfolio/ml_engine/scout_brain_v1.onnx"))

        log_path = Path(os.getenv("SCANNER_LOG_PATH", "/app/logs/market_scanner.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)

        alerts = []
        for symbol in symbols:
            try:
                alert = _scan_symbol(symbol, stable_session, penny_session)
            except Exception:
                alert = None
            if not alert:
                continue
            alerts.append(alert)
            _append_log(log_path, alert)

        if alerts:
            message = _format_message(alerts)
            _send_telegram_message(message)
            self.stdout.write(f"Scanner complete: {len(alerts)} alerts")
        else:
            self.stdout.write("Scanner complete: no alerts")


def _resolve_universe_path() -> str | None:
    candidates = [
        os.getenv("CANADIAN_UNIVERSE_PATH", ""),
        "/app/portfolio/ml_engine/data/canadian_universe.csv",
        "/app/data/canadian_universe.csv",
        "/mnt/AppStorage/apps/PersonalStock/data/canadian_universe.csv",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _load_symbols(path: str) -> list[str]:
    df = pd.read_csv(path)
    column = "symbol" if "symbol" in df.columns else df.columns[0]
    symbols = [str(s).strip().upper() for s in df[column].dropna().tolist()]
    return [s for s in symbols if s]


def _load_session(path: str) -> rt.InferenceSession | None:
    if not path or not os.path.exists(path):
        return None
    return rt.InferenceSession(path)


def _scan_symbol(
    symbol: str,
    stable_session: rt.InferenceSession | None,
    penny_session: rt.InferenceSession | None,
) -> dict[str, str | float] | None:
    hist = fetch_history(symbol, period="3mo", interval="1d")
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None

    close = hist["Close"].astype(float)
    if len(close) < 2:
        return None

    prev_close = float(close.iloc[-2])
    last_close = float(close.iloc[-1])
    if prev_close <= 0:
        return None

    change_pct = (last_close - prev_close) / prev_close * 100
    volume = hist.get("Volume", pd.Series(0.0, index=close.index)).astype(float)
    avg_volume = float(volume.tail(10).mean()) if len(volume) else 0.0

    dip_to = float(os.getenv("SCANNER_DIP_TO", "-3.5"))
    dip_v = float(os.getenv("SCANNER_DIP_V", "-12.0"))
    min_vol_to = float(os.getenv("SCANNER_MIN_VOLUME_TO", "200000"))
    min_vol_v = float(os.getenv("SCANNER_MIN_VOLUME_V", "100000"))

    if symbol.endswith(".TO"):
        if avg_volume < min_vol_to or change_pct > dip_to:
            return None
        session = stable_session
        feature_list = STABLE_FEATURES_CORE
        mode = "stable"
    elif symbol.endswith(".V"):
        if avg_volume < min_vol_v or change_pct > dip_v:
            return None
        session = penny_session
        feature_list = PENNY_FEATURES_CORE
        mode = "penny"
    else:
        return None

    if session is None:
        return None

    score = _predict_score(session, hist, feature_list, mode)
    min_score = float(os.getenv("SCANNER_MIN_SCORE", "0.6"))
    if score < min_score:
        return None

    return {
        "time": datetime.now().isoformat(),
        "symbol": symbol,
        "dip": f"{change_pct:.2f}%",
        "vol": f"{avg_volume:.0f}",
        "score": f"{score:.2f}",
    }


def _predict_score(
    session: rt.InferenceSession,
    hist: pd.DataFrame,
    feature_list: list[str],
    mode: str,
) -> float:
    if mode == "stable":
        df = _prepare_stable_features(hist)
    else:
        df = _prepare_penny_features(hist)

    row = df[feature_list].tail(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vector = row.values.astype(np.float32)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: vector})

    if len(outputs) >= 2:
        probs = outputs[1]
        if probs is not None and len(probs):
            return float(probs[0][1])
    label = outputs[0][0] if outputs and len(outputs[0]) else 0.0
    return float(label)


def _append_log(path: Path, alert: dict[str, str | float]) -> None:
    line = f"{alert['time']}|{alert['symbol']}|{alert['dip']}|{alert['vol']}|{alert['score']}\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line)


def _format_message(alerts: list[dict[str, str | float]]) -> str:
    header = "🔍 *SCAN DU MATIN*\n"
    lines = [header]
    for item in alerts[:12]:
        lines.append(f"• {item['symbol']} dip {item['dip']} | score {item['score']}")
    if len(alerts) > 12:
        lines.append(f"… +{len(alerts) - 12} autres")
    return "\n".join(lines)
