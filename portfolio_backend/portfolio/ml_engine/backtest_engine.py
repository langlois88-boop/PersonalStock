from __future__ import annotations

"""Simple ONNX backtest with max drawdown."""

import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import onnxruntime as rt
from dotenv import load_dotenv
import requests

from portfolio.ml_engine.data.market import fetch_history
from portfolio.ml_engine.features.technical import rsi, sma_ratio, volatility, volume_zscore, rvol
from portfolio.ml_engine.pipelines.stable_pipeline import STABLE_FEATURES_CORE
from portfolio.ml_engine.pipelines.penny_pipeline import PENNY_FEATURES_CORE


def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    if cumulative_returns.size == 0:
        return 0.0
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    return float(np.min(drawdown))


def save_backtest_report(
    symbol: str,
    ret_s: float,
    ret_m: float,
    mdd_s: float,
    mdd_m: float,
    signals: int,
) -> None:
    report_path = Path(os.getenv("BACKTEST_LOG_PATH", "/app/logs/backtest_history.csv"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    entry = pd.DataFrame([
        {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "symbol": symbol,
            "strat_return": round(ret_s, 4),
            "market_return": round(ret_m, 4),
            "strat_mdd": round(mdd_s, 4),
            "market_mdd": round(mdd_m, 4),
            "buy_signals": signals,
        }
    ])
    entry.to_csv(report_path, mode="a", header=not report_path.exists(), index=False)
    print(f"Saved report to {report_path}")


def _load_env() -> None:
    for path in ("/app/.env", "/app/portfolio_backend/.env"):
        try:
            load_dotenv(path, override=False)
        except Exception:
            continue


def send_telegram_summary(report_data: list[dict[str, float | str]]) -> None:
    _load_env()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram config missing; skipping notification")
        return

    message_lines = ["✅ *Weekly Pipeline Completed*", ""]
    for item in report_data:
        message_lines.append(f"🔹 *{item['symbol']}*")
        message_lines.append(f"  • Strat: {item['ret_s']:.2f}% (MDD: {item['mdd_s']:.2f}%)")
        message_lines.append(f"  • Market: {item['ret_m']:.2f}%")
        message_lines.append(f"  • Signals: {int(item['signals'])}")
        message_lines.append("")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(
            url,
            json={"chat_id": chat_id, "text": "\n".join(message_lines), "parse_mode": "Markdown"},
            timeout=20,
        )
    except Exception as exc:
        print(f"Telegram send failed: {exc}")


def _prepare_stable_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    volume = df.get("Volume", pd.Series(0.0, index=close.index)).astype(float)

    df = df.copy()
    df["rsi_14"] = rsi(close, 14)
    df["sma_ratio_10_50"] = sma_ratio(close, 10, 50)
    df["sma_ratio_20_50"] = sma_ratio(close, 20, 50)
    df["volatility_20"] = volatility(close, 20)
    df["volume_zscore_20"] = volume_zscore(volume, 20)
    df["return_20d"] = close.pct_change(20)

    spy = fetch_history("SPY", period="2y", interval="1d")
    if spy is not None and not spy.empty and "Close" in spy.columns:
        spy_close = spy["Close"].reindex(close.index).ffill().bfill()
        df["spy_correlation"] = close.pct_change().rolling(60, min_periods=60).corr(spy_close.pct_change())
    else:
        df["spy_correlation"] = 0.0

    if "sector_beta" in STABLE_FEATURES_CORE:
        df["sector_beta"] = 0.0

    for col in STABLE_FEATURES_CORE:
        if col not in df.columns:
            df[col] = 0.0
    return df


def _prepare_penny_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    volume = df.get("Volume", pd.Series(0.0, index=close.index)).astype(float)

    df = df.copy()
    df["volume_zscore_20"] = volume_zscore(volume, 20)
    df["rvol_20"] = rvol(volume, 20)
    df["return_5d"] = close.pct_change(5)

    for col in PENNY_FEATURES_CORE:
        if col not in df.columns:
            df[col] = 0.0
    return df


def run_backtest(
    symbol: str,
    model_path: Path,
    feature_list: list[str],
    days: int,
    mode: str,
) -> dict[str, float | str] | None:
    df = fetch_history(symbol, period="1y", interval="1d")
    if df is None or df.empty:
        print(f"No data for {symbol}")
        return None

    if mode == "stable":
        df = _prepare_stable_features(df)
    else:
        df = _prepare_penny_features(df)

    X = df[feature_list].tail(days).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None

    sess = rt.InferenceSession(str(model_path))
    input_name = sess.get_inputs()[0].name

    preds = []
    for _, row in X.iterrows():
        vector = row.values.astype(np.float32).reshape(1, -1)
        pred = sess.run(None, {input_name: vector})[0]
        preds.append(int(pred[0]))

    price_data = df["Close"].tail(days).values
    daily_returns = pd.Series(price_data).pct_change().fillna(0).values

    strat_returns = []
    for i in range(len(preds) - 1):
        strat_returns.append(daily_returns[i + 1] if preds[i] == 1 else 0.0)

    cum_strat = np.cumprod(1 + np.array(strat_returns))
    cum_market = np.cumprod(1 + daily_returns)

    ret_s = (cum_strat[-1] - 1) * 100 if len(cum_strat) else 0.0
    ret_m = (cum_market[-1] - 1) * 100 if len(cum_market) else 0.0
    mdd_s = calculate_max_drawdown(cum_strat) * 100
    mdd_m = calculate_max_drawdown(cum_market) * 100

    print(f"\n===== BACKTEST: {symbol} =====")
    print(f"Return Strategy: {ret_s:.2f}%")
    print(f"Return Market:   {ret_m:.2f}%")
    print(f"Max Drawdown Strat:  {mdd_s:.2f}%")
    print(f"Max Drawdown Market: {mdd_m:.2f}%")
    signals = sum(preds)
    print(f"Buy signals: {signals} / {len(preds)}")

    save_backtest_report(symbol, ret_s, ret_m, mdd_s, mdd_m, signals)
    return {
        "symbol": symbol,
        "ret_s": ret_s,
        "ret_m": ret_m,
        "mdd_s": mdd_s,
        "mdd_m": mdd_m,
        "signals": signals,
    }


if __name__ == "__main__":
    stable_model = Path(os.getenv("STABLE_ONNX_PATH", "/app/portfolio/ml_engine/stable_brain_v1.onnx"))
    penny_model = Path(os.getenv("PENNY_ONNX_PATH", "/app/portfolio/ml_engine/scout_brain_v1.onnx"))

    reports: list[dict[str, float | str]] = []
    report = run_backtest("RY.TO", stable_model, STABLE_FEATURES_CORE, days=60, mode="stable")
    if report:
        reports.append(report)
    report = run_backtest("BONK-USD", penny_model, PENNY_FEATURES_CORE, days=60, mode="penny")
    if report:
        reports.append(report)

    if reports:
        send_telegram_summary(reports)
