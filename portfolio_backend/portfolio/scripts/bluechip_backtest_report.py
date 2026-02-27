import os
import json
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv

from portfolio.ml_engine.engine.data_fusion import DataFusionEngine
from portfolio.ml_engine.backtester import AIBacktester, get_model_path, load_or_train_model
from portfolio.tasks import _news_sentiment_score


DEFAULT_TICKERS = [
    "ATD.TO", "DOL.TO", "L.TO", "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CNR.TO", "CP.TO", "TFII.TO",
    "ENB.TO", "TRP.TO", "FTS.TO", "CSU.TO", "BN.TO", "GIB-A.TO", "T.TO", "BCE.TO", "SAP.TO", "MRU.TO",
]


def _trend_label(series: pd.Series) -> dict:
    if series is None or series.empty:
        return {"trend": "N/A", "return_2m": None, "ma20": None, "ma50": None}
    closes = series.dropna()
    if closes.empty:
        return {"trend": "N/A", "return_2m": None, "ma20": None, "ma50": None}
    tail = closes.tail(60)
    ret = None
    if len(tail) >= 2:
        ret = float((tail.iloc[-1] - tail.iloc[0]) / tail.iloc[0] * 100)
    ma20 = float(closes.rolling(20, min_periods=10).mean().iloc[-1]) if len(closes) >= 10 else None
    ma50 = float(closes.rolling(50, min_periods=20).mean().iloc[-1]) if len(closes) >= 20 else None
    trend = "Neutral"
    if ret is not None and ma20 is not None and ma50 is not None:
        if ret > 2 and ma20 >= ma50:
            trend = "Haussier"
        elif ret < -2 and ma20 <= ma50:
            trend = "Baissier"
    return {"trend": trend, "return_2m": ret, "ma20": ma20, "ma50": ma50}


def run_report(tickers: list[str] | None = None, lookback_days: int = 730) -> list[dict]:
    tickers = tickers or DEFAULT_TICKERS
    results = []
    model_path = get_model_path("BLUECHIP")
    for symbol in tickers:
        symbol = symbol.strip().upper()
        if not symbol:
            continue
        try:
            fusion = DataFusionEngine(symbol, fast_mode=False)
            frame = fusion.fuse_all()
        except Exception:
            frame = None
        if frame is None or frame.empty:
            results.append({
                "symbol": symbol,
                "status": "no_data",
            })
            continue

        model_payload = load_or_train_model(frame, model_path=model_path)
        backtester = AIBacktester(frame, model_payload, symbol=symbol)
        backtest = backtester.run_simulation(lookback_days=lookback_days)

        close_col = None
        for name in ("Close", "close"):
            if name in frame.columns:
                close_col = name
                break
        close_series = pd.Series(frame[close_col]).astype(float) if close_col else None
        trend = _trend_label(close_series)

        sentiment, _ = _news_sentiment_score(symbol, days=60)
        results.append({
            "symbol": symbol,
            "status": "ok",
            "win_rate": round(float(backtest.win_rate), 2),
            "sharpe": round(float(backtest.sharpe_ratio), 2),
            "max_drawdown": round(float(backtest.max_drawdown) * 100, 2),
            "total_return_pct": round(float(backtest.total_return_pct), 2),
            "trend_2m": trend.get("trend"),
            "return_2m_pct": None if trend.get("return_2m") is None else round(float(trend.get("return_2m")), 2),
            "ma20": None if trend.get("ma20") is None else round(float(trend.get("ma20")), 2),
            "ma50": None if trend.get("ma50") is None else round(float(trend.get("ma50")), 2),
            "news_sentiment_2m": round(float(sentiment), 3),
        })

    return results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(base_dir))
    env_path = os.path.join(os.path.dirname(project_root), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)

    data = run_report()
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(os.path.dirname(project_root), 'logs')
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, f'bluechip_backtest_report_{ts}.json')
    csv_path = os.path.join(out_dir, f'bluechip_backtest_report_{ts}.csv')

    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    print(df.to_string(index=False))
