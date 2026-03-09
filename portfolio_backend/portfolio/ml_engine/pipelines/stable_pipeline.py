from __future__ import annotations

"""Stable model pipeline orchestrator."""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from portfolio.ml_engine.config import config
from portfolio.ml_engine.data.market import fetch_history
from portfolio.ml_engine.export.onnx_exporter import OnnxExporter
from portfolio.ml_engine.export.push_model import push_model
from portfolio.ml_engine.features.technical import rsi, sma_ratio, volatility, volume_zscore
from portfolio.ml_engine.training.labeling import triple_barrier_labels
from portfolio.ml_engine.training.trainer import Trainer
from portfolio.ml_engine.training.deepseek_weighter import build_sample_weights
from portfolio.ml_engine.registry.model_registry import LocalModelRegistry

STABLE_FEATURES = [
    "rsi_14",
    "sma_ratio_10_50",
    "sma_ratio_20_50",
    "volatility_20",
    "volume_zscore_20",
    "return_20d",
    "spy_correlation",
    "dividend_yield",
    "sector_beta",
    "sentiment_score",
]


def _ensure_django() -> None:
    """Ensure Django settings loaded."""
    import os

    if not os.getenv("DJANGO_SETTINGS_MODULE"):
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "portfolio_backend.settings")
    import django

    django.setup()


def _setup_logging() -> None:
    """Configure file logging for training."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"training_STABLE_{datetime.utcnow().date().isoformat()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def _pipeline_factory() -> Pipeline:
    """Create the sklearn pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=400, max_depth=8, random_state=42, class_weight="balanced")),
    ])


def build_dataset(symbols: list[str]) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Build feature matrix and labels for stable universe."""
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    sample_symbols: list[str] = []
    sample_dates: list[str] = []

    spy = fetch_history("SPY", period="2y", interval="1d")
    if spy is None or spy.empty or "Close" not in spy.columns:
        return pd.DataFrame(), pd.Series(dtype=int), [], []

    for symbol in symbols:
        hist = fetch_history(symbol, period="2y", interval="1d")
        if hist is None or hist.empty or "Close" not in hist.columns:
            continue
        close = hist["Close"].astype(float)
        volume = hist.get("Volume", pd.Series(0.0, index=close.index)).astype(float)
        high = hist.get("High", close)
        low = hist.get("Low", close)

        rsi_14 = rsi(close, 14)
        sma_10_50 = sma_ratio(close, 10, 50)
        sma_20_50 = sma_ratio(close, 20, 50)
        vol_20 = volatility(close, 20)
        vol_z = volume_zscore(volume, 20)
        ret_20 = close.pct_change(20)
        spy_ret = spy["Close"].pct_change()
        spy_corr = close.pct_change().rolling(60).corr(spy_ret).fillna(0)

        _ensure_django()
        from portfolio.models import Stock
        from portfolio.ml_engine.collectors.news_rss import fetch_news_sentiment

        stock_row = Stock.objects.filter(symbol__iexact=symbol).values("dividend_yield", "sector").first() or {}
        dividend_yield = float(stock_row.get("dividend_yield") or 0.0)
        sector_beta = 0.0
        sentiment_score = float(fetch_news_sentiment(symbol).get("news_sentiment") or 0.0)

        label_series = triple_barrier_labels(close, high=high, low=low, up_pct=0.15, down_pct=0.07, max_days=20)

        for idx in range(len(close)):
            feat_row = {
                "rsi_14": float(rsi_14.iloc[idx]) if not pd.isna(rsi_14.iloc[idx]) else np.nan,
                "sma_ratio_10_50": float(sma_10_50.iloc[idx]) if not pd.isna(sma_10_50.iloc[idx]) else 1.0,
                "sma_ratio_20_50": float(sma_20_50.iloc[idx]) if not pd.isna(sma_20_50.iloc[idx]) else 1.0,
                "volatility_20": float(vol_20.iloc[idx]) if not pd.isna(vol_20.iloc[idx]) else np.nan,
                "volume_zscore_20": float(vol_z.iloc[idx]) if not pd.isna(vol_z.iloc[idx]) else 0.0,
                "return_20d": float(ret_20.iloc[idx]) if not pd.isna(ret_20.iloc[idx]) else np.nan,
                "spy_correlation": float(spy_corr.iloc[idx]) if not pd.isna(spy_corr.iloc[idx]) else 0.0,
                "dividend_yield": dividend_yield,
                "sector_beta": sector_beta,
                "sentiment_score": sentiment_score,
            }
            label_val = label_series.iloc[idx]
            if pd.isna(label_val):
                continue
            if any(pd.isna(v) for v in feat_row.values()):
                continue
            rows.append(feat_row)
            labels.append(int(label_val))
            sample_symbols.append(symbol)
            sample_dates.append(str(close.index[idx].date()))

    X = pd.DataFrame(rows)
    X = X[STABLE_FEATURES]
    y = pd.Series(labels)
    return X, y, sample_symbols, sample_dates


def run() -> None:
    """Execute stable training pipeline."""
    _setup_logging()
    logging.info("Training started")
    _ensure_django()
    from portfolio.models import Stock

    symbols = [s.symbol for s in Stock.objects.all().order_by("symbol") if s.symbol]
    X, y, sample_symbols, sample_dates = build_dataset(symbols)
    if X.empty:
        raise RuntimeError("No training samples available for stable pipeline")

    trainer = Trainer(_pipeline_factory)
    weights = build_sample_weights(sample_symbols, sample_dates, X.to_dict("records"), y.tolist())
    result = trainer.fit(X, y, sample_weight=np.array(weights))

    exporter = OnnxExporter()
    output_path = Path("/app/portfolio/ml_engine/stable_brain_v1.onnx")
    onnx_path = exporter.export(result, output_path, "STABLE", expected_feature_count=len(STABLE_FEATURES))

    registry = LocalModelRegistry(Path("/app/portfolio/ml_engine/registry"))
    meta = json.loads(onnx_path.with_suffix(".json").read_text())
    registry.register("stable", onnx_path, meta)

    if config.integration.auto_push:
        push_model("stable", onnx_path, meta)

    logging.info("Training completed")


if __name__ == "__main__":
    run()
