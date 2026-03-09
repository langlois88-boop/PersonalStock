from __future__ import annotations

"""Crypto model pipeline orchestrator."""

import logging
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from portfolio.ml_engine.config import config
from portfolio.ml_engine.data.market import fetch_history
from portfolio.ml_engine.export.onnx_exporter import OnnxExporter
from portfolio.ml_engine.export.push_model import push_model
from portfolio.ml_engine.features.technical import rsi, rubber_band_index
from portfolio.ml_engine.training.labeling import forward_return_labels
from portfolio.ml_engine.training.trainer import Trainer
from portfolio.ml_engine.registry.model_registry import LocalModelRegistry

CRYPTO_FEATURES = [
    "return_1",
    "rsi_14",
    "rubber_band_index",
    "price_to_vwap",
    "volatility_spike",
    "btc_correlation",
]


def _setup_logging() -> None:
    """Configure file logging for training."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"training_CRYPTO_{datetime.utcnow().date().isoformat()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def _pipeline_factory() -> Pipeline:
    """Create the sklearn pipeline."""
    return Pipeline([
        ("scaler", RobustScaler()),
        ("classifier", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")),
    ])


def _vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute VWAP using cumulative volume."""
    typical = (high + low + close) / 3.0
    pv = typical * volume
    return pv.cumsum() / volume.replace(0, np.nan).cumsum()


def build_dataset(symbols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix and labels for crypto universe."""
    frames = []
    btc = fetch_history("BTC-USD", period="60d", interval="1h")
    btc_close = btc["Close"] if btc is not None and "Close" in btc.columns else None

    for symbol in symbols:
        hist = fetch_history(symbol, period="60d", interval="1h")
        if hist is None or hist.empty or "Close" not in hist.columns:
            continue
        close = hist["Close"].astype(float)
        high = hist.get("High", close)
        low = hist.get("Low", close)
        volume = hist.get("Volume", pd.Series(0.0, index=close.index)).astype(float)

        return_1 = close.pct_change()
        rsi_14 = rsi(close, 14)
        rbi = rubber_band_index(close, 20)
        vwap = _vwap(high, low, close, volume)
        price_to_vwap = close / vwap.replace(0, np.nan)
        vol_mean = volume.rolling(20).mean().replace(0, np.nan)
        volatility_spike = (volume / vol_mean).fillna(0)

        if btc_close is not None:
            btc_ret = btc_close.pct_change()
            btc_corr = return_1.rolling(30).corr(btc_ret).fillna(0)
        else:
            btc_corr = pd.Series(0.0, index=close.index)

        df = pd.DataFrame({
            "return_1": return_1,
            "rsi_14": rsi_14,
            "rubber_band_index": rbi,
            "price_to_vwap": price_to_vwap,
            "volatility_spike": volatility_spike,
            "btc_correlation": btc_corr,
        }).replace([np.inf, -np.inf], np.nan).fillna(0)

        labels = forward_return_labels(close, horizon=8, target_pct=0.02)
        df["label"] = labels
        frames.append(df)

    if not frames:
        return pd.DataFrame(), pd.Series(dtype=int)

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.dropna(subset=["label"])
    X = dataset[CRYPTO_FEATURES]
    y = dataset["label"].astype(int)
    return X, y


def run() -> None:
    """Execute crypto training pipeline."""
    _setup_logging()
    logging.info("Training started")

    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
    X, y = build_dataset(symbols)
    if X.empty:
        raise RuntimeError("No training samples available for crypto pipeline")

    trainer = Trainer(_pipeline_factory)
    result = trainer.fit(X, y)

    exporter = OnnxExporter()
    output_path = Path("/app/portfolio/ml_engine/crypto_brain_v1.onnx")
    onnx_path = exporter.export(result, output_path, "CRYPTO", expected_feature_count=len(CRYPTO_FEATURES))

    registry = LocalModelRegistry(Path("/app/portfolio/ml_engine/registry"))
    meta = json.loads(onnx_path.with_suffix(".json").read_text())
    registry.register("crypto", onnx_path, meta)

    if config.integration.auto_push:
        push_model("crypto", onnx_path, meta)

    logging.info("Training completed")


if __name__ == "__main__":
    run()
