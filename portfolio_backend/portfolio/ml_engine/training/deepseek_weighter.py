from __future__ import annotations

"""DeepSeek sample weighting helper."""

import json
from typing import Iterable

import requests

from portfolio.ml_engine.config import config


def get_deepseek_confidence(symbol: str, date: str, features: dict[str, float], raw_label: int) -> float:
    """Request a confidence score from DeepSeek.

    Args:
        symbol: Ticker symbol.
        date: Date string.
        features: Feature snapshot.
        raw_label: Raw label.

    Returns:
        Confidence score between 0 and 1.
    """
    if not config.integration.deepseek_url:
        return 1.0
    prompt = (
        "Stock: {symbol}, Date: {date}, Raw label: {label}\n"
        "Features: RSI={rsi:.1f}, SMA_ratio={sma_ratio:.3f}, "
        "Vol_zscore={vol_z:.2f}, Return={ret:.3f}\n"
        "Sentiment: {sentiment:.2f}\n\n"
        "Given these indicators, rate your confidence 0.0-1.0 that the label "
        "({label}) is correct. Respond with ONLY a float number, nothing else."
    ).format(
        symbol=symbol,
        date=date,
        label=raw_label,
        rsi=features.get("rsi_14", 0.0),
        sma_ratio=features.get("sma_ratio_10_50", features.get("sma_ratio_10_20", 1.0)),
        vol_z=features.get("volume_zscore_20", 0.0),
        ret=features.get("return_20d", features.get("return_5d", 0.0)),
        sentiment=features.get("sentiment_score", 0.0),
    )
    try:
        resp = requests.post(
            config.integration.deepseek_url,
            json={"prompt": prompt, "model": config.integration.deepseek_model},
            timeout=20,
        )
        resp.raise_for_status()
        text = resp.text
        if resp.headers.get("content-type", "").startswith("application/json"):
            text = resp.json().get("response", "")
        value = float(str(text).strip().split()[0])
        return max(0.0, min(1.0, value))
    except Exception:
        return 1.0


def build_sample_weights(
    symbols: Iterable[str],
    dates: Iterable[str],
    features: Iterable[dict[str, float]],
    labels: Iterable[int],
) -> list[float]:
    """Build sample weights from DeepSeek.

    Args:
        symbols: Symbols sequence.
        dates: Dates sequence.
        features: Feature dicts sequence.
        labels: Raw labels sequence.

    Returns:
        List of float weights.
    """
    if not config.integration.auto_deepseek_weight:
        return [1.0 for _ in labels]
    weights: list[float] = []
    for symbol, date, feat, label in zip(symbols, dates, features, labels):
        weights.append(get_deepseek_confidence(symbol, date, feat, label))
    return weights
