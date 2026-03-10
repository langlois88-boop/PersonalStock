from __future__ import annotations

"""DeepSeek sample weighting helper."""

from typing import Iterable

import os

from portfolio.services.deepseek_analyst import get_training_confidence


def _infer_universe(features: dict[str, float]) -> str:
    if "return_5d" in features or "rubber_band_20" in features or "rvol_20" in features:
        return "PENNY"
    return "BLUECHIP"


def build_sample_weights(
    symbols: Iterable[str],
    dates: Iterable[str],
    features: Iterable[dict[str, float]],
    labels: Iterable[int],
    universe: str | None = None,
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
    if (os.getenv("DEEPSEEK_WEIGHTING_ENABLED") or "").lower() not in {"1", "true", "yes", "y"}:
        return [1.0 for _ in labels]
    weights: list[float] = []
    for symbol, date, feat, label in zip(symbols, dates, features, labels):
        resolved = universe or _infer_universe(feat)
        weights.append(get_training_confidence(symbol, date, feat, label, universe=resolved))
    return weights
