from __future__ import annotations

"""Diagnostics for label balance and feature sanity checks."""

import os
from typing import Callable, Optional

import numpy as np
import pandas as pd

from portfolio.ml_engine.data.market import fetch_history
from portfolio.ml_engine.features.technical import rsi
from portfolio.ml_engine.pipelines.penny_pipeline import build_dataset as build_penny_dataset
from portfolio.ml_engine.pipelines.penny_pipeline import _get_penny_symbols
from portfolio.ml_engine.pipelines.stable_pipeline import build_dataset as build_stable_dataset
from portfolio.ml_engine.pipelines.stable_pipeline import _get_stable_symbols


def _ensure_django() -> None:
    if not os.getenv("DJANGO_SETTINGS_MODULE"):
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "portfolio_backend.settings")
    import django

    django.setup()


def _rsi_simple(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("rsi_simple")


def _stationarity_check(series: pd.Series) -> str:
    series = series.dropna()
    if series.empty:
        return "n/a"
    try:
        from statsmodels.tsa.stattools import adfuller  # type: ignore

        result = adfuller(series.values, autolag="AIC")
        pval = float(result[1]) if len(result) > 1 else np.nan
        return f"adf_p={pval:.4f}"
    except Exception:
        return f"autocorr_1={series.autocorr(lag=1):.3f}"


def _summarize(name: str, X: pd.DataFrame, y: pd.Series) -> None:
    pos = int(y.sum())
    neg = int(len(y) - pos)
    balance = float(y.mean()) if len(y) else 0.0
    print(f"[{name}] samples={len(y)} pos={pos} neg={neg} balance={balance:.3f}")


def _rsi_compare(symbol: str) -> None:
    hist = fetch_history(symbol, period="2y", interval="1d")
    if hist is None or hist.empty or "Close" not in hist.columns:
        print(f"[RSI] {symbol}: no history")
        return
    close = hist["Close"].astype(float)
    wilder = rsi(close, 14)
    simple = _rsi_simple(close, 14)
    diff = (wilder - simple).abs().dropna()
    if diff.empty:
        print(f"[RSI] {symbol}: insufficient data")
        return
    print(f"[RSI] {symbol}: mean_abs_diff={diff.tail(100).mean():.4f}")
    print(f"[Stationarity] {symbol}: {_stationarity_check(close.pct_change())}")


def _run_for(name: str, build_fn: Callable[[list[str]], tuple[pd.DataFrame, pd.Series, list[str], list[str]]], symbols: list[str]) -> None:
    X, y, sample_symbols, _ = build_fn(symbols)
    _summarize(name, X, y)
    if sample_symbols:
        _rsi_compare(sample_symbols[0])


def main() -> None:
    _ensure_django()
    print("Diagnostics start")
    _run_for("STABLE", build_stable_dataset, _get_stable_symbols())
    _run_for("PENNY", build_penny_dataset, _get_penny_symbols())
    print("Diagnostics complete")


if __name__ == "__main__":
    main()
