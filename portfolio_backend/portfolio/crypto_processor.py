from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yfinance as yfin

from .ml_engine.crypto_training import _build_crypto_features, _normalize_columns, _rsi


def _crypto_symbols() -> list[str]:
    raw = os.getenv('CRYPTO_SYMBOLS', '')
    symbols = [s.strip().upper() for s in raw.split(',') if s.strip()]
    return symbols or ['BTC-USD', 'ETH-USD', 'SOL-USD']


def _fetch_15m(symbol: str, period: str = '7d') -> pd.DataFrame:
    hist = yfin.Ticker(symbol).history(period=period, interval='15m')
    if hist is None or hist.empty:
        return pd.DataFrame()
    hist = hist.reset_index()
    return _normalize_columns(hist)


def _pct_change(series: pd.Series, periods: int = 1) -> float | None:
    if series is None or series.empty or len(series) <= periods:
        return None
    prev = float(series.iloc[-1 - periods])
    last = float(series.iloc[-1])
    if prev == 0:
        return None
    return (last - prev) / prev


def _btc_panic(btc_df: pd.DataFrame) -> bool:
    threshold = float(os.getenv('CRYPTO_BTC_PANIC_THRESHOLD', '0.05'))
    if btc_df is None or btc_df.empty:
        return False
    change = _pct_change(btc_df['close'], periods=4)  # ~1h on 15m
    if change is None:
        return False
    return change <= -abs(threshold)


def _drip_trigger(symbol_df: pd.DataFrame) -> bool:
    threshold = float(os.getenv('CRYPTO_DRIP_THRESHOLD', '0.03'))
    change = _pct_change(symbol_df['close'], periods=1)
    if change is None:
        return False
    return change <= -abs(threshold)


def _load_crypto_model() -> dict[str, Any] | None:
    model_path = Path(os.getenv('CRYPTO_MODEL_PATH', ''))
    if not model_path:
        return None
    try:
        payload = joblib.load(model_path)
        if isinstance(payload, dict) and payload.get('model'):
            return payload
    except Exception:
        return None
    return None


def scan_crypto_drip() -> list[dict[str, Any]]:
    symbols = _crypto_symbols()
    btc_df = _fetch_15m('BTC-USD')
    btc_panic = _btc_panic(btc_df)
    model_payload = _load_crypto_model()
    model = model_payload.get('model') if model_payload else None
    feature_cols = model_payload.get('features') if model_payload else None

    results: list[dict[str, Any]] = []
    for symbol in symbols:
        df = _fetch_15m(symbol)
        if df.empty:
            continue
        drip = _drip_trigger(df)
        if not drip:
            continue

        features = _build_crypto_features(df, btc_df=btc_df)
        last = features.tail(1)
        if last.empty:
            continue

        rsi_val = float(last.get('rsi_14', pd.Series([np.nan])).iloc[-1])
        oversold = rsi_val <= float(os.getenv('CRYPTO_RSI_OVERSOLD', '25'))

        score = None
        if model and feature_cols:
            sample = last.copy()
            for col in feature_cols:
                if col not in sample.columns:
                    sample[col] = 0.0
            X = sample[feature_cols].fillna(0).values
            try:
                proba = model.predict_proba(X)[0][1]
                score = float(proba)
            except Exception:
                score = None

        blocked = btc_panic and symbol != 'BTC-USD'
        results.append({
            'symbol': symbol,
            'price': float(last['close'].iloc[-1]),
            'drip': drip,
            'rsi': rsi_val,
            'oversold': oversold,
            'btc_panic': btc_panic,
            'blocked': blocked,
            'score': score,
            'price_to_vwap': float(last.get('price_to_vwap', pd.Series([0.0])).iloc[-1]),
        })

    return results
