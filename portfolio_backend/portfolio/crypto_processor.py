from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yfinance as yfin
import requests

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


def _btc_panic(symbol_df: pd.DataFrame, btc_df: pd.DataFrame) -> tuple[bool, float | None]:
    threshold = float(os.getenv('CRYPTO_BTC_PANIC_THRESHOLD', '0.05'))
    min_corr = float(os.getenv('CRYPTO_BTC_MIN_CORR', '0.35'))
    if btc_df is None or btc_df.empty or symbol_df is None or symbol_df.empty:
        return False, None
    change = _pct_change(btc_df['close'], periods=4)  # ~1h on 15m
    if change is None:
        return False, None
    try:
        sym_ret = symbol_df['close'].pct_change().dropna()
        btc_ret = btc_df['close'].pct_change().dropna()
        aligned = pd.concat([sym_ret, btc_ret], axis=1).dropna()
        corr = float(aligned.iloc[:, 0].rolling(30).corr(aligned.iloc[:, 1]).iloc[-1]) if len(aligned) >= 30 else None
    except Exception:
        corr = None
    if corr is None or corr < min_corr:
        return False, corr
    return change <= -abs(threshold), corr


_CRYPTO_PANIC_CACHE: dict[str, Any] = {'ts': 0.0, 'verdict': None, 'reason': None}


def _crypto_panic_verdict(btc_df: pd.DataFrame) -> str:
    global _CRYPTO_PANIC_CACHE
    if _CRYPTO_PANIC_CACHE.get('verdict') and (time.time() - float(_CRYPTO_PANIC_CACHE.get('ts') or 0)) < 300:
        return str(_CRYPTO_PANIC_CACHE.get('verdict'))
    change = _pct_change(btc_df['close'], periods=4) if btc_df is not None and not btc_df.empty else None
    base_url = (
        os.getenv('DANAS_CHAT_BASE_URL')
        or os.getenv('OLLAMA_CHAT_BASE_URL')
        or os.getenv('OLLAMA_BASE_URL')
        or ''
    ).strip().rstrip('/')
    if base_url and '/v1' not in base_url:
        base_url = f"{base_url}/v1"
    if not base_url:
        return 'SYSTEMIC'
    model = os.getenv('OLLAMA_MODEL', 'deepseek-r1')
    prompt = (
        "CRYPTO PANIC CHECK\n"
        f"BTC 1H change: {round((change or 0) * 100, 2)}%\n"
        "Question: Est-ce une panique systémique (SYSTEMIC) ou une rotation (ROTATION) ?\n"
        "Réponds uniquement par SYSTEMIC ou ROTATION."
    )
    try:
        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': 'Réponds en français mais un seul mot SYSTEMIC ou ROTATION.'},
                {'role': 'user', 'content': prompt},
            ],
            'stream': False,
        }
        resp = requests.post(f"{base_url}/chat/completions", json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        text = ((data.get('choices') or [{}])[0].get('message') or {}).get('content') or ''
        verdict = 'ROTATION' if 'ROTATION' in text.upper() else 'SYSTEMIC'
        _CRYPTO_PANIC_CACHE = {'ts': time.time(), 'verdict': verdict, 'reason': text.strip()[:300]}
        return verdict
    except Exception:
        return 'SYSTEMIC'


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
    panic_verdict = _crypto_panic_verdict(btc_df)
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

        btc_panic, btc_corr = _btc_panic(df, btc_df)
        blocked = btc_panic and panic_verdict == 'SYSTEMIC' and symbol != 'BTC-USD'
        results.append({
            'symbol': symbol,
            'price': float(last['close'].iloc[-1]),
            'drip': drip,
            'rsi': rsi_val,
            'oversold': oversold,
            'btc_panic': btc_panic,
            'panic_verdict': panic_verdict,
            'btc_corr': btc_corr,
            'blocked': blocked,
            'score': score,
            'price_to_vwap': float(last.get('price_to_vwap', pd.Series([0.0])).iloc[-1]),
        })

    return results
