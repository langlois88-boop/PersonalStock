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

from .ml_engine.crypto_training import (
    CRYPTO_MODEL_PATH,
    CRYPTO_SWING_MODEL_PATH,
    _build_crypto_features,
    _build_crypto_swing_features,
    _normalize_columns,
    _rsi,
    fetch_crypto_history_daily,
)


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
    raw_path = os.getenv('CRYPTO_MODEL_PATH', '').strip()
    model_path = Path(raw_path) if raw_path else CRYPTO_MODEL_PATH
    if not model_path:
        return None
    if not model_path.exists():
        return None
    try:
        payload = joblib.load(model_path)
        if isinstance(payload, dict) and payload.get('model'):
            return payload
    except Exception:
        return None
    return None


def scan_crypto_drip(symbols: list[str] | None = None) -> list[dict[str, Any]]:
    symbols = [s.strip().upper() for s in (symbols or []) if s and str(s).strip()]
    if not symbols:
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


def _load_crypto_swing_model() -> dict[str, Any] | None:
    """Même logique que _load_crypto_model, chemin séparé (voir
    crypto_training.py::CRYPTO_SWING_MODEL_PATH -- ne partage jamais de
    fichier ni de clé d'env avec le pipeline intraday)."""
    raw_path = os.getenv('CRYPTO_SWING_MODEL_PATH', '').strip()
    model_path = Path(raw_path) if raw_path else CRYPTO_SWING_MODEL_PATH
    if not model_path or not model_path.exists():
        return None
    try:
        payload = joblib.load(model_path)
        if isinstance(payload, dict) and payload.get('model'):
            return payload
    except Exception:
        return None
    return None


def _swing_pullback_pct(symbol_df: pd.DataFrame, lookback_days: int = 20) -> float | None:
    """Recul (%, négatif) du dernier close par rapport au plus haut des
    `lookback_days` derniers jours -- le déclencheur d'évaluation à l'échelle
    swing, équivalent de _drip_trigger (-3% en 1 bougie 15m) mais pour un
    horizon de jours/semaines : une baisse ponctuelle d'une bougie ne veut
    rien dire à cette échelle, ce qui compte c'est le recul depuis un sommet
    récent."""
    if symbol_df is None or symbol_df.empty or 'close' not in symbol_df.columns:
        return None
    window = symbol_df['close'].tail(max(2, lookback_days))
    recent_high = float(window.max())
    if recent_high <= 0:
        return None
    last_close = float(symbol_df['close'].iloc[-1])
    return (last_close / recent_high) - 1.0


def scan_crypto_swing(symbols: list[str] | None = None, years: int = 2) -> list[dict[str, Any]]:
    """Équivalent SWING de scan_crypto_drip -- bougies JOURNALIÈRES, modèle
    entraîné sur CRYPTO_SWING_FEATURE_NAMES (voir crypto_training.py::
    train_crypto_swing_model), déclenché sur un recul depuis un sommet
    récent plutôt qu'une chute ponctuelle intraday. Retourne [] tant
    qu'aucun modèle swing entraîné n'existe (crypto_swing_brain_v1.pkl) --
    pas d'erreur, comportement identique à scan_crypto_drip sans modèle
    (score=None pour tous)."""
    symbols = [s.strip().upper() for s in (symbols or []) if s and str(s).strip()]
    if not symbols:
        symbols = _crypto_symbols()
    pullback_threshold = float(os.getenv('CRYPTO_SWING_PULLBACK_THRESHOLD', '0.15'))
    lookback_days = int(os.getenv('CRYPTO_SWING_PULLBACK_LOOKBACK_DAYS', '20'))

    btc_df = _normalize_columns(fetch_crypto_history_daily('BTC-USD', years=years))
    panic_verdict = _crypto_panic_verdict(_fetch_15m('BTC-USD'))  # panique systémique reste une lecture court terme
    model_payload = _load_crypto_swing_model()
    model = model_payload.get('model') if model_payload else None
    feature_cols = model_payload.get('features') if model_payload else None

    results: list[dict[str, Any]] = []
    for symbol in symbols:
        df = _normalize_columns(fetch_crypto_history_daily(symbol, years=years))
        if df.empty:
            continue
        pullback = _swing_pullback_pct(df, lookback_days=lookback_days)
        if pullback is None or pullback > -abs(pullback_threshold):
            continue  # pas assez reculé depuis le sommet récent pour que la question "la chute est-elle finie ?" ait un sens

        features = _build_crypto_swing_features(df, btc_df=btc_df)
        if features.empty:
            continue
        last = features.tail(1)

        rsi_val = float(last.get('rsi_14', pd.Series([np.nan])).iloc[-1])
        oversold = rsi_val <= float(os.getenv('CRYPTO_SWING_RSI_OVERSOLD', '30'))

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

        btc_panic, btc_corr = _btc_panic(df.tail(60), btc_df.tail(60) if not btc_df.empty else btc_df)
        blocked = btc_panic and panic_verdict == 'SYSTEMIC' and symbol != 'BTC-USD'
        results.append({
            'symbol': symbol,
            'price': float(last['close'].iloc[-1]),
            'pullback_pct': round(pullback * 100, 2),
            'rsi': rsi_val,
            'oversold': oversold,
            'btc_panic': btc_panic,
            'panic_verdict': panic_verdict,
            'btc_corr': btc_corr,
            'blocked': blocked,
            'score': score,
            'bb_pct_b_20': float(last.get('bb_pct_b_20', pd.Series([np.nan])).iloc[-1]),
            'pattern_hammer': bool(last.get('pattern_hammer', pd.Series([0])).iloc[-1]),
            'pattern_morning_star': bool(last.get('pattern_morning_star', pd.Series([0])).iloc[-1]),
        })

    return results
