from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import yfinance as yfin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .export_utils import export_onnx_with_gatekeeper, save_model_with_version
from .feature_registry import CRYPTO_FEATURE_NAMES

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

DEFAULT_SYMBOLS = [
    'BTC-USD',
    'ETH-USD',
    'SOL-USD',
    'BNB-USD',
    'XRP-USD',
]

CRYPTO_MODEL_PATH = Path(__file__).resolve().parent / 'crypto_brain_v1.pkl'


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    typical = (df['high'] + df['low'] + df['close']) / 3.0
    pv = typical * df['volume']
    vwap = pv.rolling(window).sum() / df['volume'].rolling(window).sum()
    return vwap


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = [str(c).lower() for c in data.columns]
    if 'close' not in data.columns:
        data = data.rename(columns={'adjclose': 'close'})
    return data


def _interval_for_days(days: int) -> tuple[str, str]:
    if days <= 7:
        return '7d', '15m'
    if days <= 30:
        return '30d', '15m'
    return '60d', '15m'


def fetch_crypto_history(symbol: str, days: int = 60) -> pd.DataFrame:
    period, interval = _interval_for_days(days)
    hist = yfin.Ticker(symbol).history(period=period, interval=interval)
    if hist is None or hist.empty:
        return pd.DataFrame()
    hist = hist.reset_index()
    hist = _normalize_columns(hist)
    if 'datetime' in hist.columns:
        hist = hist.rename(columns={'datetime': 'timestamp'})
    elif 'date' in hist.columns:
        hist = hist.rename(columns={'date': 'timestamp'})
    return hist


def _build_crypto_features(df: pd.DataFrame, btc_df: pd.DataFrame | None = None) -> pd.DataFrame:
    data = df.copy()
    if data.empty:
        return data
    data['return_1'] = data['close'].pct_change()
    data['rsi_14'] = _rsi(data['close'])
    data['ma20'] = data['close'].rolling(20).mean()
    data['std20'] = data['close'].rolling(20).std()
    data['rubber_band_index'] = (data['close'] - data['ma20']) / data['std20'].replace(0, np.nan)
    data['vwap_20'] = _vwap(data, window=20)
    data['price_to_vwap'] = (data['close'] / data['vwap_20']) - 1.0

    vol_mean = data['volume'].rolling(96).mean()
    data['volatility_spike'] = data['volume'] / vol_mean.replace(0, np.nan)

    if btc_df is not None and not btc_df.empty:
        btc = btc_df.copy()
        btc['return_1'] = btc['close'].pct_change()
        data = data.merge(btc[['timestamp', 'return_1']], on='timestamp', how='left', suffixes=('', '_btc'))
        data['btc_correlation'] = (
            data['return_1']
            .rolling(16)
            .corr(data['return_1_btc'])
        )
    else:
        data['btc_correlation'] = np.nan

    data = data.replace([np.inf, -np.inf], np.nan)
    return data


def _label_targets(close: pd.Series, horizon: int = 8, target_pct: float = 0.02) -> pd.Series:
    future = close.shift(-horizon)
    future_return = (future / close) - 1.0
    return (future_return >= target_pct).astype(int)


def build_crypto_dataset(
    symbols: Iterable[str],
    days: int = 60,
    horizon: int | None = None,
    target_pct: float | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    btc_df = fetch_crypto_history('BTC-USD', days=days)
    btc_df = _normalize_columns(btc_df)
    frames = []
    for symbol in symbols:
        raw = fetch_crypto_history(symbol, days=days)
        if raw.empty:
            continue
        raw = _normalize_columns(raw)
        features = _build_crypto_features(raw, btc_df=btc_df)
        features['symbol'] = symbol
        features['label'] = _label_targets(features['close'], horizon=horizon or 8, target_pct=target_pct or 0.02)
        frames.append(features)

    if not frames:
        return pd.DataFrame(), pd.Series(dtype=int), []

    dataset = pd.concat(frames, ignore_index=True)
    feature_cols = list(CRYPTO_FEATURE_NAMES)
    dataset = dataset.dropna(subset=feature_cols + ['label'])
    labels = dataset['label']
    return dataset, labels, feature_cols


def train_crypto_model(
    symbols: Iterable[str],
    days: int = 60,
    horizon: int | None = None,
    target_pct: float | None = None,
) -> dict[str, object]:
    horizon = int(os.getenv('CRYPTO_LABEL_HORIZON', str(horizon or 8)))
    target_pct = float(os.getenv('CRYPTO_TARGET_PCT', str(target_pct or 0.02)))

    dataset, labels, feature_cols = build_crypto_dataset(symbols, days=days, horizon=horizon, target_pct=target_pct)
    if dataset.empty:
        raise ValueError('No crypto dataset available')

    dataset = dataset.copy()
    X = dataset[feature_cols]
    y = labels.values

    if XGBClassifier is None:
        model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight='balanced')
    else:
        model = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric='logloss',
            random_state=42,
        )

    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', model),
    ])

    splits = TimeSeriesSplit(n_splits=4)
    scores = []
    for train_idx, test_idx in splits.split(X):
        pipeline.fit(X.iloc[train_idx], y[train_idx])
        scores.append(float(pipeline.score(X.iloc[test_idx], y[test_idx])))

    pipeline.fit(X, y)
    return {
        'model': pipeline,
        'features': feature_cols,
        'scores': scores,
        'cv_mean': float(sum(scores) / len(scores)) if scores else None,
        'symbols': list(symbols),
        'horizon': horizon,
        'target_pct': target_pct,
    }


def save_crypto_model(payload: dict, output_path: Path | None = None) -> str:
    path = output_path or Path(os.getenv('CRYPTO_MODEL_PATH', str(CRYPTO_MODEL_PATH)))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_model_with_version(payload, path, 'crypto', metric_name='cv_mean', metric_value=payload.get('cv_mean'))
    export_onnx_with_gatekeeper(
        payload=payload,
        model_path=path,
        model_name='crypto',
        feature_names=list(payload.get('features') or CRYPTO_FEATURE_NAMES),
        metric_name='cv_mean',
        metric_direction='higher',
    )
    return str(path)


if __name__ == '__main__':
    symbols_env = os.getenv('CRYPTO_SYMBOLS', '')
    symbols = [s.strip().upper() for s in symbols_env.split(',') if s.strip()] or DEFAULT_SYMBOLS
    payload = train_crypto_model(symbols)
    model_path = save_crypto_model(payload)
    print(f'Saved crypto model to {model_path}')
