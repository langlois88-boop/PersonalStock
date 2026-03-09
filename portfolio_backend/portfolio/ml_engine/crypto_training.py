from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import yfinance as yfin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from portfolio.ml_engine.export_utils import export_onnx_with_gatekeeper, save_model_with_version, write_meta_sidecar
from portfolio.ml_engine.feature_registry import CRYPTO_FEATURE_NAMES
from portfolio.ml_engine.push_model import _build_meta_from_payload, push_to_portfolio_app
def _ensure_django() -> None:
    if not os.getenv('DJANGO_SETTINGS_MODULE'):
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'portfolio_backend.settings')
    try:
        import django

        django.setup()
    except Exception:
        return


DEFAULT_SYMBOLS = [
    'BTC-USD',
    'ETH-USD',
    'SOL-USD',
    'BNB-USD',
    'XRP-USD',
]

CRYPTO_MODEL_PATH = Path(__file__).resolve().parent / 'crypto_brain_v1.pkl'


def _auto_push_enabled() -> bool:
    return (
        os.getenv('AUTO_PUSH_MODEL', '')
        or os.getenv('AUTO_PUSH_MODELS', '')
    ).lower() in {'1', 'true', 'yes', 'y'}


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    typical = (df['high'] + df['low'] + df['close']) / 3.0
    pv = typical * df['volume']
    vwap = pv.cumsum() / df['volume'].replace(0, np.nan).cumsum()
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
    if days <= 60:
        return '60d', '15m'
    capped = min(days, 730)
    return f'{capped}d', '1h'


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
    data['std20'] = data['close'].rolling(20).std().replace(0, np.nan)
    data['rubber_band_index'] = (data['close'] - data['ma20']) / (2 * data['std20'])
    data['vwap_20'] = _vwap(data, window=20)
    data['price_to_vwap'] = (data['close'] / data['vwap_20']) - 1.0

    vol_mean = data['volume'].rolling(20).mean()
    data['volatility_spike'] = data['volume'] / vol_mean.replace(0, np.nan)

    if btc_df is not None and not btc_df.empty:
        btc = btc_df.copy()
        btc['return_1'] = btc['close'].pct_change()
        data = data.merge(btc[['timestamp', 'return_1']], on='timestamp', how='left', suffixes=('', '_btc'))
        data['btc_correlation'] = data['return_1'].rolling(30).corr(data['return_1_btc']).fillna(0)
    else:
        data['btc_correlation'] = 0.0

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
    _ensure_django()
    print(f"[{datetime.utcnow().isoformat()}Z] Training started")
    days = int(os.getenv('CRYPTO_HISTORY_DAYS', str(days)))
    horizon = int(os.getenv('CRYPTO_LABEL_HORIZON', str(horizon or 8)))
    target_pct = float(os.getenv('CRYPTO_TARGET_PCT', str(target_pct or 0.02)))

    dataset, labels, feature_cols = build_crypto_dataset(symbols, days=days, horizon=horizon, target_pct=target_pct)
    if dataset.empty:
        raise ValueError('No crypto dataset available')

    dataset = dataset.copy()
    X = dataset[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = labels.values
    if len(set(y)) < 2:
        raise ValueError('Only one class')

    model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight='balanced')

    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', model),
    ])

    splits = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, test_idx in splits.split(X):
        pipeline.fit(X.iloc[train_idx], y[train_idx])
        scores.append(float(pipeline.score(X.iloc[test_idx], y[test_idx])))

    def _walk_forward_f1(window_months: int = 3, min_train: int = 60) -> list[dict[str, float | int | str]]:
        reports = []
        date_index = dataset['timestamp'] if 'timestamp' in dataset.columns else dataset.index
        if not isinstance(date_index, pd.DatetimeIndex):
            try:
                date_index = pd.to_datetime(date_index, errors='coerce')
            except Exception:
                return reports
        if date_index.isna().all():
            return reports
        start = date_index.min()
        end = date_index.max()
        if start is None or end is None:
            return reports
        test_start = start + pd.DateOffset(months=window_months)
        while test_start < end:
            test_end = test_start + pd.DateOffset(months=window_months)
            train_mask = date_index < test_start
            test_mask = (date_index >= test_start) & (date_index < test_end)
            if train_mask.sum() < min_train or test_mask.sum() == 0:
                test_start = test_end
                continue
            X_train = X.loc[train_mask]
            y_train = y[train_mask]
            X_test = X.loc[test_mask]
            y_test = y[test_mask]
            model_clone = Pipeline([
                ('scaler', RobustScaler()),
                ('model', RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight='balanced')),
            ])
            model_clone.fit(X_train, y_train)
            preds = model_clone.predict(X_test)
            f1 = float(f1_score(y_test, preds, zero_division=0)) if len(y_test) else 0.0
            reports.append({
                'start': test_start.strftime('%Y-%m-%d'),
                'end': test_end.strftime('%Y-%m-%d'),
                'samples': int(len(y_test)),
                'f1': f1,
            })
            test_start = test_end
        return reports

    walk_forward = _walk_forward_f1()
    pipeline.fit(X, y)
    print(f"[{datetime.utcnow().isoformat()}Z] Training completed")
    model_obj = pipeline.named_steps.get('model') if hasattr(pipeline, 'named_steps') else None
    if model_obj is not None and hasattr(model_obj, 'feature_importances_'):
        importances = list(zip(feature_cols, model_obj.feature_importances_))
        print('Feature importances:')
        for name, score in sorted(importances, key=lambda x: x[1], reverse=True):
            print(f"{name}: {score:.4f}")
    return {
        'model': pipeline,
        'features': feature_cols,
        'cv_scores': scores,
        'cv_mean': float(sum(scores) / len(scores)) if scores else None,
        'universe': 'CRYPTO',
        'model_version': f"v{datetime.utcnow().date().isoformat()}",
        'trained_at': datetime.utcnow().isoformat() + 'Z',
        'n_samples': int(len(y)),
        'label_balance': float(np.mean(y)) if len(y) else 0.0,
        'walk_forward': walk_forward,
        'labels': y.tolist(),
        'symbols': list(symbols),
        'horizon': horizon,
        'target_pct': target_pct,
    }


def save_crypto_model(payload: dict, output_path: Path | None = None, auto_push: bool | None = None) -> str:
    path = output_path or Path(os.getenv('CRYPTO_MODEL_PATH', str(CRYPTO_MODEL_PATH)))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv_scores = payload.get('cv_scores') or []
    cv_mean = payload.get('cv_mean')
    wf_f1 = float(np.mean([row.get('f1', 0.0) for row in (payload.get('walk_forward') or [])]))
    if cv_mean is None or cv_mean < 0.55:
        raise ValueError(f"CV mean {cv_mean:.3f} below threshold 0.55 — not deploying")
    if wf_f1 < 0.50:
        raise ValueError(f"Walk-forward F1 {wf_f1:.3f} below threshold 0.50")
    version_info = save_model_with_version(payload, path, 'crypto', metric_name='cv_mean', metric_value=payload.get('cv_mean'))
    onnx_result = export_onnx_with_gatekeeper(
        payload=payload,
        model_path=path,
        model_name='crypto',
        feature_names=list(payload.get('features') or CRYPTO_FEATURE_NAMES),
        metric_name='cv_mean',
        metric_direction='higher',
    )
    if onnx_result.get('exported'):
        onnx_path = Path(onnx_result.get('onnx_path'))
        label_balance = float(payload.get('label_balance') or 0.0)
        write_meta_sidecar(
            onnx_path,
            cv_scores,
            list(payload.get('features') or CRYPTO_FEATURE_NAMES),
            'CRYPTO',
            int(payload.get('n_samples') or 0),
            label_balance,
        )
        if auto_push is None:
            auto_push = _auto_push_enabled()
        if auto_push:
            try:
                meta = _build_meta_from_payload(payload)
                meta.update({'model_version': version_info.get('model_version')})
                push_to_portfolio_app('crypto', str(onnx_path), meta=meta)
            except Exception:
                pass
    return str(path)


if __name__ == '__main__':
    symbols_env = os.getenv('CRYPTO_SYMBOLS', '')
    symbols = [s.strip().upper() for s in symbols_env.split(',') if s.strip()] or DEFAULT_SYMBOLS
    payload = train_crypto_model(symbols)
    model_path = save_crypto_model(payload)
    print(f'Saved crypto model to {model_path}')
