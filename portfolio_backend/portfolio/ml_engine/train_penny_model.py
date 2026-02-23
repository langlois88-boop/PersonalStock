from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from .. import market_data as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df['Close']
    volume = df['Volume']
    ret = close.pct_change()

    features = pd.DataFrame({
        'close': close,
        'sma_10': close.rolling(10).mean(),
        'sma_20': close.rolling(20).mean(),
        'sma_50': close.rolling(50).mean(),
        'volatility_20': ret.rolling(20).std(),
        'volume_change_10': volume.pct_change().rolling(10).mean(),
        'rsi_14': _rsi(close, 14),
    })
    return features


def _label_targets(close: pd.Series, horizon: int = 7, threshold: float = 0.1) -> pd.Series:
    future = close.shift(-horizon)
    future_return = (future - close) / close
    return (future_return >= threshold).astype(int)


def _get_symbols() -> list[str]:
    raw = os.getenv('PENNY_SEED_SYMBOLS', '')
    symbols = [s.strip().upper() for s in raw.split(',') if s.strip()]
    if symbols:
        return symbols
    return ['FF', 'ADV', 'ARAY', 'BRCC', 'RLYB', 'KAVL', 'ZOM']


def build_dataset(symbols: Iterable[str]) -> tuple[pd.DataFrame, pd.Series]:
    frames = []
    labels = []

    for symbol in symbols:
        try:
            data = yf.Ticker(symbol).history(period='60d', interval='1d', timeout=10)
        except Exception:
            continue

        if data is None or data.empty or len(data) < 40:
            continue

        feats = _build_features(data).dropna()
        target = _label_targets(data['Close']).reindex(feats.index)
        frames.append(feats)
        labels.append(target)

    if not frames:
        raise RuntimeError('No training data available. Add PENNY_SEED_SYMBOLS.')

    X = pd.concat(frames, axis=0)
    y = pd.concat(labels, axis=0).astype(int)
    mask = y.notna()
    return X[mask], y[mask]


def train_model(output_path: Path) -> None:
    symbols = _get_symbols()
    X, y = build_dataset(symbols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42,
        class_weight='balanced',
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model': model,
        'features': list(X.columns),
        'model_type': 'classifier',
    }
    joblib.dump(payload, output_path)
    print(f"Saved model to {output_path}")


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parents[1]
    model_path = base_dir / 'ml_engine' / 'scout_brain_v1.pkl'
    train_model(model_path)
