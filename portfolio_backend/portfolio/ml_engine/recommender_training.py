from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.model_selection import TimeSeriesSplit

from .model import FEATURE_COLUMNS, build_model
from ..models import Stock, PriceHistory

RECOMMENDER_MODEL_PATH = Path(__file__).resolve().parent / 'recommender_brain_v1.pkl'


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def _build_symbol_frame(symbol: str, lookback_days: int, horizon: int, target_pct: float) -> pd.DataFrame:
    stock = Stock.objects.filter(symbol__iexact=symbol).first()
    if not stock:
        return pd.DataFrame()
    cutoff = datetime.now().date() - pd.Timedelta(days=lookback_days)
    rows = PriceHistory.objects.filter(stock=stock, date__gte=cutoff).order_by('date')
    if not rows.exists():
        return pd.DataFrame()
    close = pd.Series({row.date: float(row.close_price or 0) for row in rows}).replace(0, np.nan).dropna()
    if close.empty or len(close) < max(60, horizon + 20):
        return pd.DataFrame()

    df = pd.DataFrame({'close': close}).sort_index()
    df['rsi_14'] = _rsi(df['close'], 14)
    df['vol_zscore'] = _zscore(df['close'], 20)
    df['return_20d'] = np.log(df['close'] / df['close'].shift(20))
    df['future_return'] = (df['close'].shift(-horizon) - df['close']) / df['close']
    df['label'] = (df['future_return'] >= target_pct).astype(int)

    df = df.dropna(subset=['rsi_14', 'vol_zscore', 'return_20d', 'label'])
    if df.empty:
        return pd.DataFrame()

    df['roe'] = 0.0
    df['debt_to_equity'] = 0.0
    df['news_sentiment'] = 0.0
    df['news_count'] = 0.0
    df['fred_rate'] = 0.0
    df['symbol'] = symbol

    return df


def build_recommender_dataset(symbols: Iterable[str], lookback_days: int, horizon: int, target_pct: float) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        frame = _build_symbol_frame(symbol, lookback_days, horizon, target_pct)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def _select_threshold(y_true: np.ndarray, probs: np.ndarray, min_precision: float, min_recall: float) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    if thresholds.size == 0:
        return 0.7
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
    mask = (precision >= min_precision) & (recall >= min_recall)
    if mask.any():
        idx = int(np.nanargmax(f1_scores[mask]))
        thresh = thresholds[np.flatnonzero(mask)[idx]]
    else:
        idx = int(np.nanargmax(f1_scores))
        thresh = thresholds[min(idx, len(thresholds) - 1)]
    return float(thresh)


def train_recommender_model() -> dict[str, object]:
    lookback_days = int(os.getenv('RECOMMENDER_LOOKBACK_DAYS', '720'))
    horizon = int(os.getenv('RECOMMENDER_LABEL_HORIZON', '20'))
    target_pct = float(os.getenv('RECOMMENDER_TARGET_PCT', '0.05'))
    max_symbols = int(os.getenv('RECOMMENDER_MAX_SYMBOLS', '200'))
    method = os.getenv('RECOMMENDER_CALIBRATION_METHOD', 'isotonic')
    min_precision = float(os.getenv('RECOMMENDER_MIN_PRECISION', '0.6'))
    min_recall = float(os.getenv('RECOMMENDER_MIN_RECALL', '0.1'))

    symbols = [s.symbol for s in Stock.objects.all().order_by('symbol')][:max_symbols]
    dataset = build_recommender_dataset(symbols, lookback_days, horizon, target_pct)
    if dataset.empty:
        raise RuntimeError('No recommender training samples available.')

    X = dataset[FEATURE_COLUMNS].values
    y = dataset['label'].values

    splitter = TimeSeriesSplit(n_splits=5)
    oof_probs = np.full(len(y), np.nan)
    for train_idx, test_idx in splitter.split(X):
        base = build_model()
        calibrated = CalibratedClassifierCV(estimator=base, method=method, cv=3)
        calibrated.fit(X[train_idx], y[train_idx])
        oof_probs[test_idx] = calibrated.predict_proba(X[test_idx])[:, 1]

    valid_mask = ~np.isnan(oof_probs)
    if not valid_mask.any():
        raise RuntimeError('Calibration failed to produce out-of-fold probabilities.')
    buy_threshold = _select_threshold(y[valid_mask], oof_probs[valid_mask], min_precision, min_recall)
    oof_preds = (oof_probs[valid_mask] >= buy_threshold).astype(int)
    f1 = float(f1_score(y[valid_mask], oof_preds)) if valid_mask.any() else 0.0

    final_base = build_model()
    final_calibrated = CalibratedClassifierCV(estimator=final_base, method=method, cv=3)
    final_calibrated.fit(X, y)

    return {
        'model': final_calibrated,
        'features': FEATURE_COLUMNS,
        'buy_threshold': buy_threshold,
        'f1_oof': f1,
        'calibration_method': method,
        'label_horizon': horizon,
        'target_pct': target_pct,
        'samples': int(len(dataset)),
    }


def save_recommender_model(payload: dict, output_path: Path | None = None) -> str:
    path = output_path or Path(os.getenv('RECOMMENDER_MODEL_PATH', str(RECOMMENDER_MODEL_PATH)))
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    return str(path)


if __name__ == '__main__':
    payload = train_recommender_model()
    model_path = save_recommender_model(payload)
    print(f'Saved recommender model to {model_path}')
