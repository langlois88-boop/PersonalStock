from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

from ..alpaca_data import get_intraday_bars, get_intraday_bars_range
from ..patterns import enrich_bars_with_patterns


@dataclass
class IntradayModelResult:
    model: object
    features: list[str]
    scores: list[float]
    importances: list[tuple[str, float]]


def _zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0) or 1.0
    return (series - mean) / std


def _add_market_trend(df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    if market_df is None or market_df.empty:
        df['market_trend'] = 0.0
        return df
    trend = market_df[['timestamp', 'ema200', 'close']].copy()
    trend['timestamp'] = pd.to_datetime(trend['timestamp'], errors='coerce')
    trend = trend.dropna(subset=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    merged = pd.merge_asof(
        df.sort_values('timestamp'),
        trend.sort_values('timestamp'),
        on='timestamp',
        direction='backward',
    )
    merged['market_trend'] = np.where(merged['close_y'] > merged['ema200'], 1.0, -1.0)
    merged = merged.rename(columns={'close_x': 'close'})
    merged = merged.drop(columns=['close_y'])
    return merged


def _label_future_move(df: pd.DataFrame, horizon: int = 15, target_pct: float = 0.02) -> pd.Series:
    future_max = df['close'].rolling(horizon).max().shift(-horizon)
    return (future_max >= df['close'] * (1 + target_pct)).astype(int)


def _compute_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = enrich_bars_with_patterns(df)
    df = df.rename(columns={'timestamp': 'timestamp'})
    df['log_return'] = np.log(df['close'] / df['close'].shift(1)).replace([np.inf, -np.inf], 0).fillna(0)
    df['price_z'] = _zscore(df['close'])
    return df


def fetch_intraday_history(symbol: str, days: int = 365) -> pd.DataFrame:
    all_frames = []
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    # Alpaca limits can be handled by chunking days
    chunk = timedelta(days=7)
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + chunk, end)
        bars = get_intraday_bars_range(symbol, start=cursor, end=chunk_end)
        if bars is not None and not bars.empty:
            all_frames.append(bars)
        cursor = chunk_end
    if not all_frames:
        return pd.DataFrame()
    df = pd.concat(all_frames, ignore_index=True)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp']).drop_duplicates(subset=['timestamp'])
        df = df.sort_values('timestamp')
    return df


def build_dataset(symbols: Iterable[str], market_symbol: str = 'QQQ', days: int = 365) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    frames = []
    market = fetch_intraday_history(market_symbol, days=days)
    if market is not None and not market.empty:
        market = market.rename(columns={'timestamp': 'timestamp'})
        market['ema200'] = market['close'].ewm(span=200, adjust=False).mean()
    for symbol in symbols:
        raw = fetch_intraday_history(symbol, days=days)
        if raw is None or raw.empty:
            continue
        data = _compute_intraday_features(raw)
        data['symbol'] = symbol
        if market is not None and not market.empty:
            data = _add_market_trend(data, market)
        data['label'] = _label_future_move(data)
        frames.append(data)
    if not frames:
        return pd.DataFrame(), pd.Series(dtype=int), []
    dataset = pd.concat(frames, ignore_index=True)
    feature_cols = [
        'log_return',
        'price_z',
        'rsi14',
        'ema20',
        'ema50',
        'rvol',
        'pattern_signal',
        'volatility',
        'market_trend',
    ]
    dataset = dataset.dropna(subset=feature_cols + ['label'])
    return dataset, dataset['label'].astype(int), feature_cols


def train_xgboost_model(dataset: pd.DataFrame, labels: pd.Series, feature_cols: list[str]) -> IntradayModelResult:
    if dataset.empty:
        raise ValueError('No dataset for training')
    X = dataset[feature_cols]
    y = labels.values
    if XGBClassifier is None:
        model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    else:
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric='logloss',
            random_state=42,
        )
    volume_cols = [col for col in ['rvol', 'volatility'] if col in feature_cols]
    base_cols = [col for col in feature_cols if col not in volume_cols]
    scaler = ColumnTransformer(
        transformers=[
            ('volume', RobustScaler(), volume_cols),
            ('base', RobustScaler(), base_cols),
        ],
        remainder='drop',
    )
    pipeline = Pipeline([
        ('scaler', scaler),
        ('model', model),
    ])

    splits = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, test_idx in splits.split(X):
        pipeline.fit(X.iloc[train_idx], y[train_idx])
        score = pipeline.score(X.iloc[test_idx], y[test_idx])
        scores.append(float(score))

    pipeline.fit(X, y)
    importances = []
    model_step = pipeline.named_steps['model']
    weights = getattr(model_step, 'feature_importances_', None)
    if weights is not None:
        importances = sorted(zip(feature_cols, weights.tolist()), key=lambda x: x[1], reverse=True)

    return IntradayModelResult(model=pipeline, features=feature_cols, scores=scores, importances=importances)


def train_voting_ensemble(
    bluechip_symbols: Iterable[str],
    penny_symbols: Iterable[str],
    market_symbol: str = 'QQQ',
    days: int = 365,
) -> IntradayModelResult:
    blue_df, blue_y, feature_cols = build_dataset(bluechip_symbols, market_symbol=market_symbol, days=days)
    penny_df, penny_y, _ = build_dataset(penny_symbols, market_symbol=market_symbol, days=days)
    if blue_df.empty or penny_df.empty:
        raise ValueError('Not enough data for ensemble training')

    blue_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    if XGBClassifier is None:
        penny_model = RandomForestClassifier(n_estimators=250, max_depth=10, random_state=42)
    else:
        penny_model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric='logloss',
            random_state=42,
        )

    ensemble = VotingClassifier(
        estimators=[
            ('bluechip_expert', blue_model),
            ('penny_expert', penny_model),
        ],
        voting='soft',
        weights=[1, 2],
    )

    X_blue = blue_df[feature_cols]
    y_blue = blue_y.values
    X_penny = penny_df[feature_cols]
    y_penny = penny_y.values

    volume_cols = [col for col in ['rvol', 'volatility'] if col in feature_cols]
    base_cols = [col for col in feature_cols if col not in volume_cols]
    scaler = ColumnTransformer(
        transformers=[
            ('volume', RobustScaler(), volume_cols),
            ('base', RobustScaler(), base_cols),
        ],
        remainder='drop',
    )
    X_all = pd.concat([X_blue, X_penny], ignore_index=True)
    y_all = np.concatenate([y_blue, y_penny])
    X_all_scaled = scaler.fit_transform(X_all)
    ensemble.fit(X_all_scaled, y_all)

    pipeline = Pipeline([
        ('scaler', scaler),
        ('model', ensemble),
    ])

    return IntradayModelResult(model=pipeline, features=feature_cols, scores=[], importances=[])


def save_intraday_model(result: IntradayModelResult, name: str) -> str:
    base = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, f'intraday_{name}.joblib')
    joblib.dump(
        {
            'model': result.model,
            'features': result.features,
            'scores': result.scores,
            'importances': result.importances,
        },
        path,
    )
    return path
