from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from .. import market_data as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .transformers import RollingStandardScaler
from .validation import PurgedTimeSeriesSplit

from .export_utils import export_onnx_with_gatekeeper, save_model_with_version
from .feature_registry import PENNY_FEATURE_NAMES


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _build_features(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    ret = close.pct_change()

    volume_mean_20 = volume.rolling(20).mean()
    volume_std_20 = volume.rolling(20).std().replace(0, np.nan)
    volume_zscore_20 = (volume - volume_mean_20) / volume_std_20
    rvol_20 = volume / volume_mean_20.replace(0, np.nan)

    features = pd.DataFrame({
        'close': close,
        'sma_10': close.rolling(10).mean(),
        'sma_20': close.rolling(20).mean(),
        'sma_50': close.rolling(50).mean(),
        'volatility_20': ret.rolling(20).std(),
        'volume_change_10': volume.pct_change().rolling(10).mean(),
        'volume_zscore_20': volume_zscore_20,
        'rvol_20': rvol_20,
        'rsi_14': _rsi(close, 14),
    })
    return features


def _label_targets_triple_barrier(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    up_pct: float = 0.05,
    down_pct: float = 0.03,
    max_days: int = 3,
) -> pd.Series:
    if close is None or close.empty:
        return pd.Series(dtype=int)
    prices = close.values
    highs = high.values if high is not None and not high.empty else prices
    lows = low.values if low is not None and not low.empty else prices
    labels = np.full(len(prices), np.nan)
    for i in range(len(prices) - 1):
        entry = prices[i]
        if entry <= 0:
            continue
        upper = entry * (1 + up_pct)
        lower = entry * (1 - down_pct)
        end = min(len(prices), i + max_days + 1)
        hit = 0
        for j in range(i + 1, end):
            if highs[j] >= upper:
                hit = 1
                break
            if lows[j] <= lower:
                hit = 0
                break
        labels[i] = hit
    return pd.Series(labels, index=close.index)


def _get_symbols() -> list[str]:
    raw = os.getenv('PENNY_SEED_SYMBOLS', '')
    symbols = [s.strip().upper() for s in raw.split(',') if s.strip()]
    if symbols:
        return symbols
    return ['FF', 'ADV', 'ARAY', 'BRCC', 'RLYB', 'KAVL', 'ZOM']


def _extract_series(
    data: pd.DataFrame,
    symbol: str,
    candidates: tuple[str, ...],
) -> pd.Series:
    if data is None or data.empty:
        return pd.Series(dtype='float64')
    if isinstance(data.columns, pd.MultiIndex):
        for level in (len(data.columns.levels) - 1, 0):
            level_values = data.columns.get_level_values(level)
            for name in candidates:
                if name in level_values:
                    frame = data.xs(name, level=level, axis=1)
                    if isinstance(frame, pd.Series):
                        return frame
                    if symbol in frame.columns:
                        return frame[symbol]
                    if frame.columns.size:
                        return frame.iloc[:, 0]
        return pd.Series(dtype='float64')

    for name in candidates:
        if name in data.columns:
            return data[name]
        if name.lower() in data.columns:
            return data[name.lower()]
        if name.upper() in data.columns:
            return data[name.upper()]
    return pd.Series(dtype='float64')


def build_dataset(symbols: Iterable[str]) -> tuple[pd.DataFrame, pd.Series]:
    frames = []
    labels = []

    for symbol in symbols:
        try:
            data = yf.Ticker(symbol).history(period='2y', interval='1d', timeout=10)
        except Exception:
            continue

        if data is None or data.empty or len(data) < 40:
            continue

        close = _extract_series(data, symbol, ('Adj Close', 'Close')).copy()
        high = _extract_series(data, symbol, ('High',)).copy()
        low = _extract_series(data, symbol, ('Low',)).copy()
        volume = _extract_series(data, symbol, ('Volume',)).copy()
        if close.empty:
            continue
        if volume.empty:
            volume = pd.Series(0.0, index=close.index, dtype='float64')
        features = _build_features(close, volume)
        features = features.replace([np.inf, -np.inf], np.nan)
        for col in ('volume_change_10', 'volume_zscore_20', 'rvol_20'):
            if col in features.columns:
                features[col] = features[col].fillna(0)
        feats = features.dropna()
        if not feats.empty:
            feats.index = pd.MultiIndex.from_arrays(
                [feats.index, [symbol] * len(feats)],
                names=['date', 'symbol'],
            )
        target_index = feats.index
        if isinstance(target_index, pd.MultiIndex):
            target_index = target_index.get_level_values(0)
        up_pct = float(os.getenv('PENNY_TRIPLE_BARRIER_UP_PCT', '0.05'))
        down_pct = float(os.getenv('PENNY_TRIPLE_BARRIER_DOWN_PCT', '0.03'))
        max_days = int(os.getenv('PENNY_TRIPLE_BARRIER_MAX_DAYS', '3'))
        target = _label_targets_triple_barrier(close, high, low, up_pct, down_pct, max_days).reindex(target_index)
        if not feats.empty:
            target.index = feats.index
        frames.append(feats)
        labels.append(target)

    if not frames:
        raise RuntimeError('No training data available. Add PENNY_SEED_SYMBOLS.')

    X = pd.concat(frames, axis=0)
    y = pd.concat(labels, axis=0)
    mask = y.notna()
    X = X[mask].sort_index(level=0)
    y = y[mask].astype(int).reindex(X.index)
    if X.empty:
        raise RuntimeError('No training samples after feature engineering.')
    return X, y


def _walk_forward_report(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Pipeline,
    window_months: int = 3,
    min_train: int = 60,
) -> list[dict[str, float | int | str]]:
    if isinstance(X.index, pd.MultiIndex):
        date_index = X.index.get_level_values(0)
    else:
        date_index = X.index
    if not isinstance(date_index, pd.DatetimeIndex):
        return []
    start = date_index.min()
    end = date_index.max()
    if start is None or end is None:
        return []

    reports: list[dict[str, float | int | str]] = []
    test_start = start + pd.DateOffset(months=window_months)
    while test_start < end:
        test_end = test_start + pd.DateOffset(months=window_months)
        train_mask = date_index < test_start
        test_mask = (date_index >= test_start) & (date_index < test_end)
        if train_mask.sum() < min_train or test_mask.sum() == 0:
            test_start = test_end
            continue

        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        reports.append({
            'start': test_start.strftime('%Y-%m-%d'),
            'end': test_end.strftime('%Y-%m-%d'),
            'samples': int(len(y_test)),
            'accuracy': float(accuracy_score(y_test, preds)) if len(y_test) else 0.0,
            'precision': float(precision_score(y_test, preds, zero_division=0)) if len(y_test) else 0.0,
            'recall': float(recall_score(y_test, preds, zero_division=0)) if len(y_test) else 0.0,
            'f1': float(f1_score(y_test, preds, zero_division=0)) if len(y_test) else 0.0,
        })
        test_start = test_end
    return reports


def train_model(output_path: Path) -> None:
    symbols = _get_symbols()
    X, y = build_dataset(symbols)
    feature_names = list(PENNY_FEATURE_NAMES)
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_names]

    use_rolling = os.getenv('PENNY_ROLLING_SCALER', 'true').lower() in {'1', 'true', 'yes', 'y'}
    scaler = RollingStandardScaler(window=int(os.getenv('PENNY_ROLLING_WINDOW', '60'))) if use_rolling else StandardScaler()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42,
        class_weight='balanced',
    )
    if os.getenv('PENNY_OPTUNA_TRIALS', '0').isdigit() and int(os.getenv('PENNY_OPTUNA_TRIALS', '0')) > 0:
        try:
            import optuna

            trials = int(os.getenv('PENNY_OPTUNA_TRIALS', '20'))
            def objective(trial):
                n_estimators = trial.suggest_int('n_estimators', 200, 600, step=50)
                max_depth = trial.suggest_int('max_depth', 3, 10)
                min_split = trial.suggest_int('min_samples_split', 4, 12)
                min_leaf = trial.suggest_int('min_samples_leaf', 2, 10)
                clf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_split,
                    min_samples_leaf=min_leaf,
                    random_state=42,
                    class_weight='balanced',
                )
                Xv = X.values
                yv = y.values
                splitter = PurgedTimeSeriesSplit(
                    n_splits=int(os.getenv('PENNY_PURGED_SPLITS', '5')),
                    purge_window=int(os.getenv('PENNY_PURGE_WINDOW', '5')),
                    embargo_pct=float(os.getenv('PENNY_EMBARGO_PCT', '0.02')),
                )
                scores = []
                for tr, te in splitter.split(Xv):
                    clf.fit(Xv[tr], yv[tr])
                    scores.append(float(clf.score(Xv[te], yv[te])))
                return float(np.mean(scores)) if scores else 0.0

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=trials)
            params = study.best_params
            model = RandomForestClassifier(
                n_estimators=params.get('n_estimators', 300),
                max_depth=params.get('max_depth', 6),
                min_samples_split=params.get('min_samples_split', 6),
                min_samples_leaf=params.get('min_samples_leaf', 4),
                random_state=42,
                class_weight='balanced',
            )
        except Exception:
            pass

    if os.getenv('PENNY_STACKING_ENABLED', 'false').lower() in {'1', 'true', 'yes', 'y'}:
        estimators = [('rf', model)]
        try:
            from xgboost import XGBClassifier

            estimators.append(('xgb', XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric='logloss',
                random_state=42,
            )))
        except Exception:
            pass
        meta = LogisticRegression(max_iter=2000, class_weight='balanced')
        model = StackingClassifier(estimators=estimators, final_estimator=meta, passthrough=True)

    steps = [('scaler', scaler)]
    if os.getenv('PENNY_PCA_ENABLED', 'false').lower() in {'1', 'true', 'yes', 'y'}:
        n_comp = int(os.getenv('PENNY_PCA_COMPONENTS', '6'))
        steps.append(('pca', PCA(n_components=n_comp, random_state=42)))
    steps.append(('model', model))
    pipeline = Pipeline(steps)

    use_purged = os.getenv('PENNY_PURGED_CV', 'true').lower() in {'1', 'true', 'yes', 'y'}
    splits = PurgedTimeSeriesSplit(
        n_splits=int(os.getenv('PENNY_PURGED_SPLITS', '5')),
        purge_window=int(os.getenv('PENNY_PURGE_WINDOW', '5')),
        embargo_pct=float(os.getenv('PENNY_EMBARGO_PCT', '0.02')),
    ) if use_purged else TimeSeriesSplit(n_splits=5)
    cv_scores: list[float] = []
    last_test_idx = None
    for train_idx, test_idx in splits.split(X):
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        cv_scores.append(float(pipeline.score(X.iloc[test_idx], y.iloc[test_idx])))
        last_test_idx = test_idx

    pipeline.fit(X, y)
    walk_forward = _walk_forward_report(X, y, pipeline)
    print('TimeSeriesSplit scores:', cv_scores)
    if walk_forward:
        print('Walk-forward report (3-month windows):')
        for row in walk_forward:
            print(row)

    if last_test_idx is not None:
        y_pred = pipeline.predict(X.iloc[last_test_idx])
        print(classification_report(y.iloc[last_test_idx], y_pred, zero_division=0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv_mean = float(np.mean(cv_scores)) if cv_scores else None
    payload = {
        'model': pipeline,
        'features': feature_names,
        'model_type': 'classifier',
        'cv_scores': cv_scores,
        'cv_mean': cv_mean,
        'walk_forward': walk_forward,
    }
    save_model_with_version(payload, output_path, 'penny', metric_name='cv_mean', metric_value=cv_mean)
    onnx_result = export_onnx_with_gatekeeper(
        payload=payload,
        model_path=output_path,
        model_name='penny',
        feature_names=feature_names,
        metric_name='cv_mean',
        metric_direction='higher',
    )
    print(f"Saved model to {output_path}")
    if onnx_result.get('exported'):
        print(f"Exported ONNX to {onnx_result.get('onnx_path')}")


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parents[1]
    model_path = base_dir / 'ml_engine' / 'scout_brain_v1.pkl'
    train_model(model_path)
