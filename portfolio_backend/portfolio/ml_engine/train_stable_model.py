from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from portfolio import market_data as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.decomposition import PCA
import joblib
import requests

from portfolio.ml_engine.export_utils import export_onnx_with_gatekeeper, save_model_with_version, write_meta_sidecar
from portfolio.ml_engine.feature_registry import STABLE_FEATURE_NAMES
from portfolio.ml_engine.transformers import RollingStandardScaler
from portfolio.ml_engine.collectors.news_rss import fetch_news_sentiment
from portfolio.ml_engine.push_model import _build_meta_from_payload, push_to_portfolio_app




MODEL_PATH = Path(__file__).resolve().parent / 'stable_brain_v1.pkl'
FEATURE_NAMES = list(STABLE_FEATURE_NAMES)
def _ensure_django() -> None:
    if not os.getenv('DJANGO_SETTINGS_MODULE'):
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'portfolio_backend.settings')
    try:
        import django

        django.setup()
    except Exception:
        return


SECTOR_ETF_MAP = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Health Care': 'XLV',
    'Financial Services': 'XLF',
    'Financials': 'XLF',
    'Energy': 'XLE',
    'Consumer Defensive': 'XLP',
    'Consumer Cyclical': 'XLY',
    'Industrials': 'XLI',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Communication Services': 'XLC',
    'Basic Materials': 'XLB',
}


def _is_valid_symbol(symbol: str) -> bool:
    if not symbol:
        return False
    symbol = symbol.strip().upper()
    if symbol in {'.', '-', '_'}:
        return False
    return bool(re.fullmatch(r"[A-Z0-9\.\-]{1,10}", symbol))


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _build_features(
    close: pd.Series,
    volume: pd.Series,
    spy_close: pd.Series,
    dividend_yield: float,
    sector_close: pd.Series | None = None,
    sentiment_score: float = 0.0,
):
    ret = close.pct_change()
    spy_ret = spy_close.pct_change()
    aligned = pd.concat([ret, spy_ret], axis=1, join='inner').dropna()
    aligned.columns = ['stock', 'spy']
    if len(aligned) < 60 or len(close) < 60:
        return None

    volume_mean_20 = volume.rolling(20).mean()
    volume_std_20 = volume.rolling(20).std().replace(0, np.nan)
    volume_zscore_20 = ((volume - volume_mean_20) / volume_std_20).clip(-3, 3).fillna(0)

    spy_corr = ret.rolling(60).corr(spy_ret).fillna(0)
    sector_beta = pd.Series(0.0, index=close.index)
    if sector_close is not None and not sector_close.empty:
        sector_ret = sector_close.pct_change()
        aligned_sector = pd.concat([ret, sector_ret], axis=1, join='inner').dropna()
        aligned_sector.columns = ['stock', 'sector']
        if len(aligned_sector) >= 60:
            sector_beta = (
                aligned_sector['stock'].rolling(60).cov(aligned_sector['sector'])
                / aligned_sector['sector'].rolling(60).var().replace(0, np.nan)
            ).reindex(close.index).ffill()

    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean().replace(0, np.nan)
    sma_ratio_10_50 = (sma_10 / sma_50).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    sma_ratio_20_50 = (sma_20 / sma_50).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    return [
        float(_rsi(close, 14).iloc[-1]),
        float(sma_ratio_10_50.iloc[-1]),
        float(sma_ratio_20_50.iloc[-1]),
        float(ret.rolling(20).std().iloc[-1]),
        float(volume_zscore_20.iloc[-1]),
        float(close.pct_change(20).iloc[-1]),
        float(spy_corr.iloc[-1]),
        float(dividend_yield),
        float(sector_beta.iloc[-1]),
        float(sentiment_score),
    ]


def _price_history_series(symbol: str, days: int = 365 * 2) -> pd.Series | None:
    from portfolio.models import Stock, PriceHistory

    stock = Stock.objects.filter(symbol__iexact=symbol).first()
    if not stock:
        return None
    cutoff = datetime.now().date() - pd.Timedelta(days=days)
    rows = PriceHistory.objects.filter(stock=stock, date__gte=cutoff).order_by('date')
    if not rows.exists():
        return None
    series = pd.Series({row.date: float(row.close_price or 0) for row in rows})
    series = series.replace(0, np.nan).dropna()
    return series if not series.empty else None


def _auto_push_enabled() -> bool:
    return (
        os.getenv('AUTO_PUSH_MODEL', '')
        or os.getenv('AUTO_PUSH_MODELS', '')
    ).lower() in {'1', 'true', 'yes', 'y'}


def _deepseek_enabled() -> bool:
    return os.getenv('DEEPSEEK_WEIGHTING_ENABLED', 'false').lower() in {'1', 'true', 'yes', 'y'}


def _deepseek_confidence(symbol: str, date_value: str, features: dict[str, float], raw_label: int) -> float:
    base_url = (os.getenv('DEEPSEEK_API_URL') or '').strip().rstrip('/')
    if not base_url:
        return 1.0
    prompt = (
        "Stock: {symbol}, Date: {date}, Raw label: {label}\n"
        "Features: RSI={rsi:.1f}, SMA_ratio={sma_ratio:.3f}, "
        "Vol_zscore={vol_z:.2f}, Return_20d={ret_20d:.3f}\n"
        "Sentiment: {sentiment:.2f}\n\n"
        "Given these indicators, rate your confidence 0.0-1.0 that the label "
        "({label}) is correct. Respond with ONLY a float number, nothing else."
    ).format(
        symbol=symbol,
        date=date_value,
        label=raw_label,
        rsi=features.get('rsi_14', 0.0),
        sma_ratio=features.get('sma_ratio_10_50', 1.0),
        vol_z=features.get('volume_zscore_20', 0.0),
        ret_20d=features.get('return_20d', 0.0),
        sentiment=features.get('sentiment_score', 0.0),
    )
    try:
        resp = requests.post(
            base_url,
            json={'prompt': prompt},
            timeout=int(os.getenv('DEEPSEEK_TIMEOUT', '20')),
        )
        resp.raise_for_status()
        text = (resp.json().get('response') if resp.headers.get('content-type', '').startswith('application/json') else resp.text)
        if isinstance(text, dict):
            text = json.dumps(text)
        value = float(str(text).strip().split()[0])
        return max(0.0, min(1.0, value))
    except Exception:
        return 1.0


def _label_targets_triple_barrier(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    up_pct: float = 0.15,
    down_pct: float = 0.07,
    max_days: int = 20,
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


def train_stable_model(auto_push: bool | None = None):
    _ensure_django()
    print(f"[{datetime.utcnow().isoformat()}Z] Training started")
    from portfolio.models import Stock
    symbols = [s.symbol for s in Stock.objects.all().order_by('symbol') if _is_valid_symbol(s.symbol)]
    if not symbols:
        raise RuntimeError('No stocks found to train on.')

    rows = []
    targets = []

    spy = yf.Ticker('SPY').history(period='2y', interval='1d', timeout=10)
    if spy is None or spy.empty or ('Close' not in spy and 'Adj Close' not in spy):
        spy = yf.Ticker('QQQ').history(period='2y', interval='1d', timeout=10)
    if spy is None or spy.empty or ('Close' not in spy and 'Adj Close' not in spy):
        spy_close = _price_history_series('SPY') or _price_history_series('QQQ')
        if spy_close is None:
            return {
                'status': 'no_index',
                'reason': 'Index history missing (SPY/QQQ).',
            }
    else:
        spy_close = spy['Adj Close'] if 'Adj Close' in spy else spy['Close']

    for symbol in symbols:
        try:
            data = yf.Ticker(symbol).history(period='2y', interval='1d', timeout=10)
        except Exception:
            data = None
        if data is None or data.empty or ('Close' not in data and 'Adj Close' not in data):
            close = _price_history_series(symbol)
            if close is None:
                continue
            volume = pd.Series([0.0] * len(close), index=close.index)
            high = close
            low = close
        else:
            close = data['Adj Close'] if 'Adj Close' in data else data['Close']
            volume = data['Volume'] if 'Volume' in data else pd.Series([0.0] * len(close), index=close.index)
            high = data['High'] if 'High' in data else close
            low = data['Low'] if 'Low' in data else close
        stock_row = Stock.objects.filter(symbol=symbol).values('dividend_yield', 'sector').first() or {}
        dividend_yield = float(stock_row.get('dividend_yield') or 0)
        sector_name = stock_row.get('sector') or ''
        sector_symbol = SECTOR_ETF_MAP.get(str(sector_name).strip(), '')
        sector_close = None
        if sector_symbol:
            try:
                sector_hist = yf.Ticker(sector_symbol).history(period='2y', interval='1d', timeout=10)
                if sector_hist is not None and not sector_hist.empty:
                    sector_close = sector_hist['Adj Close'] if 'Adj Close' in sector_hist else sector_hist['Close']
            except Exception:
                sector_close = None

        news_payload = fetch_news_sentiment(symbol)
        sentiment_score = float(news_payload.get('news_sentiment') or 0.0)
        label_series = _label_targets_triple_barrier(
            close=close,
            high=high,
            low=low,
            up_pct=float(os.getenv('STABLE_TRIPLE_BARRIER_UP_PCT', '0.15')),
            down_pct=float(os.getenv('STABLE_TRIPLE_BARRIER_DOWN_PCT', '0.07')),
            max_days=int(os.getenv('STABLE_TRIPLE_BARRIER_MAX_DAYS', '20')),
        )
        for i in range(60, len(close) - 20):
            if len(close.iloc[:i + 1]) < 60:
                continue
            slice_close = close.iloc[:i + 1]
            slice_volume = volume.iloc[:i + 1]
            slice_spy = spy_close.reindex(slice_close.index).ffill()
            slice_sector = sector_close.reindex(slice_close.index).ffill() if sector_close is not None else None
            feature_values = _build_features(
                slice_close,
                slice_volume,
                slice_spy,
                dividend_yield,
                sector_close=slice_sector,
                sentiment_score=sentiment_score,
            )
            if any(pd.isna(val) for val in feature_values):
                continue
            label_value = label_series.iloc[i] if i < len(label_series) else np.nan
            row = {'date': close.index[i], 'label': label_value, 'symbol': symbol}
            row.update({FEATURE_NAMES[j]: feature_values[j] for j in range(len(FEATURE_NAMES))})
            rows.append(row)

    if not rows:
        raise RuntimeError('No training samples created.')

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')
    df = df.dropna(subset=['label'])
    X = df[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = df['label'].astype(int).values
    if len(set(y)) < 2:
        raise ValueError('Only one class')

    scaler = StandardScaler()
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        random_state=42,
        class_weight='balanced',
    )
    if os.getenv('STABLE_OPTUNA_TRIALS', '0').isdigit() and int(os.getenv('STABLE_OPTUNA_TRIALS', '0')) > 0:
        try:
            import optuna

            trials = int(os.getenv('STABLE_OPTUNA_TRIALS', '20'))
            def objective(trial):
                n_estimators = trial.suggest_int('n_estimators', 200, 800, step=50)
                max_depth = trial.suggest_int('max_depth', 2, 5)
                lr = trial.suggest_float('learning_rate', 0.02, 0.2)
                reg = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=lr,
                    random_state=42,
                )
                splitter = PurgedTimeSeriesSplit(
                    n_splits=int(os.getenv('STABLE_PURGED_SPLITS', '5')),
                    purge_window=int(os.getenv('STABLE_PURGE_WINDOW', '5')),
                    embargo_pct=float(os.getenv('STABLE_EMBARGO_PCT', '0.02')),
                )
                scores = []
                for tr, te in splitter.split(X):
                    reg.fit(X[tr], y[tr])
                    preds = reg.predict(X[te])
                    scores.append(mean_absolute_error(y[te], preds))
                return float(np.mean(scores)) if scores else 999.0

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=trials)
            params = study.best_params
            model = GradientBoostingRegressor(
                n_estimators=params.get('n_estimators', 500),
                max_depth=params.get('max_depth', 3),
                learning_rate=params.get('learning_rate', 0.05),
                random_state=42,
            )
        except Exception:
            pass

    steps = [('scaler', scaler)]
    if os.getenv('STABLE_PCA_ENABLED', 'false').lower() in {'1', 'true', 'yes', 'y'}:
        n_comp = int(os.getenv('STABLE_PCA_COMPONENTS', '6'))
        steps.append(('pca', PCA(n_components=n_comp, random_state=42)))
    steps.append(('model', model))
    pipeline = Pipeline(steps)

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    weights = None
    if _deepseek_enabled():
        weights = []
        for i, row in df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
            feature_dict = {name: float(row[name]) for name in FEATURE_NAMES}
            weights.append(_deepseek_confidence(str(row.get('symbol', '')), date_str, feature_dict, int(row['label'])))
        weights = np.array(weights, dtype=float)
    for train_idx, test_idx in tscv.split(X):
        if weights is not None:
            pipeline.fit(X[train_idx], y[train_idx], model__sample_weight=weights[train_idx])
        else:
            pipeline.fit(X[train_idx], y[train_idx])
        scores.append(float(pipeline.score(X[test_idx], y[test_idx])))

    def _walk_forward_f1() -> list[dict[str, float | int | str]]:
        reports = []
        if df.empty:
            return reports
        start = df['date'].min()
        end = df['date'].max()
        if start is None or end is None:
            return reports
        test_start = start + pd.DateOffset(months=3)
        while test_start < end:
            test_end = test_start + pd.DateOffset(months=3)
            train_mask = df['date'] < test_start
            test_mask = (df['date'] >= test_start) & (df['date'] < test_end)
            if train_mask.sum() < 60 or test_mask.sum() == 0:
                test_start = test_end
                continue
            X_train = df.loc[train_mask, FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0).values
            y_train = df.loc[train_mask, 'label'].astype(int).values
            X_test = df.loc[test_mask, FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0).values
            y_test = df.loc[test_mask, 'label'].astype(int).values
            window_pipeline = clone(pipeline)
            window_pipeline.fit(X_train, y_train)
            preds = window_pipeline.predict(X_test)
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
    if weights is not None:
        pipeline.fit(X, y, model__sample_weight=weights)
    else:
        pipeline.fit(X, y)
    cv_mean = float(np.mean(scores)) if scores else None
    wf_f1 = float(np.mean([row.get('f1', 0.0) for row in walk_forward])) if walk_forward else 0.0
    if cv_mean is None or cv_mean < 0.55:
        raise ValueError(f"CV mean {cv_mean:.3f} below threshold 0.55 — not deploying")
    if wf_f1 < 0.50:
        raise ValueError(f"Walk-forward F1 {wf_f1:.3f} below threshold 0.50")
    payload = {
        'model': pipeline,
        'features': FEATURE_NAMES,
        'model_type': 'classifier',
        'universe': 'STABLE',
        'model_version': f"v{datetime.utcnow().date().isoformat()}",
        'trained_at': datetime.utcnow().isoformat() + 'Z',
        'cv_scores': scores,
        'cv_mean': cv_mean,
        'walk_forward': walk_forward,
    }
    version_info = save_model_with_version(payload, MODEL_PATH, 'stable', metric_name='cv_mean', metric_value=cv_mean)
    onnx_result = export_onnx_with_gatekeeper(
        payload=payload,
        model_path=MODEL_PATH,
        model_name='stable',
        feature_names=FEATURE_NAMES,
        metric_name='cv_mean',
        metric_direction='higher',
    )
    if onnx_result.get('exported'):
        onnx_path = Path(onnx_result.get('onnx_path'))
        label_balance = float(np.mean(y)) if len(y) else 0.0
        write_meta_sidecar(onnx_path, scores, FEATURE_NAMES, 'STABLE', len(y), label_balance)
        if auto_push is None:
            auto_push = _auto_push_enabled()
        if auto_push:
            try:
                meta = _build_meta_from_payload(payload)
                meta.update({'model_version': version_info.get('model_version')})
                push_to_portfolio_app('stable', str(onnx_path), meta=meta)
            except Exception:
                pass
    print(f"[{datetime.utcnow().isoformat()}Z] Training completed")

    model_obj = pipeline.named_steps.get('model') if hasattr(pipeline, 'named_steps') else None
    if model_obj is not None and hasattr(model_obj, 'feature_importances_'):
        importances = list(zip(FEATURE_NAMES, model_obj.feature_importances_))
        print('Feature importances:')
        for name, score in sorted(importances, key=lambda x: x[1], reverse=True):
            print(f"{name}: {score:.4f}")

    return {
        'samples': len(X),
        'cv_mean': cv_mean,
        'walk_forward': walk_forward,
        'model_path': str(MODEL_PATH),
        'onnx': onnx_result,
    }


if __name__ == '__main__':
    _ensure_django()
    result = train_stable_model()
    print(result)
