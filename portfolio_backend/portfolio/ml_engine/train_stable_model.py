from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from .. import market_data as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.decomposition import PCA
import joblib

from .export_utils import export_onnx_with_gatekeeper, save_model_with_version
from .feature_registry import STABLE_FEATURE_NAMES
from .transformers import RollingStandardScaler
from .validation import PurgedTimeSeriesSplit

from portfolio.models import Stock, MacroIndicator, PriceHistory


MODEL_PATH = Path(__file__).resolve().parent / 'stable_brain_v1.pkl'
FEATURE_NAMES = list(STABLE_FEATURE_NAMES)

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


def _build_features(close: pd.Series, volume: pd.Series, spy_close: pd.Series, dividend_yield: float, macro: MacroIndicator | None, sector_close: pd.Series | None = None):
    ret = close.pct_change().dropna()
    spy_ret = spy_close.pct_change().dropna()
    aligned = pd.concat([ret, spy_ret], axis=1, join='inner').dropna()
    aligned.columns = ['stock', 'spy']
    if len(aligned) < 60 or len(close) < 220:
        return None

    beta = float(aligned['stock'].cov(aligned['spy']) / (aligned['spy'].var() or 1))
    log_ret_20 = float(np.log(close.iloc[-1] / close.iloc[-21]))
    vol_60 = float(ret.tail(60).std())
    rel_volume_200 = float(volume.tail(5).mean() / (volume.tail(200).mean() or 1))

    sector_strength = 0.0
    if sector_close is not None and len(sector_close) >= 21:
        try:
            sector_strength = float(np.log(sector_close.iloc[-1] / sector_close.iloc[-21]))
        except Exception:
            sector_strength = 0.0

    macro_features = [
        float(macro.sp500_close) if macro else 0.0,
        float(macro.vix_index) if macro else 0.0,
        float(macro.interest_rate_10y) if macro else 0.0,
        float(macro.inflation_rate) if macro else 0.0,
        float(macro.oil_price) if (macro and macro.oil_price is not None) else 0.0,
    ]

    return [
        log_ret_20,
        vol_60,
        beta,
        rel_volume_200,
        dividend_yield,
        sector_strength,
        *macro_features,
    ]


def _price_history_series(symbol: str, days: int = 365 * 2) -> pd.Series | None:
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


def train_stable_model():
    symbols = [s.symbol for s in Stock.objects.all().order_by('symbol') if _is_valid_symbol(s.symbol)]
    if not symbols:
        raise RuntimeError('No stocks found to train on.')

    macro = MacroIndicator.objects.order_by('-date').first()

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
        else:
            close = data['Adj Close'] if 'Adj Close' in data else data['Close']
        volume = data['Volume'] if 'Volume' in data else pd.Series([0] * len(close), index=close.index)
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

        ret = close.pct_change()
        spy_ret = spy_close.pct_change()
        aligned = pd.concat([ret, spy_ret], axis=1, join='inner').dropna()
        aligned.columns = ['stock', 'spy']
        beta_series = (
            aligned['stock'].expanding(60).cov(aligned['spy'])
            / aligned['spy'].expanding(60).var()
        ).reindex(close.index).ffill()

        log_ret_20 = np.log(close / close.shift(20))
        vol_60 = ret.rolling(60).std()
        rel_volume_200 = volume.rolling(5).mean() / volume.rolling(200).mean()

        sector_strength = pd.Series(0.0, index=close.index)
        if sector_close is not None and not sector_close.empty:
            sector_strength = np.log(sector_close / sector_close.shift(20)).reindex(close.index).ffill().fillna(0.0)

        macro_features = [
            float(macro.sp500_close) if macro else 0.0,
            float(macro.vix_index) if macro else 0.0,
            float(macro.interest_rate_10y) if macro else 0.0,
            float(macro.inflation_rate) if macro else 0.0,
            float(macro.oil_price) if (macro and macro.oil_price is not None) else 0.0,
        ]

        for i in range(220, len(close) - 20):
            if len(close.iloc[:i + 1]) < 220:
                continue
            feature_values = [
                log_ret_20.iloc[i],
                vol_60.iloc[i],
                beta_series.iloc[i],
                rel_volume_200.iloc[i],
                dividend_yield,
                sector_strength.iloc[i],
                *macro_features,
            ]
            if any(pd.isna(val) for val in feature_values):
                continue
            future_return = float((close.iloc[i + 20] - close.iloc[i]) / close.iloc[i])
            row = {'date': close.index[i], 'target': future_return}
            row.update({FEATURE_NAMES[j]: feature_values[j] for j in range(len(FEATURE_NAMES))})
            rows.append(row)

    if not rows:
        raise RuntimeError('No training samples created.')

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')
    X = df[FEATURE_NAMES].values
    y = df['target'].values

    use_rolling = os.getenv('STABLE_ROLLING_SCALER', 'true').lower() in {'1', 'true', 'yes', 'y'}
    scaler = RollingStandardScaler(window=int(os.getenv('STABLE_ROLLING_WINDOW', '90'))) if use_rolling else StandardScaler()

    model = GradientBoostingRegressor(random_state=42)
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

    use_purged = os.getenv('STABLE_PURGED_CV', 'true').lower() in {'1', 'true', 'yes', 'y'}
    tscv = PurgedTimeSeriesSplit(
        n_splits=int(os.getenv('STABLE_PURGED_SPLITS', '5')),
        purge_window=int(os.getenv('STABLE_PURGE_WINDOW', '5')),
        embargo_pct=float(os.getenv('STABLE_EMBARGO_PCT', '0.02')),
    ) if use_purged else TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        pipeline.fit(X[train_idx], y[train_idx])
        preds = pipeline.predict(X[test_idx])
        scores.append(mean_absolute_error(y[test_idx], preds))

    def _walk_forward_mae() -> list[dict[str, float | int | str]]:
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
            X_train = df.loc[train_mask, FEATURE_NAMES].values
            y_train = y[train_mask.values]
            X_test = df.loc[test_mask, FEATURE_NAMES].values
            y_test = y[test_mask.values]
            window_pipeline = clone(pipeline)
            window_pipeline.fit(X_train, y_train)
            preds = window_pipeline.predict(X_test)
            mae = float(mean_absolute_error(y_test, preds)) if len(y_test) else 0.0
            reports.append({
                'start': test_start.strftime('%Y-%m-%d'),
                'end': test_end.strftime('%Y-%m-%d'),
                'samples': int(len(y_test)),
                'mae': mae,
            })
            test_start = test_end
        return reports

    walk_forward = _walk_forward_mae()
    pipeline.fit(X, y)
    mae_value = float(np.mean(scores)) if scores else None
    payload = {
        'model': pipeline,
        'features': FEATURE_NAMES,
        'model_type': 'regressor',
        'mae': mae_value,
        'walk_forward': walk_forward,
    }
    save_model_with_version(payload, MODEL_PATH, 'stable', metric_name='mae', metric_value=mae_value)
    onnx_result = export_onnx_with_gatekeeper(
        payload=payload,
        model_path=MODEL_PATH,
        model_name='stable',
        feature_names=FEATURE_NAMES,
        metric_name='mae',
        metric_direction='lower',
    )

    return {
        'samples': len(X),
        'mae': mae_value,
        'walk_forward': walk_forward,
        'model_path': str(MODEL_PATH),
        'onnx': onnx_result,
    }


if __name__ == '__main__':
    result = train_stable_model()
    print(result)
