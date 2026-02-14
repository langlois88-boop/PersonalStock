from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import joblib

from portfolio.models import Stock, MacroIndicator


MODEL_PATH = Path(__file__).resolve().parent / 'stable_brain_v1.pkl'


def _is_valid_symbol(symbol: str) -> bool:
    if not symbol:
        return False
    symbol = symbol.strip().upper()
    if symbol in {'.', '-', '_'}:
        return False
    return bool(re.fullmatch(r"[A-Z0-9\.\-]{1,10}", symbol))


def _build_features(close: pd.Series, volume: pd.Series, spy_close: pd.Series, dividend_yield: float, macro: MacroIndicator | None):
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
        *macro_features,
    ]


def train_stable_model():
    symbols = [s.symbol for s in Stock.objects.all().order_by('symbol') if _is_valid_symbol(s.symbol)]
    if not symbols:
        raise RuntimeError('No stocks found to train on.')

    macro = MacroIndicator.objects.order_by('-date').first()

    X = []
    y = []

    spy = yf.Ticker('SPY').history(period='2y', interval='1d', timeout=10)
    if spy is None or spy.empty or 'Close' not in spy:
        raise RuntimeError('SPY history missing.')
    spy_close = spy['Close']

    for symbol in symbols:
        try:
            data = yf.Ticker(symbol).history(period='2y', interval='1d', timeout=10)
        except Exception:
            continue
        if data is None or data.empty or 'Close' not in data:
            continue
        close = data['Close']
        volume = data['Volume'] if 'Volume' in data else pd.Series([0] * len(close), index=close.index)
        dividend_yield = float(Stock.objects.filter(symbol=symbol).values_list('dividend_yield', flat=True).first() or 0)

        for i in range(220, len(close) - 20):
            window_close = close.iloc[:i + 1]
            window_volume = volume.iloc[:i + 1]
            if len(window_close) < 220:
                continue

            features = _build_features(window_close, window_volume, spy_close, dividend_yield, macro)
            if not features:
                continue
            future_return = float((close.iloc[i + 20] - close.iloc[i]) / close.iloc[i])
            X.append(features)
            y.append(future_return)

    if not X:
        raise RuntimeError('No training samples created.')

    X = np.array(X)
    y = np.array(y)

    model = GradientBoostingRegressor(random_state=42)
    tscv = TimeSeriesSplit(n_splits=4)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        scores.append(mean_absolute_error(y[test_idx], preds))

    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    return {
        'samples': len(X),
        'mae': float(np.mean(scores)) if scores else None,
        'model_path': str(MODEL_PATH),
    }


if __name__ == '__main__':
    result = train_stable_model()
    print(result)
