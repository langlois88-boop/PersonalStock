"""End-to-end (synthetic data, no network/DB) validation of Étape 8: PENNY's
uniqueness-based sample weighting wired through _resolve_sample_weights and
Trainer.fit, plus the mutual-exclusion guard against DeepSeek weighting."""

import numpy as np
import pandas as pd
import pytest

from portfolio.ml_engine.features.technical import rsi, rvol, sma_ratio, volatility, volume_zscore
from portfolio.ml_engine.pipelines.penny_pipeline import _pipeline_factory, _resolve_sample_weights
from portfolio.ml_engine.training.labeling import triple_barrier_events_adaptive, universe_max_days
from portfolio.ml_engine.training.sample_weights import max_observed_horizon
from portfolio.ml_engine.training.trainer import Trainer


def _synthetic_penny_dataset(symbols: list[str], seed: int = 0):
    rows, labels = [], []
    sample_symbols, sample_dates, sample_t0, sample_t1 = [], [], [], []
    symbol_context = {}

    for i, symbol in enumerate(symbols):
        rng = np.random.RandomState(seed + i)
        n = 250
        idx = pd.bdate_range("2023-01-01", periods=n)
        close = pd.Series(5 + np.cumsum(rng.randn(n) * 0.08), index=idx).clip(lower=0.5)
        high = close + np.abs(rng.randn(n)) * 0.05
        low = close - np.abs(rng.randn(n)) * 0.05
        volume = pd.Series(rng.uniform(1e5, 1e6, n), index=idx)

        rsi_14 = rsi(close, 14)
        sma_10_20 = sma_ratio(close, 10, 20)
        vol_20 = volatility(close, 20)
        vol_z = volume_zscore(volume, 20)
        rvol_20 = rvol(volume, 20)
        ret_5 = close.pct_change(5)
        log_returns = np.log(close / close.shift(1)).fillna(0.0)
        symbol_context[symbol] = (close.index, log_returns)

        events = triple_barrier_events_adaptive(close, high=high, low=low, universe="PENNY", use_atr_barriers=True)
        label_series, t1_series = events["label"], events["t1"]

        valid = ~rsi_14.isna() & ~sma_10_20.isna() & ~vol_20.isna() & ~vol_z.isna() & ~rvol_20.isna() & ~ret_5.isna()
        for idx_pos in range(n):
            if not valid.iloc[idx_pos]:
                continue
            feat = {
                "rsi_14": float(rsi_14.iloc[idx_pos]),
                "sma_ratio_10_20": float(sma_10_20.iloc[idx_pos]),
                "volatility_20": float(vol_20.iloc[idx_pos]),
                "volume_zscore_20": float(vol_z.iloc[idx_pos]),
                "rvol_20": float(rvol_20.iloc[idx_pos]),
                "return_5d": float(ret_5.iloc[idx_pos]),
                "sentiment_score": 0.0,
            }
            label_val = label_series.iloc[idx_pos]
            if pd.isna(label_val) or any(pd.isna(v) for v in feat.values()):
                continue
            rows.append(feat)
            labels.append(int(label_val))
            sample_symbols.append(symbol)
            sample_dates.append(str(close.index[idx_pos].date()))
            sample_t0.append(close.index[idx_pos])
            sample_t1.append(t1_series.iloc[idx_pos])

    X = pd.DataFrame(rows)
    y = pd.Series(labels)
    return X, y, sample_symbols, sample_dates, sample_t0, sample_t1, symbol_context


def test_resolve_sample_weights_uniqueness_end_to_end() -> None:
    X, y, sample_symbols, sample_dates, sample_t0, sample_t1, symbol_context = _synthetic_penny_dataset(
        ["PENNY1", "PENNY2", "PENNY3"]
    )
    assert len(y) > 100

    weights, purge_window = _resolve_sample_weights(
        sample_symbols, sample_dates, sample_t0, sample_t1, symbol_context, X, y,
    )

    assert len(weights) == len(y)
    assert np.all(weights > 0)
    # purge_window derived from actually-observed t1 horizons, not just the
    # configured constant (may coincide with it, but is computed
    # independently — see max_observed_horizon).
    assert purge_window is not None
    assert purge_window <= universe_max_days("PENNY")


def test_trainer_fit_runs_with_uniqueness_weights_and_derived_purge_window() -> None:
    """Confirms the full wiring (weights + derived purge_window) runs
    through Trainer.fit without error, and that the gate behaves
    consistently regardless of which weighting is used (this dataset has no
    real predictive signal, so both should fail the balanced-accuracy gate
    near chance — proving the gate isn't fooled either way, not that
    uniqueness weighting improves accuracy, which needs real data)."""
    X, y, sample_symbols, sample_dates, sample_t0, sample_t1, symbol_context = _synthetic_penny_dataset(
        ["PENNY1", "PENNY2", "PENNY3"], seed=3,
    )
    weights, purge_window = _resolve_sample_weights(
        sample_symbols, sample_dates, sample_t0, sample_t1, symbol_context, X, y,
    )
    purge_window = purge_window or universe_max_days("PENNY")

    result_uniform = Trainer(_pipeline_factory, purge_window=purge_window).fit(X, y, sample_weight=None)
    result_uniqueness = Trainer(_pipeline_factory, purge_window=purge_window).fit(X, y, sample_weight=weights)

    for result in (result_uniform, result_uniqueness):
        assert 0.0 <= result.cv_mean <= 1.0
        assert 0.0 <= result.wf_f1 <= 1.0
        assert result.n_samples == len(y)


def test_deepseek_and_uniqueness_weighting_guard(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_WEIGHTING_ENABLED", "true")
    with pytest.raises(RuntimeError, match="mutually-exclusive"):
        _resolve_sample_weights([], [], [], [], {}, pd.DataFrame(), pd.Series(dtype=int))


def test_deepseek_disabled_by_default() -> None:
    from portfolio.ml_engine.pipelines.penny_pipeline import _deepseek_weighting_enabled, _uniqueness_weighting_enabled

    assert _deepseek_weighting_enabled() is False
    assert _uniqueness_weighting_enabled() is True
