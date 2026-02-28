from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


MODEL_PATH = Path(__file__).resolve().parent / "data_fusion_brain_v1.pkl"
BLUECHIP_MODEL_PATH = Path(__file__).resolve().parent / "data_fusion_brain_bluechip_v1.pkl"
PENNY_MODEL_PATH = Path(__file__).resolve().parent / "data_fusion_brain_penny_v1.pkl"
CRYPTO_MODEL_PATH = Path(__file__).resolve().parent / "crypto_brain_v1.pkl"
FEATURE_COLUMNS = [
    "MA20",
    "MA50",
    "MA200",
    "EMA9",
    "EMA20",
    "price_to_ema9",
    "price_to_ema20",
    "vol_regime",
    "Volatility",
    "Momentum20",
    "VolumeZ",
    "RVOL10",
    "VPT",
    "VPT_roc",
    "RSI14",
    "MACD_HIST",
    "CandleBodyPct",
    "CandleRangePct",
    "gap_pct",
    "intraday_range_pct",
    "close_pos_in_range",
    "pattern_doji",
    "pattern_hammer",
    "pattern_engulfing",
    "pattern_morning_star",
    "sentiment_score",
    "news_count",
    "VIXCLS",
    "DCOILWTICO",
    "CPIAUCSL",
    "spy_corr_60",
    "tsx_corr_60",
    "bid_ask_spread_pct",
]
CRYPTO_FEATURE_COLUMNS = [
    "return_1",
    "rsi_14",
    "rubber_band_index",
    "price_to_vwap",
    "volatility_spike",
    "btc_correlation",
]

TRIPLE_BARRIER_UP_PCT = float(os.getenv("TRIPLE_BARRIER_UP_PCT", "0.05"))
TRIPLE_BARRIER_DOWN_PCT = float(os.getenv("TRIPLE_BARRIER_DOWN_PCT", "0.03"))
TRIPLE_BARRIER_MAX_DAYS = int(os.getenv("TRIPLE_BARRIER_MAX_DAYS", "20"))
TRIPLE_BARRIER_USE_ATR = os.getenv("TRIPLE_BARRIER_USE_ATR", "true").lower() in {"1", "true", "yes", "y"}
TRIPLE_BARRIER_ATR_MULT_UP = float(os.getenv("TRIPLE_BARRIER_ATR_MULT_UP", "1.5"))
TRIPLE_BARRIER_ATR_MULT_DOWN = float(os.getenv("TRIPLE_BARRIER_ATR_MULT_DOWN", "1.0"))
PENNY_TRIPLE_BARRIER_UP_PCT = float(os.getenv("PENNY_TRIPLE_BARRIER_UP_PCT", "0.03"))
PENNY_TRIPLE_BARRIER_DOWN_PCT = float(os.getenv("PENNY_TRIPLE_BARRIER_DOWN_PCT", "0.02"))
PENNY_TRIPLE_BARRIER_MAX_DAYS = int(os.getenv("PENNY_TRIPLE_BARRIER_MAX_DAYS", "2"))
TIME_SERIES_SPLITS = int(os.getenv("TIME_SERIES_SPLITS", "4"))
TX_COST_PCT = float(os.getenv("BACKTEST_TX_COST_PCT", "0.002"))
SLIPPAGE_PCT = float(os.getenv("BACKTEST_SLIPPAGE_PCT", "0.001"))
FIXED_COST_PER_SHARE = float(os.getenv("BACKTEST_FIXED_COST_PER_SHARE", "0.01"))
MONTE_CARLO_RUNS = int(os.getenv("BACKTEST_MONTE_CARLO_RUNS", "1000"))
RUIN_THRESHOLD_PCT = float(os.getenv("BACKTEST_RUIN_THRESHOLD_PCT", "0.6"))


def get_model_path(universe: str | None = None) -> Path:
    universe_key = (universe or '').strip().upper()
    if universe_key in {'AI_PENNY', 'PENNY', 'PENNY_STOCK', 'PENNY_STOCKS'}:
        return Path(os.getenv('PENNY_MODEL_PATH', str(PENNY_MODEL_PATH)))
    if universe_key in {'AI_BLUECHIP', 'BLUECHIP', 'BLUE_CHIP', 'BLUE'}:
        return Path(os.getenv('BLUECHIP_MODEL_PATH', str(BLUECHIP_MODEL_PATH)))
    if universe_key in {'AI_CRYPTO', 'CRYPTO', 'CRYPTO_ASSET', 'CRYPTO_ASSETS'}:
        return Path(os.getenv('CRYPTO_MODEL_PATH', str(CRYPTO_MODEL_PATH)))
    return Path(os.getenv('BLUECHIP_MODEL_PATH', str(BLUECHIP_MODEL_PATH)))


def get_feature_columns(universe: str | None = None) -> list[str]:
    universe_key = (universe or '').strip().upper()
    if universe_key in {'AI_CRYPTO', 'CRYPTO', 'CRYPTO_ASSET', 'CRYPTO_ASSETS'}:
        return CRYPTO_FEATURE_COLUMNS
    return FEATURE_COLUMNS


def _new_model_version() -> str:
    return datetime.utcnow().strftime('%Y%m%d%H%M%S')


def get_model_version(payload: dict | None, model_path: Path) -> str:
    if payload and payload.get('model_version'):
        return str(payload.get('model_version'))
    try:
        mtime = int(model_path.stat().st_mtime)
        return f"{model_path.name}:{mtime}"
    except Exception:
        return model_path.name


@dataclass
class BacktestResult:
    final_balance: float
    total_return_pct: float
    win_rate: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    recovery_factor: float
    monte_carlo_ruin_pct: float
    max_drawdown: float
    equity_curve: list[float]
    buy_hold_curve: list[float]
    dates: list[str]
    raw_final_balance: float | None = None
    raw_total_return_pct: float | None = None
    raw_win_rate: float | None = None
    raw_sharpe_ratio: float | None = None
    raw_max_drawdown: float | None = None
    raw_equity_curve: list[float] | None = None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _get_feature_weights(symbol: str | None) -> dict[str, float]:
    if symbol and (symbol.endswith('.V') or symbol.endswith('.CN')):
        return {'volume': 0.5, 'sentiment': 0.3, 'macro': 0.1, 'rsi': 0.1}
    return {'macro': 0.4, 'volume': 0.3, 'sentiment': 0.2, 'rsi': 0.1}


def _compute_feature_stats(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for col in ["Volume", "GS10", "sentiment_score", "RSI14"]:
        if col in df.columns:
            series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if series.empty:
                stats[col] = (0.0, 1.0)
            else:
                stats[col] = (float(series.min()), float(series.max()))
        else:
            stats[col] = (0.0, 1.0)
    return stats


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp((value - low) / (high - low))


def _apply_feature_weighting(
    signal: float,
    row: pd.Series,
    weights: dict[str, float],
    stats: dict[str, tuple[float, float]],
) -> float:
    volume = float(row.get("Volume", 0.0) or 0.0)
    gs10 = float(row.get("GS10", 0.0) or 0.0)
    sentiment = float(row.get("sentiment_score", 0.0) or 0.0)
    rsi = float(row.get("RSI14", 50.0) or 50.0)

    vol_min, vol_max = stats.get("Volume", (0.0, 1.0))
    gs_min, gs_max = stats.get("GS10", (0.0, 1.0))
    sen_min, sen_max = stats.get("sentiment_score", (0.0, 1.0))
    rsi_min, rsi_max = stats.get("RSI14", (0.0, 100.0))

    volume_score = _normalize(np.log1p(max(volume, 0.0)), np.log1p(max(vol_min, 0.0)), np.log1p(max(vol_max, 1.0)))
    macro_raw = _normalize(gs10, gs_min, gs_max)
    macro_score = 1.0 - macro_raw
    sentiment_score = _normalize(sentiment, sen_min, sen_max)
    rsi_norm = _normalize(rsi, rsi_min, rsi_max)
    rsi_score = 1.0 - abs(rsi_norm - 0.5) * 2.0

    feature_score = (
        weights.get('volume', 0) * volume_score
        + weights.get('macro', 0) * macro_score
        + weights.get('sentiment', 0) * sentiment_score
        + weights.get('rsi', 0) * rsi_score
    )
    return _clamp(0.7 * signal + 0.3 * feature_score)


def _feature_weighting_enabled() -> bool:
    return os.getenv('FEATURE_WEIGHTING_ENABLED', 'false').lower() in {'1', 'true', 'yes', 'y'}


def _fallback_signal(df: pd.DataFrame) -> pd.Series:
    ma20 = df.get('MA20') if 'MA20' in df.columns else pd.Series(index=df.index, data=np.nan)
    ma50 = df.get('MA50') if 'MA50' in df.columns else pd.Series(index=df.index, data=np.nan)
    rsi = df.get('RSI14') if 'RSI14' in df.columns else pd.Series(index=df.index, data=50.0)

    trend = (ma20 > ma50).fillna(False)
    base = np.where(trend, 0.7, 0.3)

    rsi = pd.to_numeric(rsi, errors='coerce').fillna(50.0)
    rsi_adj = (rsi - 50.0) / 100.0
    signal = np.clip(base + rsi_adj, 0.0, 1.0)
    return pd.Series(signal, index=df.index)


def apply_feature_weighting_to_signal(
    signal: float,
    row: pd.Series,
    symbol: str | None,
    stats: dict[str, tuple[float, float]] | None = None,
) -> float:
    if not _feature_weighting_enabled():
        return signal
    weights = _get_feature_weights(symbol)
    stats = stats or _compute_feature_stats(pd.DataFrame([row]))
    return _apply_feature_weighting(signal, row, weights, stats)


def _ensure_features(df: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    data = df.copy()
    feature_columns = feature_columns or FEATURE_COLUMNS
    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0.0
    if "VolumeZ" in data.columns:
        data["VolumeZ"] = pd.to_numeric(data["VolumeZ"], errors="coerce").fillna(0.0)
    if "sector_code" in data.columns:
        data["sector_code"] = pd.to_numeric(data["sector_code"], errors="coerce").fillna(0).astype(int)
    if "sector_code" in data.columns and "VolumeZ" in data.columns:
        grouped = data.groupby("sector_code")["VolumeZ"]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        data["VolumeZ"] = (data["VolumeZ"] - mean) / std
    return data


def _atr_pct_series(close: pd.Series, high: pd.Series | None, low: pd.Series | None, window: int = 14) -> pd.Series:
    if close is None or close.empty:
        return pd.Series(dtype=float)
    if high is None or low is None or high.empty or low.empty:
        return pd.Series(dtype=float, index=close.index)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=window, min_periods=max(2, window // 2)).mean()
    atr_pct = atr / close.replace(0, np.nan)
    return atr_pct.replace([np.inf, -np.inf], np.nan)


def _resolve_barrier_pcts(
    up_pct: float,
    down_pct: float,
    atr_pct: float | None,
) -> tuple[float, float]:
    if not TRIPLE_BARRIER_USE_ATR:
        return up_pct, down_pct
    if atr_pct is None or np.isnan(atr_pct) or atr_pct <= 0:
        return up_pct, down_pct
    atr_up = atr_pct * TRIPLE_BARRIER_ATR_MULT_UP
    atr_down = atr_pct * TRIPLE_BARRIER_ATR_MULT_DOWN
    return max(up_pct, atr_up), max(down_pct, atr_down)


def _triple_barrier_labels(close: pd.Series, atr_pct: pd.Series | None = None) -> pd.Series:
    if close is None or close.empty:
        return pd.Series(dtype=float)
    prices = close.values
    labels = np.full(len(prices), np.nan)
    for i in range(len(prices) - 1):
        entry = prices[i]
        if entry <= 0:
            continue
        row_atr_pct = float(atr_pct.iloc[i]) if atr_pct is not None and i < len(atr_pct) else None
        up_pct, down_pct = _resolve_barrier_pcts(TRIPLE_BARRIER_UP_PCT, TRIPLE_BARRIER_DOWN_PCT, row_atr_pct)
        upper = entry * (1 + up_pct)
        lower = entry * (1 - down_pct)
        end = min(len(prices), i + TRIPLE_BARRIER_MAX_DAYS + 1)
        hit = 0
        for j in range(i + 1, end):
            if prices[j] >= upper:
                hit = 1
                break
            if prices[j] <= lower:
                hit = 0
                break
        labels[i] = hit
    return pd.Series(labels, index=close.index)


def _triple_barrier_labels_with_bands(
    close: pd.Series,
    high: pd.Series | None,
    low: pd.Series | None,
    up_pct: float,
    down_pct: float,
    max_days: int,
    atr_pct: pd.Series | None = None,
) -> pd.Series:
    if close is None or close.empty:
        return pd.Series(dtype=float)
    prices = close.values
    highs = high.values if high is not None and not high.empty else prices
    lows = low.values if low is not None and not low.empty else prices
    labels = np.full(len(prices), np.nan)
    for i in range(len(prices) - 1):
        entry = prices[i]
        if entry <= 0:
            continue
        row_atr_pct = float(atr_pct.iloc[i]) if atr_pct is not None and i < len(atr_pct) else None
        resolved_up, resolved_down = _resolve_barrier_pcts(up_pct, down_pct, row_atr_pct)
        upper = entry * (1 + resolved_up)
        lower = entry * (1 - resolved_down)
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


def _infer_universe(model_path: Path | None = None) -> str | None:
    if model_path is None:
        return None
    try:
        resolved = model_path.resolve()
    except Exception:
        resolved = model_path
    penny_path = Path(os.getenv('PENNY_MODEL_PATH', str(PENNY_MODEL_PATH)))
    blue_path = Path(os.getenv('BLUECHIP_MODEL_PATH', str(BLUECHIP_MODEL_PATH)))
    crypto_path = Path(os.getenv('CRYPTO_MODEL_PATH', str(CRYPTO_MODEL_PATH)))
    if resolved == penny_path:
        return 'PENNY'
    if resolved == blue_path:
        return 'BLUECHIP'
    if resolved == crypto_path:
        return 'CRYPTO'
    return None


def _select_features(X_df: pd.DataFrame, y: np.ndarray) -> list[str]:
    min_features = int(os.getenv("MIN_FEATURE_SELECTION", "6"))
    try:
        selector = LogisticRegression(
            solver="saga",
            l1_ratio=1.0,
            max_iter=2000,
            tol=1e-3,
            class_weight="balanced",
            random_state=42,
        )
        pipeline = make_pipeline(StandardScaler(), selector)
        pipeline.fit(X_df.values, y)
        coef = selector.coef_[0]
        selected = [col for col, weight in zip(X_df.columns, coef) if abs(weight) > 1e-6]
        if len(selected) < min_features:
            return list(X_df.columns)
        return selected
    except Exception:
        return list(X_df.columns)


def _build_training_set(df: pd.DataFrame, universe: str | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_cols = get_feature_columns(universe)
    data = _ensure_features(df, feature_columns=feature_cols)
    data = data.dropna(subset=["Returns"])
    if data.empty:
        return np.array([]), np.array([]), []

    X_df = data[feature_cols].fillna(0)
    atr_pct = None
    if "Close" in data.columns and "High" in data.columns and "Low" in data.columns:
        atr_pct = _atr_pct_series(data["Close"], data.get("High"), data.get("Low"))
    if "Close" in data.columns:
        is_penny = (universe or '').strip().upper() in {'PENNY', 'AI_PENNY'}
        if is_penny:
            labels = _triple_barrier_labels_with_bands(
                data["Close"],
                data.get("High"),
                data.get("Low"),
                PENNY_TRIPLE_BARRIER_UP_PCT,
                PENNY_TRIPLE_BARRIER_DOWN_PCT,
                PENNY_TRIPLE_BARRIER_MAX_DAYS,
                atr_pct=atr_pct,
            )
        else:
            labels = _triple_barrier_labels(data["Close"], atr_pct=atr_pct)
    else:
        labels = (data["Returns"].shift(-1) > 0).astype(int)
    labels = labels.astype(float)
    mask = labels.notna()
    X_df = X_df[mask]
    y = labels[mask].astype(int).values
    if len(y) < 10 or len(set(y)) < 2:
        return np.array([]), np.array([]), []

    selected = _select_features(X_df, y)
    X = X_df[selected].values
    return X, y, selected


def train_fusion_model(df: pd.DataFrame, model_path: Path = MODEL_PATH) -> dict | None:
    universe = _infer_universe(model_path)
    X, y, selected = _build_training_set(df, universe=universe)
    if X.size == 0 or y.size == 0:
        return None

    cv_scores = []
    if len(y) >= 20:
        tscv = TimeSeriesSplit(n_splits=min(TIME_SERIES_SPLITS, max(2, len(y) // 20)))
        for train_idx, test_idx in tscv.split(X):
            model_cv = RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_split=8,
                min_samples_leaf=10,
                random_state=42,
            )
            model_cv.fit(X[train_idx], y[train_idx])
            score = model_cv.score(X[test_idx], y[test_idx])
            cv_scores.append(float(score))

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_split=8,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X, y)
    payload = {
        "model": model,
        "features": selected,
        "model_version": _new_model_version(),
        "cv_scores": cv_scores,
        "cv_mean": float(np.mean(cv_scores)) if cv_scores else None,
    }
    joblib.dump(payload, model_path)
    return payload


def train_fusion_model_from_labels(
    samples: pd.DataFrame,
    label_col: str = "label",
    model_path: Path = MODEL_PATH,
    save: bool = True,
    sample_weight: list[float] | None = None,
) -> dict | None:
    if samples is None or samples.empty or label_col not in samples.columns:
        return None

    data = _ensure_features(samples.copy())
    X_df = data[FEATURE_COLUMNS].fillna(0)
    y = data[label_col].fillna(0).astype(int).values

    if len(y) < 10 or len(set(y)) < 2:
        return None

    selected = _select_features(X_df, y)
    X_df = X_df[selected]

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_split=8,
        min_samples_leaf=10,
        random_state=42,
    )
    if sample_weight and len(sample_weight) == len(y):
        model.fit(X_df.values, y, sample_weight=sample_weight)
    else:
        model.fit(X_df.values, y)
    payload = {
        "model": model,
        "features": selected,
        "trained_from": "paper_trades",
        "sample_count": int(len(X_df)),
        "model_version": _new_model_version(),
    }
    if save:
        joblib.dump(payload, model_path)
    return payload


def save_model_payload(payload: dict, model_path: Path = MODEL_PATH) -> None:
    if payload and payload.get("model"):
        joblib.dump(payload, model_path)


def load_or_train_model(df: pd.DataFrame, model_path: Path = MODEL_PATH) -> dict | None:
    if model_path.exists():
        try:
            payload = joblib.load(model_path)
            if isinstance(payload, dict) and payload.get("model") and payload.get("features"):
                return payload
            if hasattr(payload, "predict_proba"):
                return {"model": payload, "features": FEATURE_COLUMNS}
        except Exception:
            pass
    return train_fusion_model(df, model_path)


class AIBacktester:
    def __init__(self, data: pd.DataFrame, model: dict | None, symbol: str | None = None):
        self.data = data
        self.model = model
        self.symbol = symbol
        self.initial_balance = 10000.0

    def run_simulation(
        self,
        lookback_days: int | None = None,
        buy_threshold: float = 0.64,
        sell_threshold: float = 0.35,
        stop_loss_pct: float = 0.04,
        atr_multiplier: float = 1.5,
        risk_per_trade_pct: float = 0.015,
        position_cap_pct: float = 0.10,
        tx_cost_pct: float | None = None,
        slippage_pct: float | None = None,
    ) -> BacktestResult:
        df = self.data.copy().sort_index()
        if lookback_days:
            df = df.tail(lookback_days)

        df = _ensure_features(df)
        df = df.dropna(subset=["Returns", "Close"]) if "Close" in df.columns else df.dropna(subset=["Returns"])

        if df.empty or self.model is None:
            return BacktestResult(
                final_balance=self.initial_balance,
                total_return_pct=0.0,
                win_rate=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                profit_factor=0.0,
                recovery_factor=0.0,
                monte_carlo_ruin_pct=0.0,
                max_drawdown=0.0,
                equity_curve=[],
                buy_hold_curve=[],
                dates=[],
            )

        feature_list = self.model.get("features", FEATURE_COLUMNS)
        for col in feature_list:
            if col not in df.columns:
                df[col] = 0.0
        features = df[feature_list].fillna(0).values
        try:
            df["ai_signal"] = self.model["model"].predict_proba(features)[:, 1]
        except Exception:
            df["ai_signal"] = 0.0

        if _feature_weighting_enabled():
            weight_profile = _get_feature_weights(self.symbol)
            stats = _compute_feature_stats(df)
            df["ai_signal"] = df.apply(
                lambda row: _apply_feature_weighting(float(row["ai_signal"]), row, weight_profile, stats),
                axis=1,
            )

        if df["ai_signal"].nunique(dropna=False) <= 1:
            df["ai_signal"] = _fallback_signal(df)

        if {"High", "Low", "Close"}.issubset(df.columns):
            high = df["High"]
            low = df["Low"]
            close = df["Close"]
            tr = pd.concat([
                (high - low).abs(),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            df["ATR14"] = tr.rolling(14).mean()
        else:
            df["ATR14"] = 0.0

        df["position"] = 0
        df["strategy_returns"] = 0.0

        sentiment_constant = True
        if "sentiment_score" in df.columns:
            sentiment_constant = df["sentiment_score"].nunique() <= 1

        entry_price = None
        highest_price = None
        position_weight = 0.0
        position_shares = 0.0
        capital = self.initial_balance
        equity_curve = [1.0]
        cost_pct = (tx_cost_pct if tx_cost_pct is not None else TX_COST_PCT) + (
            slippage_pct if slippage_pct is not None else SLIPPAGE_PCT
        )
        fixed_cost = FIXED_COST_PER_SHARE

        for idx in range(1, len(df)):
            row = df.iloc[idx]
            price = float(row["Close"]) if "Close" in df.columns else None
            ai_signal = float(row["ai_signal"])
            sentiment = float(row.get("sentiment_score", 0.0))
            ma20 = float(row.get("MA20", 0.0))
            ma50 = float(row.get("MA50", 0.0))
            ma200 = float(row.get("MA200", 0.0))
            atr = float(row.get("ATR14", 0.0))
            volume_z = float(row.get("VolumeZ", 0.0) or 0.0)
            min_volume_z = float(os.getenv("VOLUME_ZSCORE_MIN", "0.5"))

            trend_ok = ma20 > ma50 if (ma20 and ma50) else True
            market_ok = price is not None and (not ma200 or price > ma200)
            sentiment_ok = sentiment > 0.1 or sentiment_constant
            volume_ok = volume_z >= min_volume_z
            buy_ok = (ai_signal >= buy_threshold) and sentiment_ok and trend_ok and volume_ok
            exit_ok = ai_signal < sell_threshold

            prev_pos = int(df.iloc[idx - 1]["position"])
            pos = prev_pos

            if prev_pos == 0 and buy_ok and market_ok and price is not None:
                pos = 1
                entry_price = price
                highest_price = price
                stop_distance = max(stop_loss_pct * entry_price, atr_multiplier * atr)
                stop_distance = max(stop_distance, entry_price * 0.01)
                risk_budget = capital * risk_per_trade_pct
                position_cap = capital * position_cap_pct
                position_value = min(capital, position_cap, risk_budget / (stop_distance / entry_price))
                position_weight = position_value / capital if capital else 0.0
                position_shares = position_value / entry_price if entry_price else 0.0
                if position_value > 0:
                    capital -= (position_value * cost_pct) + (position_shares * fixed_cost)
            elif prev_pos == 1:
                if price is not None and highest_price is not None:
                    highest_price = max(highest_price, price)
                stop_distance = max(stop_loss_pct * (entry_price or 0.0), atr_multiplier * atr)
                trailing_stop = (highest_price or 0.0) - stop_distance if highest_price else None
                stop_hit = trailing_stop is not None and price is not None and price <= trailing_stop
                if exit_ok or stop_hit:
                    pos = 0
                    entry_price = None
                    highest_price = None
                    position_weight = 0.0
                    capital -= (capital * cost_pct) + (position_shares * fixed_cost)
                    position_shares = 0.0

            df.iloc[idx, df.columns.get_loc("position")] = pos

            daily_return = float(df.iloc[idx]["Returns"]) if "Returns" in df.columns else 0.0
            strat_ret = position_weight * daily_return if pos == 1 else 0.0
            df.iloc[idx, df.columns.get_loc("strategy_returns")] = strat_ret
            capital = capital * (1 + strat_ret)
            equity_curve.append(capital / self.initial_balance if self.initial_balance else 1.0)

        df["cumulative_returns"] = pd.Series(equity_curve, index=df.index[: len(equity_curve)]).reindex(df.index).ffill().fillna(1.0)
        df["strategy_returns"] = df["strategy_returns"].fillna(0)
        df["cumulative_returns"] = (1 + df["strategy_returns"]).cumprod()

        cumulative = df["cumulative_returns"]
        peak = np.maximum.accumulate(cumulative.values)
        drawdown = (cumulative.values - peak) / peak
        max_drawdown = float(np.min(drawdown)) if len(drawdown) else 0.0

        strat_returns = df["strategy_returns"].values
        mean_ret = float(np.mean(strat_returns)) if len(strat_returns) else 0.0
        std_ret = float(np.std(strat_returns)) if len(strat_returns) else 0.0
        sharpe = float((mean_ret / std_ret) * np.sqrt(252)) if std_ret else 0.0
        downside = np.where(strat_returns < 0, strat_returns, 0.0)
        downside_std = float(np.std(downside)) if len(downside) else 0.0
        sortino = float((mean_ret / downside_std) * np.sqrt(252)) if downside_std else 0.0

        gains = strat_returns[strat_returns > 0].sum() if len(strat_returns) else 0.0
        losses = abs(strat_returns[strat_returns < 0].sum()) if len(strat_returns) else 0.0
        profit_factor = float(gains / losses) if losses else 0.0

        active = df[df["position"] != 0]
        wins = active[active["strategy_returns"] > 0]
        win_rate = float(len(wins) / len(active) * 100) if len(active) else 0.0

        final_balance = float(self.initial_balance * cumulative.iloc[-1]) if len(cumulative) else self.initial_balance
        total_return_pct = float((cumulative.iloc[-1] - 1) * 100) if len(cumulative) else 0.0
        recovery_factor = 0.0
        if max_drawdown:
            recovery_factor = float((final_balance - self.initial_balance) / (abs(max_drawdown) * self.initial_balance))

        monte_carlo_ruin_pct = 0.0
        if MONTE_CARLO_RUNS > 0 and len(strat_returns) > 10:
            ruin_count = 0
            for _ in range(MONTE_CARLO_RUNS):
                sampled = np.random.choice(strat_returns, size=len(strat_returns), replace=True)
                equity = self.initial_balance
                for r in sampled:
                    equity *= (1 + r)
                    if equity <= self.initial_balance * RUIN_THRESHOLD_PCT:
                        ruin_count += 1
                        break
            monte_carlo_ruin_pct = float((ruin_count / MONTE_CARLO_RUNS) * 100)

        raw_position = np.where(df["ai_signal"] > buy_threshold, 1, 0)
        raw_position = np.where(df["ai_signal"] < sell_threshold, 0, raw_position)
        raw_returns = pd.Series(raw_position, index=df.index).shift(1).fillna(0) * df["Returns"].fillna(0)
        raw_cumulative = (1 + raw_returns).cumprod()
        raw_peak = np.maximum.accumulate(raw_cumulative.values)
        raw_drawdown = (raw_cumulative.values - raw_peak) / raw_peak
        raw_max_drawdown = float(np.min(raw_drawdown)) if len(raw_drawdown) else 0.0
        raw_mean = float(np.mean(raw_returns)) if len(raw_returns) else 0.0
        raw_std = float(np.std(raw_returns)) if len(raw_returns) else 0.0
        raw_sharpe = float((raw_mean / raw_std) * np.sqrt(252)) if raw_std else 0.0
        raw_active = pd.Series(raw_position, index=df.index)
        raw_trades = raw_active[raw_active != 0]
        raw_wins = raw_returns[raw_trades.index][raw_returns[raw_trades.index] > 0]
        raw_win_rate = float(len(raw_wins) / len(raw_trades) * 100) if len(raw_trades) else 0.0
        raw_final_balance = float(self.initial_balance * raw_cumulative.iloc[-1]) if len(raw_cumulative) else self.initial_balance
        raw_total_return_pct = float((raw_cumulative.iloc[-1] - 1) * 100) if len(raw_cumulative) else 0.0

        buy_hold_curve: Iterable[float] = []
        if "Close" in df.columns and len(df["Close"]) > 0:
            buy_hold_curve = (df["Close"] / df["Close"].iloc[0]).fillna(1).tolist()

        return BacktestResult(
            final_balance=round(final_balance, 2),
            total_return_pct=round(total_return_pct, 2),
            win_rate=round(win_rate, 2),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            profit_factor=round(profit_factor, 3),
            recovery_factor=round(recovery_factor, 3),
            monte_carlo_ruin_pct=round(monte_carlo_ruin_pct, 2),
            max_drawdown=round(max_drawdown, 4),
            equity_curve=[round(v, 4) for v in cumulative.tolist()],
            buy_hold_curve=[round(v, 4) for v in buy_hold_curve],
            dates=[str(d.date()) if hasattr(d, "date") else str(d) for d in df.index],
            raw_final_balance=round(raw_final_balance, 2),
            raw_total_return_pct=round(raw_total_return_pct, 2),
            raw_win_rate=round(raw_win_rate, 2),
            raw_sharpe_ratio=round(raw_sharpe, 3),
            raw_max_drawdown=round(raw_max_drawdown, 4),
            raw_equity_curve=[round(v, 4) for v in raw_cumulative.tolist()],
        )
