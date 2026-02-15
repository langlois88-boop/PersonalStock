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


MODEL_PATH = Path(__file__).resolve().parent / "data_fusion_brain_v1.pkl"
BLUECHIP_MODEL_PATH = Path(__file__).resolve().parent / "data_fusion_brain_bluechip_v1.pkl"
PENNY_MODEL_PATH = Path(__file__).resolve().parent / "data_fusion_brain_penny_v1.pkl"
FEATURE_COLUMNS = [
    "MA20",
    "vol_regime",
    "DCOILWTICO",
    "CPIAUCSL",
    "VolumeZ",
]


def get_model_path(universe: str | None = None) -> Path:
    universe_key = (universe or '').strip().upper()
    if universe_key in {'AI_PENNY', 'PENNY', 'PENNY_STOCK', 'PENNY_STOCKS'}:
        return Path(os.getenv('PENNY_MODEL_PATH', str(PENNY_MODEL_PATH)))
    if universe_key in {'AI_BLUECHIP', 'BLUECHIP', 'BLUE_CHIP', 'BLUE'}:
        return Path(os.getenv('BLUECHIP_MODEL_PATH', str(BLUECHIP_MODEL_PATH)))
    return Path(os.getenv('BLUECHIP_MODEL_PATH', str(BLUECHIP_MODEL_PATH)))


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


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for col in FEATURE_COLUMNS:
        if col not in data.columns:
            data[col] = 0.0
    return data


def _build_training_set(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = _ensure_features(df)
    data = data.dropna(subset=["Returns"])
    if data.empty:
        return np.array([]), np.array([]), []

    X_df = data[FEATURE_COLUMNS].fillna(0)
    future_returns = data["Returns"].shift(-1)
    risk_denom = data["Volatility"].replace(0, pd.NA)
    risk_adj = future_returns / risk_denom
    y = (risk_adj > 0).astype(int).values
    if len(y) > 0:
        X_df = X_df.iloc[:-1]
        y = y[:-1]
    corr = pd.Series(index=X_df.columns, dtype=float)
    for col in X_df.columns:
        if X_df[col].nunique() <= 1:
            corr[col] = 0.0
        else:
            corr[col] = float(X_df[col].corr(pd.Series(y))) if len(y) else 0.0

    filtered = corr[abs(corr) >= 0.05].sort_values(key=lambda s: abs(s), ascending=False)
    selected = list(filtered.index[:10]) if not filtered.empty else list(X_df.columns[:10])
    X = X_df[selected].values
    return X, y, selected


def train_fusion_model(df: pd.DataFrame, model_path: Path = MODEL_PATH) -> dict | None:
    X, y, selected = _build_training_set(df)
    if X.size == 0 or y.size == 0:
        return None

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_split=8,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X, y)
    payload = {"model": model, "features": selected, "model_version": _new_model_version()}
    joblib.dump(payload, model_path)
    return payload


def train_fusion_model_from_labels(
    samples: pd.DataFrame,
    label_col: str = "label",
    model_path: Path = MODEL_PATH,
    save: bool = True,
) -> dict | None:
    if samples is None or samples.empty or label_col not in samples.columns:
        return None

    data = _ensure_features(samples.copy())
    X_df = data[FEATURE_COLUMNS].fillna(0)
    y = data[label_col].fillna(0).astype(int).values

    if len(y) < 10 or len(set(y)) < 2:
        return None

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_split=8,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_df.values, y)
    payload = {
        "model": model,
        "features": FEATURE_COLUMNS,
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
        capital = self.initial_balance
        equity_curve = [1.0]

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

        active = df[df["position"] != 0]
        wins = active[active["strategy_returns"] > 0]
        win_rate = float(len(wins) / len(active) * 100) if len(active) else 0.0

        final_balance = float(self.initial_balance * cumulative.iloc[-1]) if len(cumulative) else self.initial_balance
        total_return_pct = float((cumulative.iloc[-1] - 1) * 100) if len(cumulative) else 0.0

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
