from __future__ import annotations

"""Labeling utilities for training datasets.

Single source of truth for triple-barrier labeling. This used to be
implemented three times, with divergent numbers for what was nominally the
same universe:

- ``training/labeling.py::triple_barrier_labels`` (this file) — the base
  primitive, sequential-loop, no ATR support: up 15% / down 7% / 20 days.
- ``pipeline_fixes.py::triple_barrier_labels_adaptive`` — per-universe
  config with ATR widening, ternary (-1/0/1) output: PENNY 5%/3%/5d,
  BLUECHIP 4%/2.5%/15d, CRYPTO 8%/5%/7d.
- ``backtester.py``'s internal ``_triple_barrier_labels`` /
  ``_triple_barrier_labels_with_bands`` — vectorized-with-fallback
  algorithm, ATR widening via ``max(fixed_pct, atr_pct * mult)``, binary
  output: generic/BLUECHIP 5%/3%/20d, PENNY 3%/2%/2d.

All three are now this one module: ``triple_barrier_labels`` is the
primitive (kept vectorized-with-fallback for speed, since that was already
correctness-equivalent to the naive loop it replaces here), and
``triple_barrier_labels_adaptive`` resolves per-universe defaults — from
pipeline_fixes.py's numbers, since those came with a documented rationale
for why they replaced the too-strict earlier values — via the same
``{UNIVERSE}_TRIPLE_BARRIER_*`` env vars pipeline_fixes.py already used, and
output is binary (0/1) to match what every training pathway's classifiers
actually expect (pipeline_fixes.py's ternary output was inconsistent with
that and is not preserved).
"""

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _cfg_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _cfg_int(key: str, default: int) -> int:
    try:
        return int(float(os.getenv(key, str(default))))
    except (TypeError, ValueError):
        return default


def _cfg_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _future_window_extrema(values: np.ndarray, window: int, mode: str) -> np.ndarray:
    """For each index i, the max/min of the next `window` values (i+1..i+window)."""
    if values is None or len(values) == 0:
        return np.array([])
    series = pd.Series(values)
    rev = series.iloc[::-1].shift(1)
    rolled = rev.rolling(window=window, min_periods=1).max() if mode == "max" else rev.rolling(window=window, min_periods=1).min()
    return rolled.iloc[::-1].values


def _atr_pct_series(close: pd.Series, high: pd.Series | None, low: pd.Series | None, period: int = 14) -> pd.Series:
    """Average true range as a fraction of price, for barrier widening."""
    if close is None or close.empty:
        return pd.Series(dtype=float)
    if high is None or low is None or high.empty or low.empty:
        return pd.Series(dtype=float, index=close.index)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=period, min_periods=max(2, period // 2)).mean()
    atr_pct = atr / close.replace(0, np.nan)
    return atr_pct.replace([np.inf, -np.inf], np.nan)


def _resolve_barrier_pcts(
    up_pct: float,
    down_pct: float,
    atr_pct: float | None,
    atr_mult_up: float,
    atr_mult_down: float,
) -> tuple[float, float]:
    """Widen barriers in high-volatility regimes, never narrower than the fixed floor."""
    if atr_pct is None or np.isnan(atr_pct) or atr_pct <= 0:
        return up_pct, down_pct
    return max(up_pct, atr_pct * atr_mult_up), max(down_pct, atr_pct * atr_mult_down)


def triple_barrier_labels(
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    up_pct: float = 0.15,
    down_pct: float = 0.07,
    max_days: int = 20,
    *,
    use_atr_barriers: bool = False,
    atr_mult_up: float = 2.0,
    atr_mult_down: float = 1.5,
    atr_period: int = 14,
) -> pd.Series:
    """Generate binary (0/1) triple-barrier labels.

    For each index i, looks forward up to `max_days` bars: label is 1 if the
    upper barrier (entry * (1 + up_pct)) is hit before the lower barrier
    (entry * (1 - down_pct)), 0 otherwise (lower hit first, neither hit, or
    both hit — resolved chronologically, whichever came first). NaN where
    there isn't enough forward horizon or entry price is invalid.

    Args:
        close: Close price series.
        high: High price series (defaults to close if omitted).
        low: Low price series (defaults to close if omitted).
        up_pct: Upper barrier percentage.
        down_pct: Down barrier percentage.
        max_days: Max look-ahead days.
        use_atr_barriers: If True, widen barriers to
            ``max(fixed_pct, atr_pct * atr_mult)`` in high-volatility
            regimes (needs `high`/`low` to compute ATR; falls back to the
            fixed pct otherwise).
        atr_mult_up: ATR multiplier for the upper barrier.
        atr_mult_down: ATR multiplier for the lower barrier.
        atr_period: ATR lookback window.

    Returns:
        Series of labels (0/1) with NaNs for insufficient horizon.
    """
    if close is None or close.empty:
        return pd.Series(dtype=float)
    prices = close.values.astype(float)
    highs = high.values.astype(float) if high is not None and not high.empty else prices
    lows = low.values.astype(float) if low is not None and not low.empty else prices
    n = len(prices)
    labels = np.full(n, np.nan)

    atr_pct_vals = None
    if use_atr_barriers:
        atr_pct_vals = _atr_pct_series(close, high, low, period=atr_period).values

    future_max = _future_window_extrema(highs, max_days, "max")
    future_min = _future_window_extrema(lows, max_days, "min")

    for i in range(n - 1):
        entry = prices[i]
        if entry <= 0 or np.isnan(entry):
            continue
        row_atr_pct = float(atr_pct_vals[i]) if atr_pct_vals is not None and i < len(atr_pct_vals) else None
        resolved_up, resolved_down = _resolve_barrier_pcts(up_pct, down_pct, row_atr_pct, atr_mult_up, atr_mult_down)
        upper = entry * (1 + resolved_up)
        lower = entry * (1 - resolved_down)
        hit_up = future_max[i] >= upper if i < len(future_max) else False
        hit_down = future_min[i] <= lower if i < len(future_min) else False
        if hit_up and not hit_down:
            labels[i] = 1
        elif hit_down and not hit_up:
            labels[i] = 0
        elif not hit_up and not hit_down:
            labels[i] = 0
        else:
            # Both barriers were touched somewhere in the window — resolve
            # chronologically (whichever is hit first, checking up before
            # down on ties within the same bar).
            end = min(n, i + max_days + 1)
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


def triple_barrier_events(
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    up_pct: float = 0.15,
    down_pct: float = 0.07,
    max_days: int = 20,
    *,
    use_atr_barriers: bool = False,
    atr_mult_up: float = 2.0,
    atr_mult_down: float = 1.5,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Like `triple_barrier_labels`, but also returns each label's `t1` —
    the index at which its barrier was actually resolved (the day a barrier
    was hit, or the last bar of the window on timeout).

    This is what `training.sample_weights` needs: triple-barrier labels
    aren't independent samples — a label's window [t0, t1] overlaps other
    labels' windows and shares the same underlying price moves with them,
    so naive uniform sample weighting overweights redundant, overlapping
    labels (see Lopez de Prado, "Advances in Financial Machine Learning",
    ch. 4). Computing sample weights from that overlap needs to know each
    label's actual window end, not just its 0/1 outcome.

    Always walks day-by-day (not the vectorized fast path
    `triple_barrier_labels` uses) since the window's end is itself what's
    being computed here, not just whether a barrier was hit — fine for the
    dataset sizes this trains on (this is the same algorithm the old
    per-file implementations used before they were unified, just also
    recording where each label resolved).

    Returns:
        DataFrame indexed like `close`, with columns:
        - ``label``: 0/1, NaN where there isn't enough forward horizon.
        - ``t1``: index (same type as `close.index`) of the window's end,
          NaT/NaN where `label` is NaN.
    """
    if close is None or close.empty:
        return pd.DataFrame({"label": pd.Series(dtype=float), "t1": pd.Series(dtype=object)})
    prices = close.values.astype(float)
    highs = high.values.astype(float) if high is not None and not high.empty else prices
    lows = low.values.astype(float) if low is not None and not low.empty else prices
    n = len(prices)
    labels = np.full(n, np.nan)
    t1_pos = np.full(n, -1, dtype=int)

    atr_pct_vals = None
    if use_atr_barriers:
        atr_pct_vals = _atr_pct_series(close, high, low, period=atr_period).values

    for i in range(n - 1):
        entry = prices[i]
        if entry <= 0 or np.isnan(entry):
            continue
        row_atr_pct = float(atr_pct_vals[i]) if atr_pct_vals is not None and i < len(atr_pct_vals) else None
        resolved_up, resolved_down = _resolve_barrier_pcts(up_pct, down_pct, row_atr_pct, atr_mult_up, atr_mult_down)
        upper = entry * (1 + resolved_up)
        lower = entry * (1 - resolved_down)
        end = min(n, i + max_days + 1)
        hit = 0
        hit_pos = end - 1  # timeout: window resolves at the last bar checked
        for j in range(i + 1, end):
            if highs[j] >= upper:
                hit = 1
                hit_pos = j
                break
            if lows[j] <= lower:
                hit = 0
                hit_pos = j
                break
        labels[i] = hit
        t1_pos[i] = hit_pos

    t1 = pd.Series(pd.NaT, index=close.index, dtype=object)
    valid = t1_pos >= 0
    t1.iloc[valid] = close.index[t1_pos[valid]]
    return pd.DataFrame({"label": pd.Series(labels, index=close.index), "t1": t1})


@dataclass(frozen=True)
class _BarrierConfig:
    up_pct: float
    down_pct: float
    max_days: int
    atr_mult_up: float
    atr_mult_down: float


_UNIVERSE_ALIASES = {
    "PENNY": {"PENNY", "AI_PENNY", "PENNY_STOCK", "PENNY_STOCKS"},
    "CRYPTO": {"CRYPTO", "AI_CRYPTO", "CRYPTO_ASSET", "CRYPTO_ASSETS"},
    "BLUECHIP": {"BLUECHIP", "AI_BLUECHIP", "BLUE_CHIP", "BLUE", "STABLE"},
}


def _normalize_universe(universe: str | None) -> str:
    key = (universe or "").strip().upper()
    for canonical, aliases in _UNIVERSE_ALIASES.items():
        if key in aliases:
            return canonical
    return "BLUECHIP"


def _universe_defaults(universe: str | None) -> _BarrierConfig:
    key = _normalize_universe(universe)
    if key == "PENNY":
        return _BarrierConfig(
            up_pct=_cfg_float("PENNY_TRIPLE_BARRIER_UP_PCT", 0.05),
            down_pct=_cfg_float("PENNY_TRIPLE_BARRIER_DOWN_PCT", 0.03),
            max_days=_cfg_int("PENNY_TRIPLE_BARRIER_MAX_DAYS", 5),
            atr_mult_up=_cfg_float("PENNY_BARRIER_ATR_MULT_UP", 2.0),
            atr_mult_down=_cfg_float("PENNY_BARRIER_ATR_MULT_DOWN", 1.5),
        )
    if key == "CRYPTO":
        return _BarrierConfig(
            up_pct=_cfg_float("CRYPTO_TRIPLE_BARRIER_UP_PCT", 0.08),
            down_pct=_cfg_float("CRYPTO_TRIPLE_BARRIER_DOWN_PCT", 0.05),
            max_days=_cfg_int("CRYPTO_TRIPLE_BARRIER_MAX_DAYS", 7),
            atr_mult_up=_cfg_float("CRYPTO_BARRIER_ATR_MULT_UP", 2.0),
            atr_mult_down=_cfg_float("CRYPTO_BARRIER_ATR_MULT_DOWN", 1.5),
        )
    return _BarrierConfig(
        up_pct=_cfg_float("BLUECHIP_TRIPLE_BARRIER_UP_PCT", 0.04),
        down_pct=_cfg_float("BLUECHIP_TRIPLE_BARRIER_DOWN_PCT", 0.025),
        max_days=_cfg_int("BLUECHIP_TRIPLE_BARRIER_MAX_DAYS", 15),
        atr_mult_up=_cfg_float("BLUECHIP_BARRIER_ATR_MULT_UP", 2.5),
        atr_mult_down=_cfg_float("BLUECHIP_BARRIER_ATR_MULT_DOWN", 1.5),
    )


def universe_max_days(universe: str | None) -> int:
    """Default triple-barrier horizon (days) for a universe — e.g. for
    sizing a purged CV window so no training fold overlaps a label's
    forward-looking window."""
    return _universe_defaults(universe).max_days


def _resolve_universe_params(
    universe: str,
    up_pct: float | None,
    down_pct: float | None,
    max_days: int | None,
    use_atr_barriers: bool | None,
) -> dict:
    cfg = _universe_defaults(universe)
    atr_enabled = _cfg_bool("TRIPLE_BARRIER_USE_ATR", True) if use_atr_barriers is None else use_atr_barriers
    return dict(
        up_pct=up_pct if up_pct is not None else cfg.up_pct,
        down_pct=down_pct if down_pct is not None else cfg.down_pct,
        max_days=max_days if max_days is not None else cfg.max_days,
        use_atr_barriers=atr_enabled,
        atr_mult_up=cfg.atr_mult_up,
        atr_mult_down=cfg.atr_mult_down,
    )


def _log_label_balance(labels: pd.Series, universe: str) -> None:
    valid = labels.dropna()
    if len(valid) <= 10:
        return
    pos_pct = (valid == 1).sum() / len(valid) * 100
    logger.debug("Labels %s: positive=%.1f%% (n=%d)", universe, pos_pct, len(valid))
    if pos_pct < 10:
        logger.warning(
            "⚠️ Labels très déséquilibrés pour %s: seulement %.1f%% positifs. "
            "Considérer d'ajuster les seuils *_TRIPLE_BARRIER_* ou "
            "use_atr_barriers=True.", universe, pos_pct,
        )


def triple_barrier_labels_adaptive(
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    universe: str = "BLUECHIP",
    up_pct: float | None = None,
    down_pct: float | None = None,
    max_days: int | None = None,
    use_atr_barriers: bool | None = None,
    atr_period: int = 14,
) -> pd.Series:
    """``triple_barrier_labels`` with per-universe defaults (PENNY, CRYPTO,
    BLUECHIP/STABLE), configurable via ``{UNIVERSE}_TRIPLE_BARRIER_UP_PCT``,
    ``{UNIVERSE}_TRIPLE_BARRIER_DOWN_PCT``, ``{UNIVERSE}_TRIPLE_BARRIER_MAX_DAYS``,
    and ``{UNIVERSE}_BARRIER_ATR_MULT_UP``/``_DOWN`` env vars — explicit
    `up_pct`/`down_pct`/`max_days` arguments still take priority over both.

    Returns binary (0/1) labels — see `triple_barrier_labels`.
    """
    params = _resolve_universe_params(universe, up_pct, down_pct, max_days, use_atr_barriers)
    labels = triple_barrier_labels(close, high, low, atr_period=atr_period, **params)
    _log_label_balance(labels, universe)
    return labels


def triple_barrier_events_adaptive(
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    universe: str = "BLUECHIP",
    up_pct: float | None = None,
    down_pct: float | None = None,
    max_days: int | None = None,
    use_atr_barriers: bool | None = None,
    atr_period: int = 14,
) -> pd.DataFrame:
    """``triple_barrier_events`` with the same per-universe defaults as
    `triple_barrier_labels_adaptive` — use this instead when you need `t1`
    (e.g. for `training.sample_weights`), same label semantics either way.
    """
    params = _resolve_universe_params(universe, up_pct, down_pct, max_days, use_atr_barriers)
    events = triple_barrier_events(close, high, low, atr_period=atr_period, **params)
    _log_label_balance(events["label"], universe)
    return events


def forward_return_labels(close: pd.Series, horizon: int = 8, target_pct: float = 0.02) -> pd.Series:
    """Simple forward return label.

    Args:
        close: Close price series.
        horizon: Forward steps.
        target_pct: Target return threshold.

    Returns:
        Series of binary labels.
    """
    future = close.shift(-horizon)
    future_return = (future / close) - 1.0
    return (future_return >= target_pct).astype(int)
