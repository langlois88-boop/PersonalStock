from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def detect_hammer(open_p: float, high_p: float, low_p: float, close_p: float) -> bool:
    body = abs(close_p - open_p)
    if body <= 0:
        return False
    lower_shadow = min(open_p, close_p) - low_p
    upper_shadow = high_p - max(open_p, close_p)
    return lower_shadow >= 2 * body and upper_shadow <= body * 0.5


def detect_shooting_star(open_p: float, high_p: float, low_p: float, close_p: float) -> bool:
    body = abs(close_p - open_p)
    if body <= 0:
        return False
    lower_shadow = min(open_p, close_p) - low_p
    upper_shadow = high_p - max(open_p, close_p)
    return upper_shadow >= 2 * body and lower_shadow <= body * 0.5


def detect_doji(open_p: float, high_p: float, low_p: float, close_p: float) -> bool:
    range_size = max(high_p - low_p, 0)
    body = abs(close_p - open_p)
    if range_size <= 0:
        return False
    return body <= range_size * 0.1


def detect_bullish_engulfing(prev: dict[str, float], curr: dict[str, float]) -> bool:
    prev_red = prev['close'] < prev['open']
    curr_green = curr['close'] > curr['open']
    engulfs = curr['close'] >= prev['open'] and curr['open'] <= prev['close']
    return prev_red and curr_green and engulfs


def detect_bearish_engulfing(prev: dict[str, float], curr: dict[str, float]) -> bool:
    prev_green = prev['close'] > prev['open']
    curr_red = curr['close'] < curr['open']
    engulfs = curr['open'] >= prev['close'] and curr['close'] <= prev['open']
    return prev_green and curr_red and engulfs


def detect_morning_star(prev2: dict[str, float], prev: dict[str, float], curr: dict[str, float]) -> bool:
    """3-candle bullish reversal: a down day, a small-bodied day that gaps
    down, then an up day that recovers back above the first candle's
    midpoint."""
    prev2_red = prev2['close'] < prev2['open']
    prev_range = max(prev['high'] - prev['low'], 0.0)
    prev_small = abs(prev['close'] - prev['open']) <= prev_range * 0.3
    curr_green = curr['close'] > curr['open']
    gap_down = prev['close'] < prev2['close']
    recover = curr['close'] >= (prev2['open'] + prev2['close']) / 2
    return prev2_red and prev_small and curr_green and gap_down and recover


def add_pattern_columns(
    df: pd.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pd.DataFrame:
    """Add per-row candlestick pattern indicator columns to a copy of `df`,
    using this module's canonical detection logic — single source of truth
    for pattern features. Previously duplicated (with divergent formulas —
    a pandas_ta fast path in some places, hand-rolled fallbacks that didn't
    quite agree with each other or with this module's scalar detect_*
    functions in others) across `ml_engine/engine/data_fusion.py` and
    `tasks.py::_add_candlestick_features`; both now call this instead.

    Adds:
    - ``pattern_doji``, ``pattern_hammer``, ``pattern_morning_star`` (0/1)
    - ``pattern_bullish_engulfing``, ``pattern_bearish_engulfing`` (0/1)
    - ``pattern_engulfing`` — alias for ``pattern_bullish_engulfing``, kept
      for backward compatibility with existing consumers (this is what the
      name always meant here: the old CDL_ENGULFING-based code only ever
      captured the bullish case under it, silently losing the bearish one —
      use ``pattern_bearish_engulfing`` to not lose that signal).

    Returns `df` unchanged if the required OHLC columns aren't present.
    """
    if df is None or df.empty:
        return df
    required = {open_col, high_col, low_col, close_col}
    if not required.issubset(df.columns):
        return df

    out = df.copy()
    o = pd.to_numeric(out[open_col], errors='coerce').to_numpy(dtype=float)
    h = pd.to_numeric(out[high_col], errors='coerce').to_numpy(dtype=float)
    lo = pd.to_numeric(out[low_col], errors='coerce').to_numpy(dtype=float)
    c = pd.to_numeric(out[close_col], errors='coerce').to_numpy(dtype=float)

    body = np.abs(c - o)
    candle_range = np.maximum(h - lo, 0.0)
    lower_shadow = np.minimum(o, c) - lo
    upper_shadow = h - np.maximum(o, c)

    doji = (candle_range > 0) & (body <= candle_range * 0.1)
    hammer = (body > 0) & (lower_shadow >= 2 * body) & (upper_shadow <= body * 0.5)

    prev_o, prev_c = np.roll(o, 1), np.roll(c, 1)
    prev_h, prev_lo = np.roll(h, 1), np.roll(lo, 1)
    prev_o[0] = prev_c[0] = prev_h[0] = prev_lo[0] = np.nan
    prev_red = prev_c < prev_o
    prev_green = prev_c > prev_o
    curr_green = c > o
    curr_red = c < o
    engulfs_up = (c >= prev_o) & (o <= prev_c)
    engulfs_down = (o >= prev_c) & (c <= prev_o)
    bullish_engulfing = prev_red & curr_green & engulfs_up
    bearish_engulfing = prev_green & curr_red & engulfs_down

    prev2_o, prev2_c = np.roll(o, 2), np.roll(c, 2)
    prev2_o[:2] = prev2_c[:2] = np.nan
    prev2_red = prev2_c < prev2_o
    prev_range = np.maximum(prev_h - prev_lo, 0.0)
    prev_small = np.abs(prev_c - prev_o) <= prev_range * 0.3
    gap_down = prev_c < prev2_c
    recover = c >= (prev2_o + prev2_c) / 2
    morning_star = prev2_red & prev_small & curr_green & gap_down & recover

    out['pattern_doji'] = doji.astype(int)
    out['pattern_hammer'] = hammer.astype(int)
    out['pattern_bullish_engulfing'] = bullish_engulfing.astype(int)
    out['pattern_bearish_engulfing'] = bearish_engulfing.astype(int)
    out['pattern_engulfing'] = out['pattern_bullish_engulfing']
    out['pattern_morning_star'] = morning_star.astype(int)
    return out


def enrich_bars_with_patterns(frame: pd.DataFrame, rvol_window: int = 20) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    df = frame.copy()
    df = df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
    })
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            return pd.DataFrame()
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    patterns: list[list[str]] = []
    pattern_signal: list[float] = []
    rvols: list[float] = []

    volumes = df['volume'].rolling(rvol_window).mean().shift(1)
    returns = df['close'].pct_change().fillna(0.0)
    vol_series = returns.rolling(20).std().fillna(0.0)
    ema20 = df['close'].ewm(span=20, adjust=False).mean()
    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    delta = df['close'].diff().fillna(0.0)
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))

    vwap_series = pd.Series([pd.NA] * len(df), index=df.index)
    try:
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'], errors='coerce')
        elif isinstance(df.index, pd.DatetimeIndex):
            ts = pd.to_datetime(df.index, errors='coerce')
        else:
            ts = pd.Series([pd.NaT] * len(df), index=df.index)
        session_key = ts.dt.date
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        tp_vol = typical_price * df['volume']
        cum_tp_vol = tp_vol.groupby(session_key).cumsum()
        cum_vol = df['volume'].groupby(session_key).cumsum()
        vwap_series = (cum_tp_vol / cum_vol).fillna(pd.NA)
    except Exception:
        vwap_series = pd.Series([pd.NA] * len(df), index=df.index)

    for idx in range(len(df)):
        row = df.iloc[idx]
        prev_row = df.iloc[idx - 1] if idx > 0 else None
        labels: list[str] = []
        score = 0.0

        if detect_hammer(row['open'], row['high'], row['low'], row['close']):
            labels.append('Hammer')
            score += 1.0
        if detect_shooting_star(row['open'], row['high'], row['low'], row['close']):
            labels.append('Shooting Star')
            score -= 1.0
        if detect_doji(row['open'], row['high'], row['low'], row['close']):
            labels.append('Doji')
            score += 0.2
        if prev_row is not None:
            prev = {'open': float(prev_row['open']), 'close': float(prev_row['close'])}
            curr = {'open': float(row['open']), 'close': float(row['close'])}
            if detect_bullish_engulfing(prev, curr):
                labels.append('Bullish Engulfing')
                score += 1.2
            if detect_bearish_engulfing(prev, curr):
                labels.append('Bearish Engulfing')
                score -= 1.2

        avg_vol = float(volumes.iloc[idx] or 0.0)
        current_vol = float(row['volume'] or 0.0)
        rvol = (current_vol / avg_vol) if avg_vol else 1.0
        if abs(score) > 0 and rvol >= 2.0:
            score *= 1.15

        patterns.append(labels)
        pattern_signal.append(round(score, 3))
        rvols.append(round(rvol, 3))

    df['patterns'] = patterns
    df['pattern_signal'] = pattern_signal
    df['rvol'] = rvols
    df['volatility'] = vol_series
    df['ema20'] = ema20
    df['ema50'] = ema50
    df['rsi14'] = rsi.fillna(0.0)
    df['macd_hist'] = macd_hist.fillna(0.0)
    df['vwap'] = vwap_series
    try:
        df['price_to_vwap'] = (df['close'] - df['vwap']) / df['vwap']
    except Exception:
        df['price_to_vwap'] = 0.0
    return df


def build_pattern_annotations(df: pd.DataFrame) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    if df is None or df.empty:
        return annotations
    for _, row in df.iterrows():
        labels = row.get('patterns') or []
        if not labels:
            continue
        timestamp = row.get('timestamp')
        time_val = None
        try:
            if pd.notna(timestamp):
                time_val = int(pd.Timestamp(timestamp).timestamp())
        except Exception:
            time_val = None
        if time_val is None:
            continue
        signal = float(row.get('pattern_signal') or 0.0)
        rvol = float(row.get('rvol') or 0.0)
        label_text = ', '.join(labels)
        if rvol >= 2:
            label_text = f"{label_text} | RVOL {rvol:.2f}"
        annotations.append({
            'time': time_val,
            'text': label_text,
            'signal': signal,
            'rvol': rvol,
        })
    return annotations
