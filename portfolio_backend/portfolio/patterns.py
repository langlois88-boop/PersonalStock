from __future__ import annotations

from typing import Any

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
    delta = df['close'].diff().fillna(0.0)
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))

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
