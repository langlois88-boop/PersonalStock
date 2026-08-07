import numpy as np
import pandas as pd

from portfolio.patterns import (
    add_pattern_columns,
    detect_bearish_engulfing,
    detect_bullish_engulfing,
    detect_doji,
    detect_hammer,
    detect_morning_star,
)


def test_detect_hammer_and_doji_scalar() -> None:
    # Long lower wick, tiny upper wick, small body -> hammer.
    assert detect_hammer(open_p=10.0, high_p=10.06, low_p=8.0, close_p=10.05) is True
    # Wide range, body ~ 0 -> doji.
    assert detect_doji(open_p=10.0, high_p=11.0, low_p=9.0, close_p=10.02) is True


def test_detect_bearish_engulfing_is_the_mirror_of_bullish() -> None:
    prev_bullish = {"open": 10.0, "close": 11.0}  # green
    curr_engulfing = {"open": 11.5, "close": 9.5}  # red, engulfs prev
    assert detect_bearish_engulfing(prev_bullish, curr_engulfing) is True
    assert detect_bullish_engulfing(prev_bullish, curr_engulfing) is False


def test_detect_morning_star() -> None:
    prev2 = {"open": 12.0, "high": 12.2, "low": 10.0, "close": 10.2}  # big red day
    prev = {"open": 10.1, "high": 10.3, "low": 9.9, "close": 10.05}   # small-bodied gap down
    curr = {"open": 10.2, "high": 11.5, "low": 10.1, "close": 11.3}   # strong green, recovers
    assert detect_morning_star(prev2, prev, curr) is True


def test_add_pattern_columns_produces_bullish_and_bearish_engulfing_separately() -> None:
    """Regression test: the old CDL_ENGULFING-based code (data_fusion.py,
    tasks.py) only ever captured the bullish case, silently losing bearish
    engulfing. add_pattern_columns must expose both."""
    df = pd.DataFrame({
        "open":  [10.0, 9.4, 11.5],
        "high":  [10.1, 11.1, 11.6],
        "low":   [9.4, 9.3, 8.9],
        "close": [9.5, 11.0, 9.0],
    })
    # row0: red day (open 10.0 -> close 9.5), just context for row1.
    # row1: opens below row0's close, closes above row0's open, green
    #       -> bullish engulfing of row0.
    # row2: opens above row1's close, closes below row1's open, red
    #       -> bearish engulfing of row1.
    out = add_pattern_columns(df)
    assert out["pattern_bullish_engulfing"].tolist() == [0, 1, 0]
    assert out["pattern_bearish_engulfing"].tolist() == [0, 0, 1]
    # Backward-compat alias still reflects only the bullish case, as before.
    assert out["pattern_engulfing"].tolist() == out["pattern_bullish_engulfing"].tolist()


def test_add_pattern_columns_missing_ohlc_returns_unchanged() -> None:
    df = pd.DataFrame({"close": [1.0, 2.0]})
    out = add_pattern_columns(df)
    assert "pattern_doji" not in out.columns
    assert list(out.columns) == ["close"]


def test_add_pattern_columns_empty_dataframe() -> None:
    df = pd.DataFrame(columns=["open", "high", "low", "close"])
    out = add_pattern_columns(df)
    assert out.empty
