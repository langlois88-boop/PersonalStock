from __future__ import annotations

import os
from datetime import datetime, timedelta
import time
from typing import Optional

import pandas as pd
import requests
from ... import market_data as yf

from ...alpaca_data import get_daily_bars, get_latest_bid_ask_spread_pct, get_order_book_imbalance
from ..feature_engineering import fractional_diff_series
from ..features.technical import add_all_technical_features

from ..collectors.news_rss import fetch_news_sentiment


class DataFusionEngine:
    def __init__(self, ticker: str, fast_mode: bool = False, use_alpaca: bool = False):
        self.ticker = (ticker or "").upper().strip()
        self.end_date = datetime.utcnow()
        self.start_date = self.end_date - timedelta(days=365)
        self.use_alpaca = bool(
            use_alpaca
            or str(os.getenv("DATAFUSION_USE_ALPACA", "")).strip().lower() in {"1", "true", "yes", "y"}
        )
        self.fast_mode = bool(
            fast_mode
            or str(os.getenv("DATAFUSION_FAST_MODE", "")).strip().lower() in {"1", "true", "yes", "y"}
        )

    def _get_market_data_from_db(self) -> pd.DataFrame:
        try:
            from ...models import Stock, PriceHistory
        except Exception:
            return pd.DataFrame()
        stock = Stock.objects.filter(symbol__iexact=self.ticker).first()
        if not stock:
            return pd.DataFrame()
        rows = list(PriceHistory.objects.filter(stock=stock).order_by("date"))
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(
            [{"Date": row.date, "Close": float(row.close_price)} for row in rows if row.close_price is not None]
        )
        if df.empty:
            return pd.DataFrame()
        df = df.set_index("Date")
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        df["Volume"] = 0.0
        df["Returns"] = df["Close"].pct_change()
        return df

    def get_market_data(self) -> pd.DataFrame:
        if self.fast_mode:
            return self._get_market_data_from_db()
        if self.use_alpaca:
            alpaca_df = get_daily_bars(self.ticker, days=730)
            if alpaca_df is not None and not alpaca_df.empty:
                if isinstance(alpaca_df.index, pd.MultiIndex):
                    alpaca_df = alpaca_df.reset_index()
                if 'timestamp' in alpaca_df.columns:
                    alpaca_df = alpaca_df.set_index('timestamp')
                alpaca_df.index = pd.to_datetime(alpaca_df.index, errors="coerce")
                alpaca_df = alpaca_df[~alpaca_df.index.isna()]
                alpaca_df = alpaca_df.rename(
                    columns={
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume',
                    }
                )
                alpaca_df['Returns'] = alpaca_df['Close'].pct_change()
                return alpaca_df
        try:
            df = yf.download(
                self.ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                timeout=8,
            )
        except TypeError:
            df = yf.download(
                self.ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
            )
        if df is None or df.empty:
            try:
                df = yf.Ticker(self.ticker).history(period="2y", auto_adjust=False, timeout=8)
            except TypeError:
                df = yf.Ticker(self.ticker).history(period="2y", auto_adjust=False)
            except Exception:
                df = pd.DataFrame()
        if df is None or df.empty:
            db_df = self._get_market_data_from_db()
            return db_df if db_df is not None and not db_df.empty else pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            level_0 = df.columns.get_level_values(0)
            level_last = df.columns.get_level_values(-1)
            mapped_ticker = getattr(yf, "_map_symbol", None)
            mapped_ticker = mapped_ticker(self.ticker) if callable(mapped_ticker) else self.ticker
            candidates = [self.ticker, mapped_ticker]
            selected = None
            for candidate in candidates:
                if candidate in level_0:
                    selected = (candidate, 0)
                    break
                if candidate in level_last:
                    selected = (candidate, -1)
                    break
            if selected is not None:
                df = df.xs(selected[0], level=selected[1], axis=1)
            else:
                df.columns = [
                    "_".join([str(part) for part in col if part not in (None, "")])
                    for col in df.columns
                ]
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
            if "Date" in df.columns:
                df = df.set_index("Date")
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        df = _normalize_datetime_index(df)
        if df is None or df.empty:
            db_df = self._get_market_data_from_db()
            return db_df if db_df is not None and not db_df.empty else pd.DataFrame()
        df = df.rename(columns={"Adj Close": "Adj_Close"})
        df = _normalize_ohlc_columns(df)
        if df is None or df.empty:
            db_df = self._get_market_data_from_db()
            return db_df if db_df is not None and not db_df.empty else pd.DataFrame()
        if "Close" not in df.columns:
            db_df = self._get_market_data_from_db()
            return db_df if db_df is not None and not db_df.empty else pd.DataFrame()
        if df["Close"].dropna().empty:
            db_df = self._get_market_data_from_db()
            return db_df if db_df is not None and not db_df.empty else pd.DataFrame()
        df["Returns"] = df["Close"].pct_change()
        return df

    def get_macro_data(self) -> pd.DataFrame:
        fred_key = os.getenv("FRED_API_KEY")
        if not fred_key:
            return pd.DataFrame()

        series = ["GS10", "VIXCLS", "CPIAUCSL", "DCOILWTICO"]
        frames = []
        for series_id in series:
            data = self._fetch_fred_series(series_id, fred_key)
            if data is not None and not data.empty:
                frames.append(data.rename(columns={"value": series_id}))

        if not frames:
            return pd.DataFrame()

        macro = pd.concat(frames, axis=1).sort_index()
        if isinstance(macro.index, pd.MultiIndex):
            macro = macro.reset_index()
            if "date" in macro.columns:
                macro = macro.set_index("date")
        macro.index = pd.to_datetime(macro.index, errors="coerce")
        macro = macro[~macro.index.isna()]
        macro = _normalize_datetime_index(macro)
        return macro.resample("D").ffill()

    def fuse_all(self) -> pd.DataFrame:
        market = self.get_market_data()
        if market.empty:
            return pd.DataFrame()

        market = _normalize_datetime_index(market)

        if self.fast_mode:
            fused = market.copy()
            fused["sentiment_score"] = 0.0
            fused["news_count"] = 0
        else:
            macro = self.get_macro_data()
            macro = _normalize_datetime_index(macro)
            fused = market.join(macro, how="left").ffill()
            sentiment_days = int(os.getenv("NEWS_SENTIMENT_DAYS", "7"))
            sentiment = fetch_news_sentiment(self.ticker, days=sentiment_days)
            fused["sentiment_score"] = sentiment.get("news_sentiment", 0.0)
            fused["news_count"] = sentiment.get("news_count", 0)

        fused["MA20"] = fused["Close"].rolling(window=20, min_periods=10).mean()
        fused["MA50"] = fused["Close"].rolling(window=50, min_periods=20).mean()
        fused["MA200"] = fused["Close"].rolling(window=200, min_periods=60).mean()
        fused["EMA9"] = fused["Close"].ewm(span=9, adjust=False).mean()
        fused["EMA20"] = fused["Close"].ewm(span=20, adjust=False).mean()
        fused["price_to_ema9"] = (fused["Close"] - fused["EMA9"]) / fused["EMA9"].replace(0, pd.NA)
        fused["price_to_ema20"] = (fused["Close"] - fused["EMA20"]) / fused["EMA20"].replace(0, pd.NA)
        fused["Volatility"] = fused["Returns"].rolling(window=21, min_periods=10).std()
        fused["Momentum20"] = fused["Close"].pct_change(20)
        fused["vol_regime"] = fused["Volatility"] / fused["Volatility"].rolling(window=252, min_periods=60).mean()
        if "Volume" not in fused.columns:
            fused["Volume"] = 0.0
        vol_mean = fused["Volume"].rolling(window=20, min_periods=10).mean()
        vol_std = fused["Volume"].rolling(window=20, min_periods=10).std()
        fused["VolumeZ"] = (fused["Volume"] - vol_mean) / vol_std.replace(0, pd.NA)
        rvol_base = fused["Volume"].rolling(window=10, min_periods=5).mean()
        fused["RVOL10"] = fused["Volume"] / rvol_base.replace(0, pd.NA)
        fused["VPT"] = (fused["Returns"].fillna(0.0) * fused["Volume"]).cumsum()
        fused["VPT_roc"] = fused["VPT"].pct_change().replace([pd.NA, float('inf'), float('-inf')], 0.0)
        fused["sector_code"] = _get_sector_code(self.ticker)

        if "Open" not in fused.columns:
            fused["Open"] = fused["Close"]
        if "High" not in fused.columns:
            fused["High"] = fused["Close"]
        if "Low" not in fused.columns:
            fused["Low"] = fused["Close"]

        try:
            vol_base = fused["Volume"].rolling(window=10, min_periods=5).mean()
            fused["trade_velocity"] = fused["Volume"] / vol_base.replace(0, pd.NA)
        except Exception:
            fused["trade_velocity"] = 0.0

        delta = fused["Close"].diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=7).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=7).mean()
        rs = gain / loss.replace(0, pd.NA)
        fused["RSI14"] = 100 - (100 / (1 + rs))

        ema_fast = fused["Close"].ewm(span=12, adjust=False).mean()
        ema_slow = fused["Close"].ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9, adjust=False).mean()
        fused["MACD"] = macd
        fused["MACD_SIGNAL"] = signal
        fused["MACD_HIST"] = macd - signal

        if "Open" in fused.columns and "High" in fused.columns and "Low" in fused.columns:
            body = fused["Close"] - fused["Open"]
            range_ = fused["High"] - fused["Low"]
            fused["CandleBody"] = body
            fused["CandleRange"] = range_
            fused["CandleBodyPct"] = body.abs() / range_.replace(0, pd.NA)
            fused["CandleRangePct"] = range_ / fused["Close"].replace(0, pd.NA)
            prev_close = fused["Close"].shift(1)
            fused["gap_pct"] = (fused["Open"] - prev_close) / prev_close.replace(0, pd.NA)
            fused["intraday_range_pct"] = range_ / fused["Open"].replace(0, pd.NA)
            fused["close_pos_in_range"] = (fused["Close"] - fused["Low"]) / (range_.replace(0, pd.NA))
            fused = _add_candlestick_patterns(fused)

        try:
            fused = add_all_technical_features(
                fused,
                close_col="Close",
                high_col="High",
                low_col="Low",
                open_col="Open",
                volume_col="Volume",
            )
        except Exception:
            pass

        if "RSI14" not in fused.columns and "rsi_14" in fused.columns:
            fused["RSI14"] = fused["rsi_14"]
        if "MACD_HIST" not in fused.columns and "macd_hist" in fused.columns:
            fused["MACD_HIST"] = fused["macd_hist"]
        if "bollinger_pct_b" not in fused.columns and "bb_pct_b_20" in fused.columns:
            fused["bollinger_pct_b"] = fused["bb_pct_b_20"]
        if "ADX14" not in fused.columns and "adx_14" in fused.columns:
            fused["ADX14"] = fused["adx_14"]
        if "VolumeZ" not in fused.columns and "volume_zscore_20" in fused.columns:
            fused["VolumeZ"] = fused["volume_zscore_20"]
        if "rubber_band_index" not in fused.columns and "rubber_band_20" in fused.columns:
            fused["rubber_band_index"] = fused["rubber_band_20"]
        if "Momentum20" not in fused.columns and "return_20d" in fused.columns:
            fused["Momentum20"] = fused["return_20d"]
        if "sector_beta_60" not in fused.columns:
            fused["sector_beta_60"] = 0.0
        if "dividend_yield" not in fused.columns:
            fused["dividend_yield"] = 0.0

        spread_pct = get_latest_bid_ask_spread_pct(self.ticker)
        if spread_pct is None:
            spread_pct = _fallback_bid_ask_spread_pct(self.ticker)
        fused["bid_ask_spread_pct"] = float(spread_pct or 0.0)

        imbalance = get_order_book_imbalance(self.ticker)
        fused["order_book_imbalance"] = float(imbalance or 0.0)

        if os.getenv("FRACTIONAL_DIFF_D", "0.0").strip() not in {"0", "0.0", "0.00"}:
            try:
                d_val = float(os.getenv("FRACTIONAL_DIFF_D", "0.4"))
                fused["frac_diff_close"] = fractional_diff_series(fused["Close"], d=d_val)
            except Exception:
                fused["frac_diff_close"] = 0.0
        else:
            fused["frac_diff_close"] = 0.0

        try:
            spy = yf.download(
                "SPY",
                start=self.start_date,
                end=self.end_date,
                progress=False,
                timeout=8,
            )
            tsx = yf.download(
                "^GSPTSE",
                start=self.start_date,
                end=self.end_date,
                progress=False,
                timeout=8,
            )
            spy_close = spy["Close"].dropna() if spy is not None and not spy.empty and "Close" in spy else pd.Series(dtype=float)
            tsx_close = tsx["Close"].dropna() if tsx is not None and not tsx.empty and "Close" in tsx else pd.Series(dtype=float)
            if not spy_close.empty:
                spy_ret = spy_close.pct_change()
                aligned = pd.concat([fused["Returns"], spy_ret], axis=1).dropna()
                if not aligned.empty:
                    fused["spy_corr_60"] = aligned.iloc[:, 0].rolling(60).corr(aligned.iloc[:, 1])
            if not tsx_close.empty:
                tsx_ret = tsx_close.pct_change()
                aligned = pd.concat([fused["Returns"], tsx_ret], axis=1).dropna()
                if not aligned.empty:
                    fused["tsx_corr_60"] = aligned.iloc[:, 0].rolling(60).corr(aligned.iloc[:, 1])
        except Exception:
            pass

        if self.fast_mode:
            return fused.dropna(subset=["Close"])
        return fused.dropna()

    def _fetch_fred_series(self, series_id: str, api_key: str) -> Optional[pd.DataFrame]:
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": self.start_date.strftime("%Y-%m-%d"),
            "observation_end": self.end_date.strftime("%Y-%m-%d"),
        }
        try:
            resp = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params=params,
                timeout=12,
            )
            if resp.status_code != 200:
                return None
            payload = resp.json() or {}
        except Exception:
            return None

        observations = payload.get("observations") or []
        if not observations:
            return None

        df = pd.DataFrame(observations)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.set_index("date")
        return df[["value"]]


SECTOR_CODES = {
    "technology": 1,
    "financial services": 2,
    "consumer defensive": 3,
    "consumer cyclical": 4,
    "healthcare": 5,
    "industrials": 6,
    "energy": 7,
    "basic materials": 8,
    "communication services": 9,
    "utilities": 10,
    "real estate": 11,
}


_SECTOR_CACHE: dict[str, dict[str, float | int]] = {}


def _get_sector_code(ticker: str) -> int:
    ttl_sec = int(os.getenv('SECTOR_CACHE_TTL_SEC', '86400'))
    now = time.time()
    cached = _SECTOR_CACHE.get(ticker)
    if cached and (now - float(cached.get('ts') or 0)) < ttl_sec:
        return int(cached.get('code') or 0)
    try:
        info = yf.Ticker(ticker).info or {}
        sector = (info.get("sector") or "").strip().lower()
    except Exception:
        sector = ""
    code = SECTOR_CODES.get(sector, 0)
    _SECTOR_CACHE[ticker] = {'ts': now, 'code': code}
    return code


def _normalize_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return frame
    if frame.index.tz is not None:
        frame = frame.copy()
        frame.index = frame.index.tz_convert(None)
        return frame
    frame = frame.copy()
    frame.index = frame.index.tz_localize(None)
    return frame


def _normalize_ohlc_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    normalize = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj_close": "Adj_Close",
        "adjclose": "Adj_Close",
        "adj close": "Adj_Close",
        "volume": "Volume",
    }
    rename_map = {}
    for col in frame.columns:
        key = str(col).strip().lower().replace(" ", "_")
        target = normalize.get(key)
        if target and col != target:
            rename_map[col] = target
    if rename_map:
        frame = frame.rename(columns=rename_map)
    if "Close" not in frame.columns and "Adj_Close" in frame.columns:
        frame = frame.copy()
        frame["Close"] = frame["Adj_Close"]
    return frame


def _fallback_bid_ask_spread_pct(symbol: str) -> float | None:
    try:
        info = yf.Ticker(symbol).info or {}
        bid = info.get("bid")
        ask = info.get("ask")
        bid = float(bid) if bid is not None else None
        ask = float(ask) if ask is not None else None
        if bid is None or ask is None or ask <= 0:
            return None
        mid = (bid + ask) / 2
        if mid <= 0:
            return None
        return float((ask - bid) / mid)
    except Exception:
        return None


def _add_candlestick_patterns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    if not {"Open", "High", "Low", "Close"}.issubset(frame.columns):
        return frame
    df = frame.copy()
    df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        return frame

    df["pattern_doji"] = 0
    df["pattern_hammer"] = 0
    df["pattern_engulfing"] = 0
    df["pattern_morning_star"] = 0

    try:
        import pandas_ta as ta

        patterns = ta.cdl_pattern(
            open_=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=["doji", "hammer", "engulfing", "morningstar"],
        )
        if patterns is not None and not patterns.empty:
            if "CDL_DOJI" in patterns:
                df["pattern_doji"] = (patterns["CDL_DOJI"] != 0).astype(int)
            if "CDL_HAMMER" in patterns:
                df["pattern_hammer"] = (patterns["CDL_HAMMER"] != 0).astype(int)
            if "CDL_ENGULFING" in patterns:
                df["pattern_engulfing"] = (patterns["CDL_ENGULFING"] > 0).astype(int)
            if "CDL_MORNINGSTAR" in patterns:
                df["pattern_morning_star"] = (patterns["CDL_MORNINGSTAR"] != 0).astype(int)
            return frame.join(df[["pattern_doji", "pattern_hammer", "pattern_engulfing", "pattern_morning_star"]])
    except Exception:
        pass

    open_arr = df["Open"].to_numpy(dtype=float)
    high_arr = df["High"].to_numpy(dtype=float)
    low_arr = df["Low"].to_numpy(dtype=float)
    close_arr = df["Close"].to_numpy(dtype=float)
    body = np.abs(close_arr - open_arr)
    candle_range = np.maximum(high_arr - low_arr, 0.0)
    lower_shadow = np.minimum(open_arr, close_arr) - low_arr
    upper_shadow = high_arr - np.maximum(open_arr, close_arr)

    doji = (candle_range > 0) & (body <= candle_range * 0.1)
    hammer = (body > 0) & (lower_shadow >= 2 * body) & (upper_shadow <= body * 0.5)

    prev_open = np.roll(open_arr, 1)
    prev_close = np.roll(close_arr, 1)
    prev_high = np.roll(high_arr, 1)
    prev_low = np.roll(low_arr, 1)
    prev_open[0] = prev_close[0] = prev_high[0] = prev_low[0] = np.nan
    prev_red = prev_close < prev_open
    curr_green = close_arr > open_arr
    engulfs = (close_arr >= prev_open) & (open_arr <= prev_close)
    engulfing = prev_red & curr_green & engulfs

    prev2_open = np.roll(open_arr, 2)
    prev2_close = np.roll(close_arr, 2)
    prev2_high = np.roll(high_arr, 2)
    prev2_low = np.roll(low_arr, 2)
    prev2_open[:2] = prev2_close[:2] = prev2_high[:2] = prev2_low[:2] = np.nan
    prev2_red = prev2_close < prev2_open
    prev_range = np.maximum(prev_high - prev_low, 0.0)
    prev_small = np.abs(prev_close - prev_open) <= (prev_range * 0.3)
    gap_down = prev_close < prev2_close
    recover = close_arr >= (prev2_open + prev2_close) / 2
    morning_star = prev2_red & prev_small & curr_green & gap_down & recover

    df["pattern_doji"] = doji.astype(int)
    df["pattern_hammer"] = hammer.astype(int)
    df["pattern_engulfing"] = engulfing.astype(int)
    df["pattern_morning_star"] = morning_star.astype(int)

    return frame.join(df[["pattern_doji", "pattern_hammer", "pattern_engulfing", "pattern_morning_star"]])
