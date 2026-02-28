from __future__ import annotations

import os
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

import pandas as pd
import requests
from ... import market_data as yf

from ...alpaca_data import get_daily_bars, get_latest_bid_ask_spread_pct

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
            if self.ticker in level_0:
                df = df.xs(self.ticker, level=0, axis=1)
            elif self.ticker in level_last:
                df = df.xs(self.ticker, level=-1, axis=1)
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
        df = df.rename(columns={"Adj Close": "Adj_Close"})
        df = _normalize_ohlc_columns(df)
        if "Close" not in df.columns:
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

        spread_pct = get_latest_bid_ask_spread_pct(self.ticker)
        if spread_pct is None:
            spread_pct = _fallback_bid_ask_spread_pct(self.ticker)
        fused["bid_ask_spread_pct"] = float(spread_pct or 0.0)

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


@lru_cache(maxsize=256)
def _get_sector_code(ticker: str) -> int:
    try:
        info = yf.Ticker(ticker).info or {}
        sector = (info.get("sector") or "").strip().lower()
    except Exception:
        sector = ""
    return SECTOR_CODES.get(sector, 0)


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

    for idx in range(len(df)):
        row = df.iloc[idx]
        prev = df.iloc[idx - 1] if idx > 0 else None
        prev2 = df.iloc[idx - 2] if idx > 1 else None

        body = abs(float(row["Close"]) - float(row["Open"]))
        candle_range = max(float(row["High"]) - float(row["Low"]), 0.0)
        lower_shadow = min(float(row["Open"]), float(row["Close"])) - float(row["Low"])
        upper_shadow = float(row["High"]) - max(float(row["Open"]), float(row["Close"]))
        if candle_range > 0 and body <= candle_range * 0.1:
            df.at[df.index[idx], "pattern_doji"] = 1
        if body > 0 and lower_shadow >= 2 * body and upper_shadow <= body * 0.5:
            df.at[df.index[idx], "pattern_hammer"] = 1
        if prev is not None:
            prev_red = float(prev["Close"]) < float(prev["Open"])
            curr_green = float(row["Close"]) > float(row["Open"])
            engulfs = float(row["Close"]) >= float(prev["Open"]) and float(row["Open"]) <= float(prev["Close"])
            if prev_red and curr_green and engulfs:
                df.at[df.index[idx], "pattern_engulfing"] = 1
        if prev is not None and prev2 is not None:
            prev2_red = float(prev2["Close"]) < float(prev2["Open"])
            prev_small = abs(float(prev["Close"]) - float(prev["Open"])) <= (
                max(float(prev["High"]) - float(prev["Low"]), 0.0) * 0.3
            )
            curr_green = float(row["Close"]) > float(row["Open"])
            gap_down = float(prev["Close"]) < float(prev2["Close"])
            recover = float(row["Close"]) >= (float(prev2["Open"]) + float(prev2["Close"])) / 2
            if prev2_red and prev_small and curr_green and gap_down and recover:
                df.at[df.index[idx], "pattern_morning_star"] = 1

    return frame.join(df[["pattern_doji", "pattern_hammer", "pattern_engulfing", "pattern_morning_star"]])
