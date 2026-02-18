from __future__ import annotations

import os
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

from ..collectors.news_rss import fetch_news_sentiment


class DataFusionEngine:
    def __init__(self, ticker: str, fast_mode: bool = False):
        self.ticker = (ticker or "").upper().strip()
        self.end_date = datetime.utcnow()
        self.start_date = self.end_date - timedelta(days=365)
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
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            tickers = df.columns.get_level_values(-1)
            if self.ticker in tickers:
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
        df = df.rename(columns={"Adj Close": "Adj_Close"})
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
        return macro.resample("D").ffill()

    def fuse_all(self) -> pd.DataFrame:
        market = self.get_market_data()
        if market.empty:
            return pd.DataFrame()

        if self.fast_mode:
            fused = market.copy()
            fused["sentiment_score"] = 0.0
            fused["news_count"] = 0
        else:
            macro = self.get_macro_data()
            fused = market.join(macro, how="left").ffill()
            sentiment = fetch_news_sentiment(self.ticker)
            fused["sentiment_score"] = sentiment.get("news_sentiment", 0.0)
            fused["news_count"] = sentiment.get("news_count", 0)

        fused["MA20"] = fused["Close"].rolling(window=20, min_periods=10).mean()
        fused["MA50"] = fused["Close"].rolling(window=50, min_periods=20).mean()
        fused["MA200"] = fused["Close"].rolling(window=200, min_periods=60).mean()
        fused["Volatility"] = fused["Returns"].rolling(window=21, min_periods=10).std()
        fused["Momentum20"] = fused["Close"].pct_change(20)
        fused["vol_regime"] = fused["Volatility"] / fused["Volatility"].rolling(window=252, min_periods=60).mean()
        if "Volume" not in fused.columns:
            fused["Volume"] = 0.0
        vol_mean = fused["Volume"].rolling(window=20, min_periods=10).mean()
        vol_std = fused["Volume"].rolling(window=20, min_periods=10).std()
        fused["VolumeZ"] = (fused["Volume"] - vol_mean) / vol_std.replace(0, pd.NA)
        fused["sector_code"] = _get_sector_code(self.ticker)

        delta = fused["Close"].diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=7).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=7).mean()
        rs = gain / loss.replace(0, pd.NA)
        fused["RSI14"] = 100 - (100 / (1 + rs))

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
