from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from .. import market_data as yf

from .collectors.polygon_api import fetch_fundamentals, fetch_polygon_sentiment
from .collectors.fred_api import fetch_fred_latest


class MacroImputerMixin:
    MACRO_DEFAULTS: Dict[str, float] = {
        'fred_rate': 0.0,
    }

    def _impute_macro_features(self, payload: Dict[str, Optional[float]]) -> Dict[str, float]:
        cleaned: Dict[str, float] = {}
        for key, default in self.MACRO_DEFAULTS.items():
            value = payload.get(key)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                cleaned[key] = float(default)
            else:
                cleaned[key] = float(value)
        return cleaned


@dataclass
class DataMerger(MacroImputerMixin):
    fred_rate_series: str = "GS10"

    def _rsi(self, series: pd.Series, window: int = 14) -> float:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_val = 100 - (100 / (1 + rs))
        return float(rsi_val.iloc[-1])

    def _zscore_volatility(self, series: pd.Series, window: int = 20) -> float:
        mean = series.rolling(window).mean().iloc[-1]
        std = series.rolling(window).std().iloc[-1]
        if std == 0 or pd.isna(std):
            return 0.0
        return float((series.iloc[-1] - mean) / std)

    def fetch_price_features(self, symbol: str) -> Dict[str, Optional[float]]:
        data = yf.Ticker(symbol).history(period="1y", interval="1d", timeout=10)
        if data is None or data.empty or "Close" not in data:
            return {}

        close = data["Close"].dropna()
        if len(close) < 60:
            return {}

        return {
            "rsi_14": self._rsi(close, 14),
            "vol_zscore": self._zscore_volatility(close, 20),
            "return_20d": float(np.log(close.iloc[-1] / close.iloc[-21])),
        }

    def fetch_macro_features(self) -> Dict[str, Optional[float]]:
        rate = fetch_fred_latest(self.fred_rate_series)
        return {"fred_rate": rate}

    def fetch_fundamental_features(self, symbol: str) -> Dict[str, Optional[float]]:
        # Polygon (2026-08-23) remplace FMP (mort, v3 -> 403 depuis le
        # 31 août 2025) ET le sentiment RSS/FinBERT (de toute façon écrasé
        # sans condition par ce dernier avant ce changement -- voir
        # docstring de collectors/polygon_api.py). Cache renommé
        # (fmp_fundamentals -> polygon_fundamentals) pour ne jamais lire
        # une entrée fmp_fundamentals:* mise en cache 24h avant ce
        # changement -- même précaution que le cache technical_trend
        # trouvé stale plus tôt cette session.
        cache_backend = None
        try:
            from django.core.cache import cache as django_cache

            cache_backend = django_cache
        except Exception:
            cache_backend = None

        cache_key = f"polygon_fundamentals:{symbol}"
        if cache_backend is not None:
            cached = cache_backend.get(cache_key)
            if cached is not None:
                return cached

        fundamentals = fetch_fundamentals(symbol)
        sentiment = fetch_polygon_sentiment(symbol)
        payload = {**fundamentals, **sentiment}
        if cache_backend is not None:
            cache_backend.set(cache_key, payload, timeout=60 * 60 * 24)
        return payload

    def merge(self, symbol: str) -> Dict[str, Optional[float]]:
        symbol = (symbol or "").upper().strip()
        if not symbol:
            return {}

        macro_features = self._impute_macro_features(self.fetch_macro_features())
        features = {
            "symbol": symbol,
            **self.fetch_price_features(symbol),
            **self.fetch_fundamental_features(symbol),
            **macro_features,
        }
        return features
