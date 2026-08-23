"""Régression (2026-08-23) : découverte en creusant pourquoi le modèle
recommender retrainé (voir recommender_training.py) produisait des
probabilités identiques pour tous les tickers -- portfolio.ml_engine.
processor.py::DataMerger.fetch_price_features faisait `"Close" in data`
sur le DataFrame renvoyé par market_data.Ticker(...).history(), qui
échouait TOUJOURS silencieusement (retournait {}) pour QQQ, RY, ONCY...
en direct sur le vrai wrapper (pas juste en isolant du code) :

    >>> from portfolio import market_data as yf
    >>> h = yf.Ticker('QQQ').history(period='1y', interval='1d')
    >>> list(h.columns)
    [('QQQ', 'Open'), ('QQQ', 'High'), ..., ('QQQ', 'Volume')]
    >>> 'Close' in h
    False

Cause : quand le chemin Alpaca (_bars_for_symbols/_frame_from_bars) ne
renvoie rien, download() retombe sur yfinance.download(), qui renvoie des
colonnes MultiIndex (ticker, champ) même pour un seul symbole selon la
version de yfinance -- alors que le chemin Alpaca ET le DEUXIÈME repli
(_yfinance.Ticker(...).history(), utilisé seulement si download() est
entièrement vide) renvoient tous les deux des colonnes plates. Silencieux
pour de vrai : aucune exception, juste des features de prix manquantes
pour n'importe quel symbole retombant sur ce chemin -- potentiellement
plus large que le seul pipeline recommender (market_data.Ticker est
importé sous l'alias `yf` dans tasks.py, views.py, data_fusion.py,
ai_scout.py, validation_service.py, train_penny_model.py, consumers.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from django.test import TestCase

from portfolio import market_data


def _multiindex_ohlcv_frame(ticker: str, n: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2026-08-01", periods=n, freq="B")
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    columns = pd.MultiIndex.from_product([[ticker], fields])
    data = np.tile(np.arange(1, n + 1, dtype=float).reshape(-1, 1), (1, len(fields)))
    return pd.DataFrame(data, index=dates, columns=columns)


class YfDownloadFallbackFlattensMultiIndexTests(TestCase):
    def test_multiindex_columns_from_yfinance_download_are_flattened(self):
        multiindex_frame = _multiindex_ohlcv_frame("QQQ")
        with patch.object(market_data, "_bars_for_symbols", return_value=pd.DataFrame()), \
                patch.object(market_data, "_allow_yf_price_fallback", return_value=True), \
                patch.object(market_data, "_yfinance", MagicMock(download=MagicMock(return_value=multiindex_frame))):
            result = market_data.download("QQQ", period="1y", interval="1d")

        self.assertFalse(isinstance(result.columns, pd.MultiIndex))
        self.assertIn("Close", result.columns)
        self.assertEqual(list(result["Close"]), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_ticker_history_exposes_flat_close_column_after_yf_fallback(self):
        # Reproduction directe du bug tel qu'observé (DataMerger.
        # fetch_price_features fait exactement `"Close" in data`).
        multiindex_frame = _multiindex_ohlcv_frame("QQQ")
        with patch.object(market_data, "_bars_for_symbols", return_value=pd.DataFrame()), \
                patch.object(market_data, "_allow_yf_price_fallback", return_value=True), \
                patch.object(market_data, "_yfinance", MagicMock(download=MagicMock(return_value=multiindex_frame))):
            data = market_data.Ticker("QQQ").history(period="1y", interval="1d")

        self.assertIn("Close", data)

    def test_already_flat_columns_pass_through_unchanged(self):
        flat_frame = pd.DataFrame(
            {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [100]},
            index=pd.date_range("2026-08-01", periods=1),
        )
        with patch.object(market_data, "_bars_for_symbols", return_value=pd.DataFrame()), \
                patch.object(market_data, "_allow_yf_price_fallback", return_value=True), \
                patch.object(market_data, "_yfinance", MagicMock(download=MagicMock(return_value=flat_frame))):
            result = market_data.download("SPY", period="1y", interval="1d")

        self.assertFalse(isinstance(result.columns, pd.MultiIndex))
        self.assertIn("Close", result.columns)
