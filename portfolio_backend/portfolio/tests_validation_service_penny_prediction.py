"""Régression (2026-08-23) : ValidationService._penny_prediction appelait
train_penny_model._build_features(data) avec le DataFrame OHLCV complet
en UN seul argument, alors que la fonction exige (close, volume,
sentiment_score) -- un TypeError avalé silencieusement par le except
Exception, donc _penny_prediction (et DanasConsensusView qui l'affiche)
retournait TOUJOURS None peu importe la qualité de la donnée de prix.
Trouvé en vérifiant en direct l'impact du fix Alpaca/MultiIndex du même
jour -- la donnée de prix était enfin bonne, mais la prédiction restait
None pour une raison complètement indépendante."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from django.test import TestCase

from portfolio.services.validation_service import ValidationService


def _synthetic_penny_history(days: int = 60) -> pd.DataFrame:
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="B")
    n = len(dates)
    rng = np.random.RandomState(0)
    close = 0.50 + np.cumsum(rng.randn(n) * 0.01)
    close = np.clip(close, 0.05, None)
    volume = rng.uniform(100_000, 500_000, n)
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": volume},
        index=dates,
    )


class PennyPredictionTests(TestCase):
    def _service_with_fake_model(self, predict_proba_return=None, predict_return=None):
        service = ValidationService()
        model = MagicMock()
        if predict_proba_return is not None:
            model.predict_proba.return_value = predict_proba_return
        else:
            del model.predict_proba  # force le repli sur .predict
            model.predict.return_value = predict_return
        service._penny_model_cache = {
            "model": model,
            "features": ["rsi_14", "sma_ratio_10_20", "volatility_20", "volume_zscore_20", "rvol_20", "return_5d", "sentiment_score"],
        }
        return service, model

    def test_calls_build_features_with_close_volume_sentiment_not_the_raw_dataframe(self):
        service, model = self._service_with_fake_model(predict_proba_return=np.array([[0.3, 0.7]]))
        history = _synthetic_penny_history()

        with patch("portfolio.services.validation_service.yf.Ticker") as mock_ticker, \
                patch(
                    "portfolio.services.validation_service.fetch_news_sentiment",
                    return_value={"news_sentiment": 0.25, "news_count": 3},
                ):
            mock_ticker.return_value.history.return_value = history
            result = service._penny_prediction("TEAD")

        self.assertIsNotNone(result)  # ne devrait plus jamais planter en TypeError silencieux
        self.assertAlmostEqual(result, 0.7)

    def test_falls_back_to_predict_when_predict_proba_unavailable(self):
        service, model = self._service_with_fake_model(predict_return=np.array([0.42]))
        history = _synthetic_penny_history()

        with patch("portfolio.services.validation_service.yf.Ticker") as mock_ticker, \
                patch(
                    "portfolio.services.validation_service.fetch_news_sentiment",
                    return_value={"news_sentiment": 0.0, "news_count": 0},
                ):
            mock_ticker.return_value.history.return_value = history
            result = service._penny_prediction("TEAD")

        self.assertAlmostEqual(result, 0.42)

    def test_no_model_loaded_returns_none_without_network_call(self):
        service = ValidationService()
        service._penny_model_cache = None
        with patch.object(service, "_load_penny_model", return_value=None), \
                patch("portfolio.services.validation_service.yf.Ticker") as mock_ticker:
            result = service._penny_prediction("TEAD")

        mock_ticker.assert_not_called()
        self.assertIsNone(result)

    def test_empty_price_history_returns_none(self):
        service, _ = self._service_with_fake_model(predict_proba_return=np.array([[0.5, 0.5]]))
        with patch("portfolio.services.validation_service.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = pd.DataFrame()
            result = service._penny_prediction("TEAD")

        self.assertIsNone(result)
