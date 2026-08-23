"""Collecteur Polygon.io (2026-08-23) -- remplace fmp_api.py (mort, 403
depuis le 31 août 2025) et le sentiment RSS/FinBERT (écrasé sans condition
dans l'ancien processor.py::merge()) pour le pipeline recommender. Voir
docstring de polygon_api.py pour le contexte complet."""

from unittest.mock import MagicMock, patch

from portfolio.ml_engine.collectors.polygon_api import (
    fetch_polygon_fundamentals,
    fetch_polygon_sentiment,
)


def _financials_response(equity, net_income, long_term_debt=0.0):
    return {
        "results": [
            {
                "financials": {
                    "balance_sheet": {
                        "equity": {"value": equity},
                        "long_term_debt": {"value": long_term_debt},
                    },
                    "income_statement": {
                        "net_income_loss_attributable_to_parent": {"value": net_income},
                    },
                }
            }
        ]
    }


def _mock_get(json_payload, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_payload
    return resp


class TestFetchPolygonFundamentals:
    def test_computes_roe_and_debt_to_equity_from_real_shape(self, monkeypatch):
        monkeypatch.setenv("POLYGON_API_KEY", "test-key")
        payload = _financials_response(equity=100_000_000.0, net_income=20_000_000.0, long_term_debt=50_000_000.0)
        with patch("portfolio.ml_engine.collectors.polygon_api.requests.get", return_value=_mock_get(payload)):
            result = fetch_polygon_fundamentals("AAPL")

        assert result["roe"] == 20.0
        assert result["debt_to_equity"] == 50.0

    def test_negative_equity_never_produces_a_flipped_positive_roe(self, monkeypatch):
        # Régression directe du cas ONCY vérifié en direct (2026-08-23) :
        # equity et net_income tous deux négatifs -> ratio brut = +607%,
        # une "excellente" ROE trompeuse pour une biotech en détresse.
        monkeypatch.setenv("POLYGON_API_KEY", "test-key")
        payload = _financials_response(equity=-6_167_000.0, net_income=-37_456_000.0)
        with patch("portfolio.ml_engine.collectors.polygon_api.requests.get", return_value=_mock_get(payload)):
            result = fetch_polygon_fundamentals("ONCY")

        assert result == {"roe": None, "debt_to_equity": None}

    def test_missing_long_term_debt_field_defaults_to_zero_not_none(self, monkeypatch):
        monkeypatch.setenv("POLYGON_API_KEY", "test-key")
        payload = {
            "results": [
                {
                    "financials": {
                        "balance_sheet": {"equity": {"value": 100_000_000.0}},
                        "income_statement": {"net_income_loss": {"value": 10_000_000.0}},
                    }
                }
            ]
        }
        with patch("portfolio.ml_engine.collectors.polygon_api.requests.get", return_value=_mock_get(payload)):
            result = fetch_polygon_fundamentals("KEGS")

        assert result["roe"] == 10.0
        assert result["debt_to_equity"] == 0.0

    def test_no_financials_available_returns_default(self, monkeypatch):
        monkeypatch.setenv("POLYGON_API_KEY", "test-key")
        with patch("portfolio.ml_engine.collectors.polygon_api.requests.get", return_value=_mock_get({"results": []})):
            result = fetch_polygon_fundamentals("KEGS")

        assert result == {"roe": None, "debt_to_equity": None}

    def test_missing_api_key_never_makes_a_network_call(self, monkeypatch):
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        with patch("portfolio.ml_engine.collectors.polygon_api.requests.get") as mock_get:
            result = fetch_polygon_fundamentals("AAPL")

        mock_get.assert_not_called()
        assert result == {"roe": None, "debt_to_equity": None}


class TestFetchPolygonSentiment:
    def test_averages_insight_sentiment_for_the_matching_ticker_only(self, monkeypatch):
        monkeypatch.setenv("POLYGON_API_KEY", "test-key")
        from datetime import datetime, timezone

        recent = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        payload = {
            "results": [
                {
                    "published_utc": recent,
                    "insights": [
                        {"ticker": "AAPL", "sentiment": "positive"},
                        {"ticker": "NVDA", "sentiment": "negative"},  # autre ticker -- ignoré
                    ],
                },
                {
                    "published_utc": recent,
                    "insights": [{"ticker": "AAPL", "sentiment": "neutral"}],
                },
            ]
        }
        with patch("portfolio.ml_engine.collectors.polygon_api.requests.get", return_value=_mock_get(payload)):
            result = fetch_polygon_sentiment("AAPL")

        assert result["news_count"] == 2
        assert result["news_sentiment"] == 0.5  # moyenne de positive(1.0) et neutral(0.0)

    def test_articles_older_than_the_window_are_excluded(self, monkeypatch):
        monkeypatch.setenv("POLYGON_API_KEY", "test-key")
        payload = {
            "results": [
                {
                    "published_utc": "2025-01-01T00:00:00Z",  # bien avant la fenêtre de 7 jours
                    "insights": [{"ticker": "ONCY", "sentiment": "positive"}],
                }
            ]
        }
        with patch("portfolio.ml_engine.collectors.polygon_api.requests.get", return_value=_mock_get(payload)):
            result = fetch_polygon_sentiment("ONCY", days=7)

        assert result == {"news_sentiment": 0.0, "news_count": 0}

    def test_no_articles_returns_default(self, monkeypatch):
        monkeypatch.setenv("POLYGON_API_KEY", "test-key")
        with patch("portfolio.ml_engine.collectors.polygon_api.requests.get", return_value=_mock_get({"results": []})):
            result = fetch_polygon_sentiment("KEGS")

        assert result == {"news_sentiment": 0.0, "news_count": 0}

    def test_missing_api_key_never_makes_a_network_call(self, monkeypatch):
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        with patch("portfolio.ml_engine.collectors.polygon_api.requests.get") as mock_get:
            result = fetch_polygon_sentiment("AAPL")

        mock_get.assert_not_called()
        assert result == {"news_sentiment": 0.0, "news_count": 0}
