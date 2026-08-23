"""_fetch_polygon_penny_screener (2026-08-23) : remplace le company-
screener FMP dans run_penny_ai_scout, mort en 402 'Restricted Endpoint'
sur le plan Basic gratuit. Vérifié en direct avant d'écrire ce code :
l'endpoint 'grouped daily' de Polygon (clôtures + volume de ~12 500
titres US en un seul appel) fonctionne sur ce plan, contrairement au
snapshot temps réel gainers/losers (402 lui aussi). Voir docstring de la
fonction dans tasks.py pour le contexte complet (dont la comparaison
avec les paliers de repli Yahoo/yfinance existants : seulement 7/703
résultats de leurs catégories génériques passaient les filtres penny en
vérification directe)."""

from __future__ import annotations

from unittest.mock import patch

from django.test import TestCase

from portfolio import tasks


def _grouped_daily_payload(rows):
    return {"results": rows}


def _ticker_details_payload(**overrides):
    base = {"type": "CS", "name": "Test Co", "primary_exchange": "XNAS", "market_cap": 20_000_000.0}
    base.update(overrides)
    return {"results": base}


class FetchPolygonPennyScreenerTests(TestCase):
    def test_filters_by_price_and_volume_from_grouped_daily(self):
        rows = [
            {"T": "PENNYA", "c": 0.50, "v": 500_000},   # passe
            {"T": "TOOHIGH", "c": 25.0, "v": 500_000},  # prix trop élevé
            {"T": "TOOTHIN", "c": 0.50, "v": 1_000},    # volume trop faible
        ]
        with patch.object(tasks, "os") as mock_os, \
                patch.object(tasks, "_fetch_json") as mock_fetch:
            mock_os.getenv.side_effect = lambda key, default=None: {
                "POLYGON_API_KEY": "test-key",
            }.get(key, default)
            mock_fetch.side_effect = [
                _grouped_daily_payload(rows),
                _ticker_details_payload(),
            ]
            result = tasks._fetch_polygon_penny_screener(
                min_price=0.01, max_price=1.0, min_volume=100_000, max_market_cap=50_000_000, max_symbols=10,
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["symbol"], "PENNYA")

    def test_excludes_non_common_stock_types(self):
        # Régression directe : TMCWW (warrant) vu dans les résultats Yahoo
        # en vérification directe -- un vrai screener penny ne devrait pas
        # inclure des warrants/ETF sous le couvert d'un "titre à 2 cents".
        rows = [{"T": "WARRANTX", "c": 0.10, "v": 500_000}]
        with patch.object(tasks, "os") as mock_os, \
                patch.object(tasks, "_fetch_json") as mock_fetch:
            mock_os.getenv.side_effect = lambda key, default=None: {
                "POLYGON_API_KEY": "test-key",
            }.get(key, default)
            mock_fetch.side_effect = [
                _grouped_daily_payload(rows),
                _ticker_details_payload(type="WARRANT"),
            ]
            result = tasks._fetch_polygon_penny_screener(
                min_price=0.01, max_price=1.0, min_volume=100_000, max_market_cap=50_000_000, max_symbols=10,
            )

        self.assertEqual(result, [])

    def test_excludes_market_cap_above_threshold(self):
        rows = [{"T": "BIGCAP", "c": 0.90, "v": 500_000}]
        with patch.object(tasks, "os") as mock_os, \
                patch.object(tasks, "_fetch_json") as mock_fetch:
            mock_os.getenv.side_effect = lambda key, default=None: {
                "POLYGON_API_KEY": "test-key",
            }.get(key, default)
            mock_fetch.side_effect = [
                _grouped_daily_payload(rows),
                _ticker_details_payload(market_cap=500_000_000.0),
            ]
            result = tasks._fetch_polygon_penny_screener(
                min_price=0.01, max_price=1.0, min_volume=100_000, max_market_cap=50_000_000, max_symbols=10,
            )

        self.assertEqual(result, [])

    def test_missing_api_key_returns_empty_without_network_call(self):
        with patch.object(tasks, "os") as mock_os, patch.object(tasks, "_fetch_json") as mock_fetch:
            mock_os.getenv.return_value = None
            result = tasks._fetch_polygon_penny_screener(
                min_price=0.01, max_price=1.0, min_volume=100_000, max_market_cap=50_000_000, max_symbols=10,
            )

        mock_fetch.assert_not_called()
        self.assertEqual(result, [])

    def test_no_grouped_data_for_a_week_returns_empty(self):
        # Ex. longue coupure de service Polygon -- dégrade proprement vers
        # les paliers de repli Yahoo/yfinance existants, jamais d'exception.
        with patch.object(tasks, "os") as mock_os, patch.object(tasks, "_fetch_json", return_value={}):
            mock_os.getenv.side_effect = lambda key, default=None: {
                "POLYGON_API_KEY": "test-key",
            }.get(key, default)
            result = tasks._fetch_polygon_penny_screener(
                min_price=0.01, max_price=1.0, min_volume=100_000, max_market_cap=50_000_000, max_symbols=10,
            )

        self.assertEqual(result, [])

    def test_stops_at_max_symbols_even_with_more_candidates(self):
        rows = [{"T": f"SYM{i}", "c": 0.50, "v": 1_000_000 - i} for i in range(5)]
        with patch.object(tasks, "os") as mock_os, \
                patch.object(tasks, "_fetch_json") as mock_fetch:
            mock_os.getenv.side_effect = lambda key, default=None: {
                "POLYGON_API_KEY": "test-key",
            }.get(key, default)
            mock_fetch.side_effect = [_grouped_daily_payload(rows)] + [_ticker_details_payload()] * 5
            result = tasks._fetch_polygon_penny_screener(
                min_price=0.01, max_price=1.0, min_volume=100_000, max_market_cap=50_000_000, max_symbols=2,
            )

        self.assertEqual(len(result), 2)
        # Trié par volume décroissant -- les deux plus liquides d'abord.
        self.assertEqual([r["symbol"] for r in result], ["SYM0", "SYM1"])
