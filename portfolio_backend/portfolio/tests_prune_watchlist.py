"""prune_watchlist_weekly (2026-08-23) : rien ne retirait jamais un titre
d'une SandboxWatchlist une fois ajouté -- seulement des mécanismes d'AJOUT
(market_scanner_task, penny_opportunity_scanner, etc.). Trouvé en creusant
pourquoi WATCHLIST contenait SONI.CN (ROE -231.6%, dette/équité 355% --
vérifié en direct le même jour dans un tout autre contexte, Factor
Investing dans quant_filter.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.test import TestCase

from portfolio import tasks
from portfolio.models import PaperTrade, SandboxWatchlist


class PruneWatchlistReasonsTests(TestCase):
    def test_flags_deeply_negative_roe(self):
        reasons = tasks._prune_watchlist_reasons({"returnOnEquity": -2.3159})  # -231.59%
        self.assertEqual(len(reasons), 1)
        self.assertIn("ROE", reasons[0])

    def test_flags_extreme_leverage(self):
        # debtToEquity déjà en points de pourcentage (vérifié en direct
        # 2026-08-23) -- 354.97 veut dire 354.97%, pas 35497%.
        reasons = tasks._prune_watchlist_reasons({"debtToEquity": 354.97})
        self.assertEqual(len(reasons), 1)
        self.assertIn("Dette", reasons[0])

    def test_soni_cn_real_numbers_trigger_both_reasons(self):
        # Régression directe : les vrais chiffres SONI.CN trouvés en
        # direct ce jour-là (ROE -231.6%, dette/équité 355%).
        reasons = tasks._prune_watchlist_reasons({"returnOnEquity": -2.316, "debtToEquity": 354.97})
        self.assertEqual(len(reasons), 2)

    def test_healthy_ratios_trigger_nothing(self):
        reasons = tasks._prune_watchlist_reasons({"returnOnEquity": 0.20, "debtToEquity": 50.0})
        self.assertEqual(reasons, [])

    def test_missing_fields_trigger_nothing(self):
        reasons = tasks._prune_watchlist_reasons({})
        self.assertEqual(reasons, [])


class PruneWatchlistWeeklyTests(TestCase):
    def setUp(self):
        self.watch = SandboxWatchlist.objects.create(
            sandbox="WATCHLIST", symbols=["GOOD", "BADCO", "PROTECTED"], source="manual_reseed_test",
        )

    def _mock_ticker(self, info_by_symbol):
        def factory(symbol):
            return MagicMock(info=info_by_symbol.get(symbol, {}))
        return factory

    def test_removes_bad_ticker_without_open_position(self):
        info_by_symbol = {
            "GOOD": {"returnOnEquity": 0.15, "debtToEquity": 40.0},
            "BADCO": {"returnOnEquity": -2.5, "debtToEquity": 400.0},
            "PROTECTED": {"returnOnEquity": 0.10, "debtToEquity": 30.0},
        }
        with patch.object(tasks, "yf") as mock_yf:
            mock_yf.Ticker.side_effect = self._mock_ticker(info_by_symbol)
            result = tasks.prune_watchlist_weekly("WATCHLIST")

        self.watch.refresh_from_db()
        self.assertNotIn("BADCO", self.watch.symbols)
        self.assertIn("GOOD", self.watch.symbols)
        self.assertIn("PROTECTED", self.watch.symbols)
        self.assertEqual([r["symbol"] for r in result["removed"]], ["BADCO"])

    def test_never_removes_a_ticker_with_an_open_position(self):
        PaperTrade.objects.create(
            ticker="PROTECTED", sandbox="WATCHLIST", entry_price=10.0, quantity=5,
            stop_loss=9.0, status="OPEN",
        )
        info_by_symbol = {
            "GOOD": {"returnOnEquity": 0.15, "debtToEquity": 40.0},
            "BADCO": {"returnOnEquity": -2.5, "debtToEquity": 400.0},
            "PROTECTED": {"returnOnEquity": -2.5, "debtToEquity": 400.0},  # tout aussi mauvais que BADCO
        }
        with patch.object(tasks, "yf") as mock_yf:
            mock_yf.Ticker.side_effect = self._mock_ticker(info_by_symbol)
            result = tasks.prune_watchlist_weekly("WATCHLIST")

        self.watch.refresh_from_db()
        self.assertNotIn("BADCO", self.watch.symbols)
        self.assertIn("PROTECTED", self.watch.symbols)  # signalé mais gardé -- position ouverte
        self.assertEqual([r["symbol"] for r in result["kept_flagged"]], ["PROTECTED"])

    def test_network_failure_keeps_the_ticker_by_default(self):
        with patch.object(tasks, "yf") as mock_yf:
            mock_yf.Ticker.side_effect = Exception("boom")
            result = tasks.prune_watchlist_weekly("WATCHLIST")

        self.watch.refresh_from_db()
        self.assertEqual(set(self.watch.symbols), {"GOOD", "BADCO", "PROTECTED"})
        self.assertEqual(result["removed"], [])

    def test_no_watchlist_returns_no_data(self):
        result = tasks.prune_watchlist_weekly("AI_CRYPTO")
        self.assertEqual(result["status"], "no_data")

    def test_nothing_flagged_never_writes_to_the_watchlist(self):
        info_by_symbol = {
            "GOOD": {"returnOnEquity": 0.15, "debtToEquity": 40.0},
            "BADCO": {"returnOnEquity": 0.12, "debtToEquity": 35.0},
            "PROTECTED": {"returnOnEquity": 0.10, "debtToEquity": 30.0},
        }
        original_source = self.watch.source
        with patch.object(tasks, "yf") as mock_yf:
            mock_yf.Ticker.side_effect = self._mock_ticker(info_by_symbol)
            result = tasks.prune_watchlist_weekly("WATCHLIST")

        self.watch.refresh_from_db()
        self.assertEqual(result["removed"], [])
        self.assertEqual(self.watch.source, original_source)  # jamais touché si rien n'est retiré
