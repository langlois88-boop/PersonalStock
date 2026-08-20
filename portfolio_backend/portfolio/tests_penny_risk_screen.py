"""Tests pour penny_risk_screen.py (2026-08-20) -- filtre de risque/
catalyseur pour pennystocks (going concern, dette convertible toxique,
8-K récent via SEC EDGAR ; cash runway et dilution ATM via yfinance)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
from django.core.cache import cache
from django.test import TestCase

from portfolio.services import penny_risk_screen as prs


class _FakeResponse:
    def __init__(self, json_data=None, text="", status_code=200):
        self._json = json_data
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._json


class TickerToCikTests(TestCase):
    def setUp(self):
        cache.clear()

    def test_resolves_known_ticker_and_caches_the_whole_map(self):
        fake_map = _FakeResponse(json_data={"0": {"cik_str": 1819516, "ticker": "UP", "title": "Wheels Up"}})
        with patch.object(prs.requests, "get", return_value=fake_map) as mock_get:
            cik1 = prs._ticker_to_cik("UP")
            cik2 = prs._ticker_to_cik("up")  # casse différente, doit résoudre pareil
        self.assertEqual(cik1, "0001819516")
        self.assertEqual(cik2, "0001819516")
        mock_get.assert_called_once()  # le 2e appel utilise le cache, pas un 2e fetch

    def test_unknown_ticker_returns_none_not_exception(self):
        fake_map = _FakeResponse(json_data={"0": {"cik_str": 1819516, "ticker": "UP", "title": "Wheels Up"}})
        with patch.object(prs.requests, "get", return_value=fake_map):
            result = prs._ticker_to_cik("SONI.CN")  # ticker canadien, jamais dans SEC
        self.assertIsNone(result)

    def test_network_failure_returns_none_not_exception(self):
        with patch.object(prs.requests, "get", side_effect=Exception("network down")):
            self.assertIsNone(prs._ticker_to_cik("UP"))


class CheckGoingConcernTests(TestCase):
    def setUp(self):
        cache.clear()

    def _mock_ticker_and_filing(self):
        cache.set(prs._SEC_TICKER_MAP_CACHE_KEY, {"AIFF": "0001234567"}, timeout=3600)
        cache.set("sec_recent_filing:0001234567:10-Q", {"accession": "0001234567-26-000099", "filing_date": "2026-08-01"}, timeout=3600)

    def test_going_concern_found_in_current_filing(self):
        self._mock_ticker_and_filing()
        search_hit = _FakeResponse(json_data={
            "hits": {"hits": [{"_source": {"adsh": "0001234567-26-000099"}}]},
        })
        with patch.object(prs.requests, "get", return_value=search_hit):
            result = prs.check_going_concern("AIFF")
        self.assertTrue(result["going_concern_risk"])
        self.assertEqual(result["filing_date"], "2026-08-01")

    def test_going_concern_absent_from_current_filing(self):
        self._mock_ticker_and_filing()
        search_hit = _FakeResponse(json_data={"hits": {"hits": []}})
        with patch.object(prs.requests, "get", return_value=search_hit):
            result = prs.check_going_concern("AIFF")
        self.assertFalse(result["going_concern_risk"])

    def test_stale_mention_in_an_old_filing_is_not_counted(self):
        """Une mention 'going concern' dans un VIEUX 10-Q (accession
        différente du plus récent) ne doit pas compter -- seul le dépôt le
        plus récent doit être vérifié, pas tout l'historique."""
        self._mock_ticker_and_filing()
        search_hit = _FakeResponse(json_data={
            "hits": {"hits": [{"_source": {"adsh": "0001234567-23-000011"}}]},  # vieux dépôt
        })
        with patch.object(prs.requests, "get", return_value=search_hit):
            result = prs.check_going_concern("AIFF")
        self.assertFalse(result["going_concern_risk"])

    def test_non_sec_ticker_returns_neutral_defaults(self):
        cache.set(prs._SEC_TICKER_MAP_CACHE_KEY, {}, timeout=3600)
        result = prs.check_going_concern("SONI.CN")
        self.assertIsNone(result["going_concern_risk"])


class CashRunwayTests(TestCase):
    def _fake_ticker_with_burn(self):
        idx_bs = ["Cash And Cash Equivalents"]
        idx_cf = ["Operating Cash Flow", "Capital Expenditure"]
        bs = pd.DataFrame({"q1": [2_000_000.0]}, index=idx_bs)
        cf = pd.DataFrame({"q1": [-1_000_000.0, -0.0]}, index=idx_cf)
        fake = MagicMock()
        fake.quarterly_balance_sheet = bs
        fake.quarterly_cashflow = cf
        return fake

    def test_computes_runway_from_cash_and_burn(self):
        with patch.object(prs.yf, "Ticker", return_value=self._fake_ticker_with_burn()):
            result = prs.compute_cash_runway_quarters("AIFF")
        # 2M$ dispo / 1M$ de burn trimestriel = 2 trimestres de runway
        self.assertEqual(result["cash_runway_quarters"], 2.0)
        self.assertEqual(result["cash_available"], 2_000_000.0)

    def test_positive_operating_cash_flow_means_no_runway_concern(self):
        idx_bs = ["Cash And Cash Equivalents"]
        idx_cf = ["Operating Cash Flow", "Capital Expenditure"]
        bs = pd.DataFrame({"q1": [2_000_000.0]}, index=idx_bs)
        cf = pd.DataFrame({"q1": [500_000.0, -0.0]}, index=idx_cf)  # cash-flow positif
        fake = MagicMock(quarterly_balance_sheet=bs, quarterly_cashflow=cf)
        with patch.object(prs.yf, "Ticker", return_value=fake):
            result = prs.compute_cash_runway_quarters("PROFITABLE")
        self.assertIsNone(result["cash_runway_quarters"])

    def test_missing_data_returns_neutral_defaults_not_exception(self):
        fake = MagicMock(quarterly_balance_sheet=pd.DataFrame(), quarterly_cashflow=pd.DataFrame())
        with patch.object(prs.yf, "Ticker", return_value=fake):
            result = prs.compute_cash_runway_quarters("EMPTY")
        self.assertIsNone(result["cash_runway_quarters"])


class AtmDilutionTests(TestCase):
    def test_detects_share_count_growth(self):
        row = pd.DataFrame({
            "q1": [11_000_000.0], "q2": [10_500_000.0], "q3": [10_200_000.0], "q4": [10_000_000.0],
        }, index=["Ordinary Shares Number"])
        fake = MagicMock(quarterly_balance_sheet=row)
        with patch.object(prs.yf, "Ticker", return_value=fake):
            result = prs.detect_atm_dilution("DILUTED")
        self.assertEqual(result["shares_growth_4q_pct"], 10.0)  # 11M vs 10M il y a 4 trimestres

    def test_missing_data_returns_neutral_defaults(self):
        fake = MagicMock(quarterly_balance_sheet=pd.DataFrame())
        with patch.object(prs.yf, "Ticker", return_value=fake):
            result = prs.detect_atm_dilution("EMPTY")
        self.assertIsNone(result["shares_growth_4q_pct"])


class Recent8kCatalystTests(TestCase):
    def setUp(self):
        cache.clear()

    def test_recent_8k_within_window_is_flagged(self):
        cache.set(prs._SEC_TICKER_MAP_CACHE_KEY, {"AIFF": "0001234567"}, timeout=3600)
        import datetime
        recent_date = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=3)).strftime("%Y-%m-%d")
        cache.set(
            "sec_recent_filing:0001234567:8-K",
            {"accession": "0001234567-26-000123", "filing_date": recent_date},
            timeout=3600,
        )
        result = prs.check_recent_8k_catalyst("AIFF", days=14)
        self.assertTrue(result["recent_8k"])
        self.assertLessEqual(result["days_ago"], 4)

    def test_old_8k_outside_window_is_not_flagged(self):
        cache.set(prs._SEC_TICKER_MAP_CACHE_KEY, {"AIFF": "0001234567"}, timeout=3600)
        cache.set(
            "sec_recent_filing:0001234567:8-K",
            {"accession": "0001234567-24-000010", "filing_date": "2024-01-01"},
            timeout=3600,
        )
        result = prs.check_recent_8k_catalyst("AIFF", days=14)
        self.assertFalse(result["recent_8k"])


class CachedRedFlagsTests(TestCase):
    def setUp(self):
        cache.clear()

    def test_refresh_and_cache_then_read_back(self):
        with patch.object(prs, "check_going_concern", return_value={"going_concern_risk": False}), \
                patch.object(prs, "check_toxic_convertible_debt", return_value={"toxic_convertible_debt": False}), \
                patch.object(prs, "check_recent_8k_catalyst", return_value={"recent_8k": True}), \
                patch.object(prs, "compute_cash_runway_quarters", return_value={"cash_runway_quarters": 3.0}), \
                patch.object(prs, "detect_atm_dilution", return_value={"shares_growth_4q_pct": 5.0}):
            prs.refresh_and_cache_red_flags("AIFF")

        cached = prs.get_cached_red_flags("AIFF")
        self.assertIsNotNone(cached)
        self.assertEqual(cached["cash_runway"]["cash_runway_quarters"], 3.0)

    def test_uncached_ticker_returns_none(self):
        self.assertIsNone(prs.get_cached_red_flags("NEVERFETCHED"))
