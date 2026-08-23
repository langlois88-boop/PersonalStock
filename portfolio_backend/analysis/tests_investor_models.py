"""Modèles "style d'investisseur" (2026-08-23) : révision de bénéfices
(méthode Thorp), Trend Template (Minervini), et CAN SLIM (O'Neil, partie
quantifiable). Même patron que QuantFilterValueCatalystTests dans
tests.py -- mock market_data + yf.Ticker, jamais de vrai appel réseau."""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
from django.core.cache import cache
from django.test import TestCase

from analysis.services import quant_filter


def _mock_info(**overrides):
    base = {
        "regularMarketPrice": 50.0,
        "sector": "Technology",
        "trailingPE": 20.0,
        "returnOnEquity": 0.18,
        "totalDebt": 100.0,
        "ebitda": 200.0,
        "numberOfAnalystOpinions": 20,
        "earningsQuarterlyGrowth": 0.30,
        "revenueGrowth": 0.25,
    }
    base.update(overrides)
    return base


def _flat_margin_trend():
    return pd.DataFrame(
        {"Q1": [1000.0, 350.0], "Q2": [1000.0, 350.0], "Q3": [1000.0, 350.0], "Q4": [1000.0, 350.0]},
        index=["Total Revenue", "Gross Profit"],
    )


def _earnings_revision_frames(*, num_analysts=20, up_0y=True, up_1y=True, up30=18, down30=3):
    est = pd.DataFrame(
        {"numberOfAnalysts": [10, 8, num_analysts, num_analysts + 2]},
        index=["0q", "+1q", "0y", "+1y"],
    )
    trend = pd.DataFrame(
        {
            "current": [1.0, 1.1, 10.0 if up_0y else 9.0, 12.0 if up_1y else 10.5],
            "30daysAgo": [1.0, 1.1, 9.5, 11.0],
        },
        index=["0q", "+1q", "0y", "+1y"],
    )
    revisions = pd.DataFrame(
        {"upLast30days": [1, 1, up30 // 2, up30 - up30 // 2], "downLast30days": [1, 1, down30 // 2, down30 - down30 // 2]},
        index=["0q", "+1q", "0y", "+1y"],
    )
    return est, trend, revisions


def _uptrend_history(days=310, start=30.0, end=50.0):
    """Série croissante lisse -- prix > MA50 > MA150 > MA200, MA200 en
    hausse sur le dernier mois, proche du sommet 52 semaines : satisfait
    les 8 critères du Trend Template par construction."""
    import numpy as np
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="B")
    n = len(dates)  # freq="B" peut ajuster le compte réel selon la date de fin
    closes = np.linspace(start, end, n)
    volume = np.full(n, 500_000)
    return pd.DataFrame({"Close": closes, "Volume": volume}, index=dates)


def _flat_benchmark_history(days=150, value=4000.0):
    """Marché plat -- pas de jour de distribution (aucune variation, donc
    'market_not_under_distribution' est vrai par construction)."""
    import numpy as np
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="B")
    n = len(dates)
    return pd.DataFrame({"Close": np.full(n, value), "Volume": np.full(n, 3_000_000_000)}, index=dates)


class _ClearsCacheTestCase(TestCase):
    """Le cache (Redis, partagé entre processus -- voir settings.py CACHES)
    n'est PAS réinitialisé automatiquement entre les tests par Django (ça
    ne concerne que la base de données). Sans ce setUp, un test peut lire
    une valeur mise en cache par une exécution précédente de la suite au
    lieu d'appeler le yf.Ticker mocké -- trouvé en écrivant
    MarketDistributionDaysTests (le deuxième test lisait le résultat
    caché du premier, résultat toujours 0 peu importe le mock)."""

    def setUp(self):
        cache.clear()


class ThorpEarningsRevisionTests(_ClearsCacheTestCase):
    def test_passes_when_both_years_revised_up_with_strong_ratio(self):
        est, trend, revisions = _earnings_revision_frames()
        mock_ticker = MagicMock(earnings_estimate=est, eps_trend=trend, eps_revisions=revisions)
        with patch.object(quant_filter, "market_data") as mock_md, \
                patch.object(quant_filter.yf, "Ticker", return_value=mock_ticker), \
                patch.object(quant_filter, "_fetch_margin_trend", return_value=[]):
            mock_md.Ticker.return_value.info = _mock_info()
            result = quant_filter.evaluate_ticker("NVDA", preset_thresholds={"require_earnings_revision": True})

        self.assertTrue(result.passed)
        self.assertTrue(result.details["earnings_revision"]["thorp_signal"])
        json.dumps(result.details)

    def test_fails_when_next_year_estimate_was_cut(self):
        est, trend, revisions = _earnings_revision_frames(up_1y=False)
        mock_ticker = MagicMock(earnings_estimate=est, eps_trend=trend, eps_revisions=revisions)
        with patch.object(quant_filter, "market_data") as mock_md, \
                patch.object(quant_filter.yf, "Ticker", return_value=mock_ticker), \
                patch.object(quant_filter, "_fetch_margin_trend", return_value=[]):
            mock_md.Ticker.return_value.info = _mock_info()
            result = quant_filter.evaluate_ticker("AAPL", preset_thresholds={"require_earnings_revision": True})

        self.assertFalse(result.passed)
        self.assertFalse(result.details["earnings_revision"]["thorp_signal"])

    def test_fails_when_down_revisions_dominate_despite_rising_average(self):
        # Régression directe du problème trouvé en testant sur du vrai
        # (MSFT) : moyenne en hausse mais plus de baisses que de hausses
        # -- le ratio doit rejeter ça, pas juste regarder la tendance nette.
        est, trend, revisions = _earnings_revision_frames(up30=4, down30=20)
        mock_ticker = MagicMock(earnings_estimate=est, eps_trend=trend, eps_revisions=revisions)
        with patch.object(quant_filter, "market_data") as mock_md, \
                patch.object(quant_filter.yf, "Ticker", return_value=mock_ticker), \
                patch.object(quant_filter, "_fetch_margin_trend", return_value=[]):
            mock_md.Ticker.return_value.info = _mock_info()
            result = quant_filter.evaluate_ticker("MSFT", preset_thresholds={"require_earnings_revision": True})

        self.assertFalse(result.passed)

    def test_default_preset_never_fetches_earnings_revision(self):
        with patch.object(quant_filter, "market_data") as mock_md, \
                patch.object(quant_filter.yf, "Ticker") as mock_yf, \
                patch.object(quant_filter, "_fetch_margin_trend", return_value=[]), \
                patch.object(quant_filter, "_fetch_earnings_revision") as mock_fetch:
            mock_md.Ticker.return_value.info = _mock_info()
            mock_yf.return_value.quarterly_financials = _flat_margin_trend()
            quant_filter.evaluate_ticker("AAPL", preset_thresholds={})
        mock_fetch.assert_not_called()


class MinerviniTrendTemplateTests(_ClearsCacheTestCase):
    def test_passes_on_clean_uptrend(self):
        uptrend = _uptrend_history()
        bench = _flat_benchmark_history()

        def ticker_side_effect(symbol):
            m = MagicMock()
            m.history.return_value = bench if symbol == "^GSPC" else uptrend
            m.quarterly_financials = _flat_margin_trend()
            return m

        with patch.object(quant_filter, "market_data") as mock_md, \
                patch.object(quant_filter.yf, "Ticker", side_effect=ticker_side_effect), \
                patch.object(quant_filter, "_fetch_margin_trend", return_value=[]):
            mock_md.Ticker.return_value.info = _mock_info(regularMarketPrice=50.0)
            result = quant_filter.evaluate_ticker("NVDA", preset_thresholds={"require_minervini_trend": True})

        self.assertTrue(result.passed, result.reasons_failed)
        self.assertEqual(result.details["minervini"]["criteria_met"], 8)

    def test_fails_when_price_below_moving_averages(self):
        import numpy as np
        dates = pd.date_range(end=pd.Timestamp.today(), periods=310, freq="B")
        n = len(dates)
        # Tendance baissière -- prix sous ses moyennes mobiles.
        closes = np.linspace(50.0, 30.0, n)
        downtrend = pd.DataFrame({"Close": closes, "Volume": np.full(n, 500_000)}, index=dates)
        bench = _flat_benchmark_history()

        def ticker_side_effect(symbol):
            m = MagicMock()
            m.history.return_value = bench if symbol == "^GSPC" else downtrend
            m.quarterly_financials = _flat_margin_trend()
            return m

        with patch.object(quant_filter, "market_data") as mock_md, \
                patch.object(quant_filter.yf, "Ticker", side_effect=ticker_side_effect), \
                patch.object(quant_filter, "_fetch_margin_trend", return_value=[]):
            mock_md.Ticker.return_value.info = _mock_info(regularMarketPrice=30.0)
            result = quant_filter.evaluate_ticker("XYZ", preset_thresholds={"require_minervini_trend": True})

        self.assertFalse(result.passed)
        self.assertLess(result.details["minervini"]["criteria_met"], 8)


class MarketDistributionDaysTests(_ClearsCacheTestCase):
    def test_flat_market_has_no_distribution_days(self):
        bench = _flat_benchmark_history(days=60)
        with patch.object(quant_filter.yf, "Ticker", return_value=MagicMock(history=MagicMock(return_value=bench))):
            result = quant_filter._fetch_market_distribution_days()
        self.assertTrue(result["available"])
        self.assertEqual(result["distribution_days"], 0)
        self.assertFalse(result["under_distribution"])

    def test_repeated_down_days_on_rising_volume_trigger_the_warning(self):
        import numpy as np
        dates = pd.date_range(end=pd.Timestamp.today(), periods=60, freq="B")
        n = len(dates)
        # Alterne clôtures plates et baisses de 0,5% sur volume croissant --
        # au moins 4 vraies journées de distribution dans les 25 dernières séances.
        closes = [4000.0]
        for i in range(1, n):
            closes.append(closes[-1] * 0.995 if i % 3 == 0 else closes[-1])
        volume = np.linspace(2_000_000_000, 4_000_000_000, n)
        bench = pd.DataFrame({"Close": closes, "Volume": volume}, index=dates)

        with patch.object(quant_filter.yf, "Ticker", return_value=MagicMock(history=MagicMock(return_value=bench))):
            result = quant_filter._fetch_market_distribution_days()

        self.assertTrue(result["available"])
        self.assertGreaterEqual(result["distribution_days"], 4)
        self.assertTrue(result["under_distribution"])


class CanSlimTests(_ClearsCacheTestCase):
    def test_passes_with_strong_growth_and_uptrend(self):
        uptrend = _uptrend_history()
        bench = _flat_benchmark_history()

        def ticker_side_effect(symbol):
            m = MagicMock()
            m.history.return_value = bench if symbol == "^GSPC" else uptrend
            return m

        with patch.object(quant_filter, "market_data") as mock_md, \
                patch.object(quant_filter.yf, "Ticker", side_effect=ticker_side_effect), \
                patch.object(quant_filter, "_fetch_margin_trend", return_value=[]):
            mock_md.Ticker.return_value.info = _mock_info(
                regularMarketPrice=50.0, earningsQuarterlyGrowth=0.30, revenueGrowth=0.25,
            )
            result = quant_filter.evaluate_ticker("NVDA", preset_thresholds={"require_can_slim": True})

        self.assertTrue(result.passed, result.reasons_failed)
        self.assertTrue(result.details["can_slim"]["passes_screen"])
        json.dumps(result.details)

    def test_fails_when_growth_too_weak(self):
        uptrend = _uptrend_history()
        bench = _flat_benchmark_history()

        def ticker_side_effect(symbol):
            m = MagicMock()
            m.history.return_value = bench if symbol == "^GSPC" else uptrend
            return m

        with patch.object(quant_filter, "market_data") as mock_md, \
                patch.object(quant_filter.yf, "Ticker", side_effect=ticker_side_effect), \
                patch.object(quant_filter, "_fetch_margin_trend", return_value=[]):
            # Croissance BPA/ventes bien sous le seuil de 20% de CAN SLIM.
            mock_md.Ticker.return_value.info = _mock_info(
                regularMarketPrice=50.0, earningsQuarterlyGrowth=0.05, revenueGrowth=0.03,
            )
            result = quant_filter.evaluate_ticker("SLOW", preset_thresholds={"require_can_slim": True})

        self.assertFalse(result.passed)
        self.assertFalse(result.details["can_slim"]["passes_screen"])
