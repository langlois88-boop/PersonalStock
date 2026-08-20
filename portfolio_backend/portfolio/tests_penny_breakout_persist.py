"""Tests pour _persist_penny_breakout_signals (2026-08-20, tasks.py) --
écrit les candidats de _analyze_penny_breakouts (scan marché direct via
yfinance most_actives, indépendant des mentions Reddit/news) dans
PennySignal, la MÊME table que generate_penny_signals_v2, pour que la
chaîne existante (/api/penny-analytics/ -> refresh_ai_penny_watchlist)
les ramasse sans changement ailleurs."""

from __future__ import annotations

from django.test import TestCase
from django.utils import timezone

from portfolio import tasks
from portfolio.models import PennySignal


class PersistPennyBreakoutSignalsTests(TestCase):
    def test_writes_a_pennysignal_row_per_candidate(self):
        penny_result = {
            'watchlist': [
                {
                    'symbol': 'BOOM', 'last_close': 0.75, 'price_range_pct': 4.2,
                    'volume_ratio': 2.1, 'sentiment': 0.3, 'score': 0.82,
                    'catalysts': ['fda approval granted'],
                },
                {
                    'symbol': 'QUIET', 'last_close': 0.30, 'price_range_pct': 6.0,
                    'volume_ratio': 1.3, 'sentiment': 0.0, 'score': 0.41,
                    'catalysts': [],
                },
            ],
        }
        written = tasks._persist_penny_breakout_signals(penny_result)
        self.assertEqual(written, 2)

        boom = PennySignal.objects.get(symbol='BOOM', as_of=timezone.now().date())
        self.assertEqual(boom.combined_score, 0.82)
        self.assertEqual(boom.last_price, 0.75)
        self.assertEqual(boom.pattern_score, 1.0)  # a un catalyseur détecté
        self.assertEqual(boom.data['source'], 'market_breakout_scan')
        self.assertEqual(boom.data['catalysts'], ['fda approval granted'])

        quiet = PennySignal.objects.get(symbol='QUIET', as_of=timezone.now().date())
        self.assertEqual(quiet.pattern_score, 0.0)  # aucun catalyseur

    def test_score_is_clamped_to_0_1_even_if_sentiment_boost_overflows(self):
        """_analyze_penny_breakouts peut renvoyer un score > 1.0 (le
        multiplicateur de sentiment n'est pas borné dans son propre
        calcul) -- ne doit jamais planter l'écriture DB ni fausser un
        classement par score."""
        penny_result = {'watchlist': [{'symbol': 'OVER', 'last_close': 1.0, 'score': 1.4}]}
        tasks._persist_penny_breakout_signals(penny_result)
        row = PennySignal.objects.get(symbol='OVER')
        self.assertEqual(row.combined_score, 1.0)

    def test_empty_watchlist_writes_nothing(self):
        written = tasks._persist_penny_breakout_signals({'watchlist': []})
        self.assertEqual(written, 0)
        self.assertEqual(PennySignal.objects.count(), 0)

    def test_re_running_same_day_updates_not_duplicates(self):
        penny_result = {'watchlist': [{'symbol': 'BOOM', 'last_close': 0.75, 'score': 0.5}]}
        tasks._persist_penny_breakout_signals(penny_result)
        penny_result2 = {'watchlist': [{'symbol': 'BOOM', 'last_close': 0.80, 'score': 0.9}]}
        tasks._persist_penny_breakout_signals(penny_result2)

        self.assertEqual(PennySignal.objects.filter(symbol='BOOM').count(), 1)
        row = PennySignal.objects.get(symbol='BOOM')
        self.assertEqual(row.combined_score, 0.9)


class WeekendDeepResearchIntegrationTests(TestCase):
    def test_weekend_deep_research_persists_penny_signals(self):
        from unittest.mock import patch

        fake_penny = {'watchlist': [{'symbol': 'REAL', 'last_close': 0.5, 'score': 0.6}]}
        with patch.object(tasks, '_analyze_penny_breakouts', return_value=fake_penny), \
                patch.object(tasks, '_analyze_bluechip_rebounds', return_value={'watchlist': []}):
            result = tasks.weekend_deep_research()

        self.assertEqual(result['penny_signals_written'], 1)
        self.assertTrue(PennySignal.objects.filter(symbol='REAL').exists())
