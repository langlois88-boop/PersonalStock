"""Tests pour refresh_penny_red_flags_weekly (2026-08-20, tasks.py) --
la tâche Beat hebdomadaire qui peuple le cache de red flags pennystock
lu ensuite par _execute_paper_trades_for_sandbox (AI_PENNY)."""

from __future__ import annotations

from unittest.mock import patch

from django.test import TestCase

from portfolio import tasks
from portfolio.models import SandboxWatchlist, TaskRunLog
from portfolio.services import penny_risk_screen


class RefreshPennyRedFlagsWeeklyTests(TestCase):
    def test_processes_every_symbol_in_ai_penny_watchlist(self):
        SandboxWatchlist.objects.update_or_create(
            sandbox='AI_PENNY', defaults={'symbols': ['UP', 'HAIN']},
        )
        with patch.object(penny_risk_screen, 'refresh_and_cache_red_flags', return_value={
            'going_concern': {'going_concern_risk': False},
        }) as mock_refresh, patch.object(tasks, 'sleep'):
            result = tasks.refresh_penny_red_flags_weekly()

        self.assertEqual(result['symbols_processed'], 2)
        self.assertEqual(mock_refresh.call_count, 2)
        mock_refresh.assert_any_call('UP')
        mock_refresh.assert_any_call('HAIN')

    def test_counts_going_concern_flags(self):
        SandboxWatchlist.objects.update_or_create(
            sandbox='AI_PENNY', defaults={'symbols': ['RISKY', 'SAFE']},
        )

        def _fake_refresh(symbol):
            return {'going_concern': {'going_concern_risk': symbol == 'RISKY'}}

        with patch.object(penny_risk_screen, 'refresh_and_cache_red_flags', side_effect=_fake_refresh), \
                patch.object(tasks, 'sleep'):
            result = tasks.refresh_penny_red_flags_weekly()

        self.assertEqual(result['going_concern_flagged'], 1)

    def test_one_symbol_failing_does_not_abort_the_rest(self):
        SandboxWatchlist.objects.update_or_create(
            sandbox='AI_PENNY', defaults={'symbols': ['BROKEN', 'FINE']},
        )

        def _fake_refresh(symbol):
            if symbol == 'BROKEN':
                raise RuntimeError('SEC EDGAR down')
            return {'going_concern': {'going_concern_risk': False}}

        with patch.object(penny_risk_screen, 'refresh_and_cache_red_flags', side_effect=_fake_refresh), \
                patch.object(tasks, 'sleep'):
            result = tasks.refresh_penny_red_flags_weekly()

        self.assertEqual(result['symbols_processed'], 1)  # FINE seulement
        self.assertEqual(len(result['errors']), 1)
        self.assertIn('BROKEN', result['errors'][0])
        log = TaskRunLog.objects.filter(task_name='refresh_penny_red_flags_weekly').latest('started_at')
        self.assertEqual(log.status, 'SUCCESS')  # des erreurs partielles restent un SUCCESS de tâche

    def test_empty_watchlist_does_not_crash(self):
        with patch.object(tasks, 'sleep'):
            result = tasks.refresh_penny_red_flags_weekly()
        self.assertEqual(result['symbols_processed'], 0)


class RedFlagsInformAiPennyEntryFeaturesTests(TestCase):
    """_signal() (AI_PENNY) lit le cache -- jamais un appel réseau SEC
    depuis ce chemin. Test focalisé sur cette lecture seule, pas tout le
    harnais de _execute_paper_trades_for_sandbox déjà couvert ailleurs."""

    def test_get_cached_red_flags_used_by_signal_features(self):
        from django.core.cache import cache
        cache.set('penny_red_flags:UP', {
            'going_concern': {'going_concern_risk': True},
            'cash_runway': {'cash_runway_quarters': 1.5},
        }, timeout=3600)

        result = penny_risk_screen.get_cached_red_flags('UP')
        self.assertTrue(result['going_concern']['going_concern_risk'])
        self.assertEqual(result['cash_runway']['cash_runway_quarters'], 1.5)
