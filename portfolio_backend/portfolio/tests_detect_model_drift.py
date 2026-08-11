"""Régression : detect_model_drift était accidentellement imbriquée à
l'intérieur de _alpaca_position_avg_price (indentation erronée), qui
n'est appelée nulle part dans ce code -- le décorateur @shared_task ne
s'exécutait donc jamais et Celery n'a jamais enregistré
'portfolio.tasks.detect_model_drift', indépendamment du flag
AFTER_HOURS_TASKS_ENABLED qui la désactive par ailleurs (elle n'aurait
échoué qu'avec "Received unregistered task" si jamais dispatchée).
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

from django.test import TestCase
from django.utils import timezone

from portfolio import tasks
from portfolio.models import ModelRegistry, PaperTrade


class DetectModelDriftIsModuleLevelTests(TestCase):
    def test_is_a_module_level_attribute_not_a_local_closure(self):
        # Avant le fix : hasattr renvoyait False, la fonction n'existait
        # que comme variable locale à l'intérieur de
        # _alpaca_position_avg_price, jamais exécutée.
        self.assertTrue(hasattr(tasks, 'detect_model_drift'))

    def test_registered_in_the_celery_app_task_registry(self):
        from portfolio_backend.celery import app
        self.assertIn('portfolio.tasks.detect_model_drift', app.tasks)


class DetectModelDriftLogicTests(TestCase):
    def _make_closed_trade(self, sandbox, pnl, outcome=None):
        PaperTrade.objects.create(
            ticker='TEST', sandbox=sandbox, entry_price=10, quantity=1,
            stop_loss=9, status='CLOSED', pnl=pnl, outcome=outcome,
            exit_date=timezone.now(),
        )

    @patch.object(tasks, '_send_telegram_alert')
    def test_no_drift_when_paper_win_rate_matches_baseline(self, mock_alert):
        ModelRegistry.objects.create(
            model_name='BLUECHIP', model_version='v1', status='ACTIVE', backtest_win_rate=50.0,
        )
        for _ in range(5):
            self._make_closed_trade('AI_BLUECHIP', pnl=1, outcome='WIN')
        for _ in range(5):
            self._make_closed_trade('AI_BLUECHIP', pnl=-1, outcome='LOSS')

        result = tasks.detect_model_drift()

        bluechip = next(r for r in result['results'] if r['sandbox'] == 'AI_BLUECHIP')
        self.assertEqual(bluechip['paper_win_rate'], 50.0)
        self.assertFalse(bluechip['drift'])
        mock_alert.assert_not_called()

    @patch.object(tasks, '_send_telegram_alert')
    def test_drift_detected_and_alerted_when_win_rate_drops(self, mock_alert):
        ModelRegistry.objects.create(
            model_name='PENNY', model_version='v1', status='ACTIVE', backtest_win_rate=60.0,
        )
        for _ in range(1):
            self._make_closed_trade('AI_PENNY', pnl=1, outcome='WIN')
        for _ in range(9):
            self._make_closed_trade('AI_PENNY', pnl=-1, outcome='LOSS')

        result = tasks.detect_model_drift()

        penny = next(r for r in result['results'] if r['sandbox'] == 'AI_PENNY')
        self.assertEqual(penny['paper_win_rate'], 10.0)
        self.assertTrue(penny['drift'])
        mock_alert.assert_called_once()
        self.assertIn('Drift détecté', mock_alert.call_args[0][0])

    @patch.object(tasks, '_send_telegram_alert')
    def test_no_baseline_means_no_drift(self, mock_alert):
        # Pas de ModelRegistry ACTIVE pour ce sandbox -> baseline=0, jamais
        # de drift (évite une fausse alerte au tout premier jour d'un modèle).
        self._make_closed_trade('AI_BLUECHIP', pnl=-1, outcome='LOSS')

        result = tasks.detect_model_drift()

        bluechip = next(r for r in result['results'] if r['sandbox'] == 'AI_BLUECHIP')
        self.assertEqual(bluechip['baseline_win_rate'], 0.0)
        self.assertFalse(bluechip['drift'])
        mock_alert.assert_not_called()

    @patch.object(tasks, '_send_telegram_alert')
    def test_old_trades_outside_lookback_window_are_excluded(self, mock_alert):
        ModelRegistry.objects.create(
            model_name='BLUECHIP', model_version='v1', status='ACTIVE', backtest_win_rate=50.0,
        )
        old_trade = PaperTrade.objects.create(
            ticker='TEST', sandbox='AI_BLUECHIP', entry_price=10, quantity=1,
            stop_loss=9, status='CLOSED', pnl=-1, outcome='LOSS',
        )
        PaperTrade.objects.filter(id=old_trade.id).update(
            exit_date=timezone.now() - timedelta(days=30),
        )

        result = tasks.detect_model_drift()

        bluechip = next(r for r in result['results'] if r['sandbox'] == 'AI_BLUECHIP')
        # Le trade existe mais est hors fenêtre de lookback -> exclu du
        # calcul (paper_win_rate retombe sur le défaut 0%, pas d'exception).
        # Comportement préexistant inchangé par ce fix (uniquement le
        # déplacement de la fonction) : 0% vs une baseline de 50% est
        # traité comme un drift par la formule elle-même.
        self.assertEqual(bluechip['paper_win_rate'], 0.0)
        self.assertTrue(bluechip['drift'])
