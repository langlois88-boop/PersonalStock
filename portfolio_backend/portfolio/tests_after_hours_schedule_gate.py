"""Régression : cleanup-taskrunlog-daily et cleanup-system-logs-weekly
doivent rester planifiés dans CELERY_BEAT_SCHEDULE même quand
AFTER_HOURS_TASKS_ENABLED est désactivé (défaut sur le NAS -- jamais
défini dans deploy/.env, donc toujours 'false'). Le reste du lot
(retraining/rollback de modèles, rapports) doit rester exclu, comme
avant -- ce changement ne touche que ces 2 tâches de ménage de logs.
"""

from __future__ import annotations

from django.conf import settings
from django.test import TestCase


class AfterHoursScheduleGateTests(TestCase):
    def test_log_cleanup_tasks_always_scheduled(self):
        # Ce test tourne dans l'environnement de test, où
        # AFTER_HOURS_TASKS_ENABLED n'est pas défini (donc False, comme sur
        # le NAS en production) -- les 2 tâches de ménage doivent quand même
        # apparaître.
        self.assertFalse(settings.AFTER_HOURS_TASKS_ENABLED)
        self.assertIn('cleanup-taskrunlog-daily', settings.CELERY_BEAT_SCHEDULE)
        self.assertIn('cleanup-system-logs-weekly', settings.CELERY_BEAT_SCHEDULE)

    def test_rest_of_after_hours_batch_still_excluded(self):
        still_excluded = [
            'deep-learning-retro-nightly',
            'nightly-closed-market-retrain',
            'nightly-intraday-retrain',
            'trading-journal-daily',
            'daily-bot-journal-2005',
            'sunday-evening-briefing',
            'weekend-deep-research-sat',
            'economic-calendar-weekly',
            'daily-performance-report',
            'daily-profit-tracker',
            'paper-trade-retrain-daily',
            'model-evaluation-daily',
            'model-drift-check-daily',
            'data-pipeline-daily',
            'data-qa-daily',
            'continuous-evaluation-daily',
            'backtest-retrain-guard-daily',
            'drift-retrain-daily',
            'model-rollback-daily',
        ]
        for key in still_excluded:
            self.assertNotIn(key, settings.CELERY_BEAT_SCHEDULE, key)
