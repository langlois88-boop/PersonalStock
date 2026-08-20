"""Régression : cleanup-taskrunlog-daily, cleanup-system-logs-weekly ET
(depuis le 2026-08-20) weekend-deep-research-sat doivent rester
planifiés dans CELERY_BEAT_SCHEDULE même quand AFTER_HOURS_TASKS_ENABLED
est désactivé (défaut sur le NAS -- jamais défini dans deploy/.env, donc
toujours 'false'). Le reste du lot (retraining/rollback de modèles,
rapports) doit rester exclu, comme avant -- ces 3 tâches sont toutes en
LECTURE SEULE (ménage de logs ou scan/score sans commande d'achat/vente),
sans rapport avec la décision explicite de ne pas activer la gouvernance
ML en bloc (item 12, TECH_DEBT_NOTES.md).
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

    def test_weekend_deep_research_always_scheduled(self):
        """Carve-out du 2026-08-20 -- scan/score en lecture seule
        (_analyze_penny_breakouts/_analyze_bluechip_rebounds), trouvé en
        diagnostiquant pourquoi AI_PENNY ne découvre plus de candidats
        depuis des mois (generate_penny_signals dépend de credentials
        Reddit jamais configurées)."""
        self.assertIn('weekend-deep-research-sat', settings.CELERY_BEAT_SCHEDULE)

    def test_rest_of_after_hours_batch_still_excluded(self):
        still_excluded = [
            'deep-learning-retro-nightly',
            'nightly-closed-market-retrain',
            'nightly-intraday-retrain',
            'trading-journal-daily',
            'daily-bot-journal-2005',
            'sunday-evening-briefing',
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
