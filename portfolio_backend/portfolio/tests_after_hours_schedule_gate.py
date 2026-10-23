"""Régression : cleanup-taskrunlog-daily, cleanup-system-logs-weekly,
weekend-deep-research-sat (depuis le 2026-08-20) ET (depuis le 2026-10)
model-evaluation-daily, model-drift-check-daily, continuous-evaluation-
daily doivent rester planifiées dans CELERY_BEAT_SCHEDULE même quand
AFTER_HOURS_TASKS_ENABLED est désactivé (défaut sur le NAS -- jamais
défini dans deploy/.env, donc toujours 'false'). Le reste du lot
(retraining/rollback de modèles, rapports) doit rester exclu, comme
avant -- toutes les tâches sorties du bloc sont en LECTURE SEULE
(ménage de logs, scan/score, ou mesure pure sans écriture dans
ModelRegistry/.pkl ni déclenchement d'une autre tâche -- voir
docs/TECH_DEBT_NOTES.md item 27 pour l'audit des 3 tâches de mesure),
sans rapport avec la décision explicite de ne pas activer la
gouvernance ML (retrain/rollback/promotion) en bloc (item 12,
TECH_DEBT_NOTES.md).
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

    def test_measurement_only_ml_tasks_always_scheduled(self):
        """Carve-out du 2026-10 (volet B) -- les 3 tâches auditées
        (compute_model_evaluation_daily, detect_model_drift,
        compute_continuous_evaluation_daily) n'écrivent que dans leurs
        tables de mesure, jamais dans ModelRegistry ni un .pkl, et ne
        déclenchent aucune autre tâche. Voir docs/TECH_DEBT_NOTES.md."""
        self.assertIn('model-evaluation-daily', settings.CELERY_BEAT_SCHEDULE)
        self.assertIn('model-drift-check-daily', settings.CELERY_BEAT_SCHEDULE)
        self.assertIn('continuous-evaluation-daily', settings.CELERY_BEAT_SCHEDULE)

    def test_rest_of_after_hours_batch_still_excluded(self):
        """Gouvernance ML réelle (retrain/rollback/promotion) -- reste
        désactivée par défaut, décision explicite non révisée par le
        carve-out des 3 tâches de mesure ci-dessus."""
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
            'data-pipeline-daily',
            'data-qa-daily',
            'backtest-retrain-guard-daily',
            'drift-retrain-daily',
            'model-rollback-daily',
        ]
        for key in still_excluded:
            self.assertNotIn(key, settings.CELERY_BEAT_SCHEDULE, key)
