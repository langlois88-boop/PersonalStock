"""Régression : cleanup_task_run_logs élague désormais aussi les succès
rapides et routiniers (durée < seuil), en gardant les N plus récents par
tâche -- pour que TaskRunLog reste navigable sans que le dashboard de
statut système (portfolio/views.py, `.order_by('-started_at').first()`
par task_name) ne perde la visibilité "dernière exécution" pour les
tâches qui sont presque toujours rapides (ex. fetch_finnhub_news_daily,
avg 5.6ms sur 182 exécutions réelles -- vérifié avant d'implémenter).

Le chemin chaud (_task_log_start/_task_log_finish, partagé par ~50
tâches) n'est PAS touché -- tout se passe dans le job de nettoyage
quotidien déjà existant.
"""

from __future__ import annotations

import os
from datetime import timedelta
from unittest.mock import patch

from django.test import TestCase
from django.utils import timezone

from portfolio import tasks
from portfolio.models import TaskRunLog


def _make_log(task_name: str, status: str, duration_ms, minutes_ago: int) -> int:
    log = TaskRunLog.objects.create(task_name=task_name, status=status, duration_ms=duration_ms)
    started_at = timezone.now() - timedelta(minutes=minutes_ago)
    TaskRunLog.objects.filter(id=log.id).update(started_at=started_at, finished_at=started_at)
    return log.id


class PruneFastSuccessTaskRunLogsTests(TestCase):
    def test_keeps_only_n_most_recent_fast_successes_per_task(self):
        ids = [
            _make_log('penny_sniper_alert', 'SUCCESS', 5, minutes_ago=m)
            for m in [50, 40, 30, 20, 10]
        ]
        with patch.dict(os.environ, {
            'TASKRUNLOG_FAST_SUCCESS_MS': '1000',
            'TASKRUNLOG_FAST_SUCCESS_KEEP_RECENT': '3',
        }):
            pruned = tasks._prune_fast_success_task_run_logs()

        self.assertEqual(pruned, 2)
        remaining_ids = set(
            TaskRunLog.objects.filter(task_name='penny_sniper_alert').values_list('id', flat=True)
        )
        # Les 3 plus récents (minutes_ago 30, 20, 10) doivent survivre.
        self.assertEqual(remaining_ids, set(ids[-3:]))

    def test_never_touches_failures_or_slow_successes(self):
        failed_id = _make_log('monitor_hive_trade', 'FAILED', 5, minutes_ago=100)
        slow_id = _make_log('fetch_prices_hourly', 'SUCCESS', 12000, minutes_ago=100)
        [_make_log('penny_sniper_alert', 'SUCCESS', 5, minutes_ago=m) for m in [50, 40, 30, 20, 10]]

        with patch.dict(os.environ, {
            'TASKRUNLOG_FAST_SUCCESS_MS': '1000',
            'TASKRUNLOG_FAST_SUCCESS_KEEP_RECENT': '3',
        }):
            tasks._prune_fast_success_task_run_logs()

        self.assertTrue(TaskRunLog.objects.filter(id=failed_id).exists())
        self.assertTrue(TaskRunLog.objects.filter(id=slow_id).exists())
        self.assertEqual(TaskRunLog.objects.filter(task_name='penny_sniper_alert').count(), 3)

    def test_disabled_via_keep_recent_zero(self):
        [_make_log('penny_sniper_alert', 'SUCCESS', 5, minutes_ago=m) for m in [50, 40, 30, 20, 10]]
        with patch.dict(os.environ, {'TASKRUNLOG_FAST_SUCCESS_KEEP_RECENT': '0'}):
            pruned = tasks._prune_fast_success_task_run_logs()
        self.assertEqual(pruned, 0)
        self.assertEqual(TaskRunLog.objects.filter(task_name='penny_sniper_alert').count(), 5)

    def test_disabled_via_threshold_zero(self):
        [_make_log('penny_sniper_alert', 'SUCCESS', 5, minutes_ago=m) for m in [50, 40, 30, 20, 10]]
        with patch.dict(os.environ, {'TASKRUNLOG_FAST_SUCCESS_MS': '0'}):
            pruned = tasks._prune_fast_success_task_run_logs()
        self.assertEqual(pruned, 0)
        self.assertEqual(TaskRunLog.objects.filter(task_name='penny_sniper_alert').count(), 5)

    def test_cleanup_task_run_logs_reports_pruned_count(self):
        [_make_log('penny_sniper_alert', 'SUCCESS', 5, minutes_ago=m) for m in [50, 40, 30, 20, 10]]
        with patch.dict(os.environ, {
            'TASKRUNLOG_FAST_SUCCESS_MS': '1000',
            'TASKRUNLOG_FAST_SUCCESS_KEEP_RECENT': '3',
        }):
            result = tasks.cleanup_task_run_logs()
        self.assertEqual(result['pruned_fast_successes'], 2)
