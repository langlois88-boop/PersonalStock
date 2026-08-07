"""Regression test for bluechip_dip_scanner's funnel accounting (Volet 3):
every symbol pulled from the screener must land in exactly one bucket, so
funnel['evaluated'] == funnel['passed'] + sum(the other funnel counters).
"""

from __future__ import annotations

from unittest.mock import patch

from django.test import TestCase

from portfolio import tasks


class BluechipDipScannerFunnelTests(TestCase):
    def test_funnel_accounts_for_every_evaluated_symbol(self):
        quotes = [
            {'symbol': 'NOPRICE', 'regularMarketPrice': None},
            {'symbol': 'TOOCHEAP', 'regularMarketPrice': 1.0},
            {'symbol': 'NORSIDIV', 'regularMarketPrice': 50.0},
            {'symbol': 'LOWSCORE', 'regularMarketPrice': 50.0},
            {'symbol': 'UNHEALTHY', 'regularMarketPrice': 50.0},
            {'symbol': 'WINNER', 'regularMarketPrice': 50.0},
        ]

        def fake_rsi_divergence(ctx):
            return ctx.get('symbol') != 'NORSIDIV'

        def fake_mean_reversion_score(symbol):
            scores = {'LOWSCORE': 0.1, 'UNHEALTHY': 0.9, 'WINNER': 0.9}
            return scores.get(symbol, 0.9), 20.0, 45.0

        def fake_check_dip_health(symbol, price, score, min_score):
            return (symbol != 'UNHEALTHY'), {}

        with patch.object(tasks, '_fetch_yfinance_screeners', return_value=quotes), \
             patch.object(tasks, '_intraday_context_for_timeframe', lambda symbol, **k: {'symbol': symbol}), \
             patch.object(tasks, '_rsi_divergence', fake_rsi_divergence), \
             patch.object(tasks, '_mean_reversion_score', fake_mean_reversion_score), \
             patch.object(tasks, '_canadian_boost', lambda score, symbol: score), \
             patch('portfolio.ml_engine.pipeline_fixes.check_dip_health', fake_check_dip_health), \
             patch.object(tasks, '_system_log', lambda *a, **k: None):
            result = tasks.bluechip_dip_scanner()

        funnel = result['funnel']
        accounted = sum(v for k, v in funnel.items() if k not in {'evaluated', 'passed'})
        self.assertEqual(funnel['evaluated'], funnel['passed'] + accounted)

        # And confirm each synthetic symbol landed exactly where designed,
        # not just that the totals happen to balance.
        self.assertEqual(funnel['failed_no_symbol_or_price'], 1)  # NOPRICE
        self.assertEqual(funnel['failed_price_range'], 1)  # TOOCHEAP
        self.assertEqual(funnel['failed_rsi_divergence'], 1)  # NORSIDIV
        self.assertEqual(funnel['failed_dip_threshold'], 1)  # LOWSCORE
        self.assertEqual(funnel['failed_health_check'], 1)  # UNHEALTHY
        self.assertEqual(funnel['passed'], 1)  # WINNER
