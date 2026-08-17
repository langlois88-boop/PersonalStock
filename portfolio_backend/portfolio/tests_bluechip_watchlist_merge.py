"""Regression tests for the AI_BLUECHIP watchlist merge (dip scanner vs.
hourly refresh no longer overwrite each other unconditionally)."""

from __future__ import annotations

import os
from datetime import timedelta
from unittest.mock import patch

import pandas as pd
from django.test import TestCase
from django.utils import timezone

from portfolio import tasks
from portfolio.models import SandboxWatchlist


class MergeBluechipWatchlistTests(TestCase):
    def test_protected_symbols_survive_an_unprotected_merge(self):
        # bluechip_dip_scanner writes 2 symbols with a 4h protection window.
        tasks._merge_bluechip_watchlist(
            ['DIPA', 'DIPB'], new_source='bluechip_dip_scanner', limit=25, protect_hours=4,
        )
        # refresh_ai_bluechip_watchlist runs 5 minutes later — grants its own
        # symbols no protection.
        result = tasks._merge_bluechip_watchlist(
            ['REFA', 'REFB', 'REFC'], new_source='ai/opportunities', limit=25, protect_hours=None,
        )
        watchlist = SandboxWatchlist.objects.get(sandbox='AI_BLUECHIP')
        self.assertIn('DIPA', watchlist.symbols)
        self.assertIn('DIPB', watchlist.symbols)
        self.assertIn('REFA', watchlist.symbols)
        self.assertEqual(result['kept'], 2)
        self.assertEqual(result['added'], 3)
        self.assertEqual(result['removed'], 0)
        self.assertEqual(watchlist.symbol_sources['DIPA']['source'], 'bluechip_dip_scanner')
        self.assertEqual(watchlist.symbol_sources['REFA']['source'], 'ai/opportunities')

    def test_expired_protection_lets_symbol_be_displaced(self):
        tasks._merge_bluechip_watchlist(
            ['DIPA'], new_source='bluechip_dip_scanner', limit=25, protect_hours=4,
        )
        # Simulate the protection window having expired (rewrite the
        # protect_until timestamp into the past, as if 5 hours had passed).
        watchlist = SandboxWatchlist.objects.get(sandbox='AI_BLUECHIP')
        watchlist.symbol_sources['DIPA']['protect_until'] = (
            timezone.now() - timedelta(hours=1)
        ).isoformat()
        watchlist.save(update_fields=['symbol_sources'])

        result = tasks._merge_bluechip_watchlist(
            ['REFA'], new_source='ai/opportunities', limit=25, protect_hours=None,
        )
        watchlist.refresh_from_db()
        self.assertNotIn('DIPA', watchlist.symbols)
        self.assertIn('REFA', watchlist.symbols)
        self.assertEqual(result['kept'], 0)
        self.assertEqual(result['removed'], 1)
        self.assertIn('DIPA', result['removed_symbols'])

    def test_limit_is_respected_with_protected_symbols_prioritized(self):
        tasks._merge_bluechip_watchlist(
            ['DIPA', 'DIPB', 'DIPC'], new_source='bluechip_dip_scanner', limit=25, protect_hours=4,
        )
        result = tasks._merge_bluechip_watchlist(
            ['REFA', 'REFB', 'REFC', 'REFD'], new_source='ai/opportunities', limit=4, protect_hours=None,
        )
        watchlist = SandboxWatchlist.objects.get(sandbox='AI_BLUECHIP')
        self.assertEqual(len(watchlist.symbols), 4)
        # All 3 protected dip symbols survive; only 1 of 4 new ones fits.
        for sym in ('DIPA', 'DIPB', 'DIPC'):
            self.assertIn(sym, watchlist.symbols)
        self.assertEqual(result['kept'], 3)
        self.assertEqual(result['added'], 1)

    def test_dip_scanner_and_refresh_end_to_end_via_real_tasks(self):
        """Runs both actual @shared_task functions (heavily mocked
        externally) in dip-scanner-then-refresh order and confirms the
        merge — not just the helper in isolation."""
        quotes = [{'symbol': 'DIPWIN', 'regularMarketPrice': 50.0}]

        with patch.object(tasks, '_fetch_yfinance_screeners', return_value=quotes), \
             patch.object(tasks, '_intraday_context_for_timeframe', lambda symbol, **k: {'symbol': symbol}), \
             patch.object(tasks, '_rsi_divergence', lambda ctx: True), \
             patch.object(tasks, '_mean_reversion_score', lambda symbol: (0.9, 20.0, 45.0)), \
             patch.object(tasks, '_canadian_boost', lambda score, symbol: score), \
             patch('portfolio.ml_engine.pipeline_fixes.check_dip_health', lambda *a, **k: (True, {})), \
             patch.object(tasks, '_system_log', lambda *a, **k: None):
            dip_result = tasks.bluechip_dip_scanner()
        self.assertEqual(dip_result['watchlist_merge']['added'], 1)

        class _FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {'results': []}  # /api/ai/opportunities/ found nothing this hour

        with patch.object(tasks.requests, 'get', return_value=_FakeResponse()):
            refresh_result = tasks.refresh_ai_bluechip_watchlist()

        # DIPWIN survives the refresh even though opportunities/ returned
        # zero results — this is exactly the bug from the audit.
        watchlist = SandboxWatchlist.objects.get(sandbox='AI_BLUECHIP')
        self.assertIn('DIPWIN', watchlist.symbols)
        self.assertEqual(refresh_result['watchlist_merge']['kept'], 1)
        self.assertEqual(refresh_result['watchlist_merge']['added'], 0)


class AfterhoursScanMergesIntoWatchlistTests(TestCase):
    """
    Confirmed live 2026-08-17: market-scanner-5min stays scheduled through
    all of hour 16 ET (hour='9-16' covers 16:00-16:59, not just up to the
    16:00 close), so market_scanner_task's _market_closed_now() branch into
    _afterhours_market_scan fires repeatedly after the close. That function
    used to do an unconditional SandboxWatchlist.objects.update_or_create
    on sandbox='WATCHLIST' -- a thin afterhours result (sometimes a single
    symbol) wiped out the entire ML-evaluated (Option A, base_model_signal)
    ticker list WATCHLIST otherwise runs on. First attempt reused
    _merge_bluechip_watchlist -- wrong fit (that function only keeps
    symbols carrying an explicit `protect_until`, so WATCHLIST's
    never-protected base was still getting evicted, just via a different
    path -- caught by this very test). Now uses a dedicated, simpler
    additive merge inline in _afterhours_market_scan: always keep the
    existing list, only fill remaining room (up to main_limit) with new
    afterhours picks not already present.
    """

    def _daily_history_with_breakout(self) -> pd.DataFrame:
        # 20 daily closes, flat, then a jump + volume spike on the last bar
        # -- clears SCANNER_MIN_PRICE/MAX_PRICE/MIN_CHANGE_PCT/MIN_VOLUME/
        # AFTERHOURS_DAILY_RVOL_MIN/SCANNER_MIN_CONFIDENCE at their defaults.
        closes = [5.0] * 19 + [6.0]  # +20% on the last bar
        volumes = [200000] * 19 + [900000]  # well above min_volume, big rvol
        idx = pd.date_range('2026-07-20', periods=20, freq='D')
        return pd.DataFrame({'Close': closes, 'Volume': volumes}, index=idx)

    def test_thin_afterhours_result_does_not_wipe_existing_watchlist(self):
        SandboxWatchlist.objects.update_or_create(
            sandbox='WATCHLIST',
            defaults={'symbols': [f'ML{i}' for i in range(10)]},  # this morning's Option A evaluation
        )

        with patch.object(tasks.yf, 'Ticker') as mock_ticker, \
                patch.dict(os.environ, {
                    # AI_SCANNER_UPDATE_WATCHLIST_MAIN's whole block (including
                    # the WATCHLIST merge) is nested inside `if update_watchlist
                    # and results:` -- AI_SCANNER_UPDATE_WATCHLIST must stay
                    # true (its own default) for the WATCHLIST branch to be
                    # reached at all. This test doesn't assert anything about
                    # AI_PENNY's own write in that same block.
                    'AI_SCANNER_UPDATE_WATCHLIST': 'true',
                    'AI_SCANNER_UPDATE_WATCHLIST_MAIN': 'true',
                    'AI_SCANNER_MAIN_LIMIT': '15',
                }):
            mock_ticker.return_value.history.return_value = self._daily_history_with_breakout()
            tasks._afterhours_market_scan(symbols=['NEWPICK'])

        watchlist = SandboxWatchlist.objects.get(sandbox='WATCHLIST')
        for i in range(10):
            self.assertIn(f'ML{i}', watchlist.symbols, f"ML{i} was wiped out by the afterhours merge")
        self.assertIn('NEWPICK', watchlist.symbols)
