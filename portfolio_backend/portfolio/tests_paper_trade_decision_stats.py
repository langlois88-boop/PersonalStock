"""Regression test for the decision_stats accounting invariant in
_execute_paper_trades_for_sandbox: every continue/return inside the
watchlist loop must increment exactly one counter, so
watchlist == created + sum(all other decision_stats counters) always.

Kept in its own file (rather than tests.py) since it needs heavy mocking of
external calls (models, network, other services) to run deterministically.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import pandas as pd
from django.test import TestCase

from portfolio import tasks
from portfolio.models import PaperTrade, SandboxWatchlist


def _fake_yf_history(*args, **kwargs) -> pd.DataFrame:
    idx = pd.date_range('2024-01-01', periods=20, freq='D')
    return pd.DataFrame(
        {'Open': 10.0, 'High': 10.2, 'Low': 9.8, 'Close': 10.0}, index=idx,
    )


class _FakeTicker:
    """_latest_price/_atr are closures inside _execute_paper_trades_for_sandbox
    that call `yf.Ticker(symbol).history(...)` directly when use_alpaca is
    False (WATCHLIST) — can't patch the closures themselves, so this patches
    what they call instead."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def history(self, *args, **kwargs) -> pd.DataFrame:
        return _fake_yf_history()


def _fake_get_daily_bars(symbol, days=30) -> pd.DataFrame:
    idx = pd.date_range('2024-01-01', periods=days, freq='D')
    return pd.DataFrame(
        {'open': 10.0, 'high': 10.2, 'low': 9.8, 'close': 10.0}, index=idx,
    )


def _model_signal_side_effect(signal_by_symbol):
    def _side_effect(symbol, universe, model_path, use_alpaca=False):
        signal = signal_by_symbol.get(symbol)
        if signal is None:
            return None
        return {
            'signal': signal,
            'features': {},
            'explanations': [],
            'model_version': 'test-v1',
            'model_name': universe,
        }
    return _side_effect


def _assert_invariant(testcase, decision_stats):
    watchlist = decision_stats['watchlist']
    created = decision_stats['created']
    blocked_total = sum(
        v for k, v in decision_stats.items() if k.startswith('blocked_')
    )
    testcase.assertEqual(
        watchlist,
        created + blocked_total,
        f"watchlist ({watchlist}) != created ({created}) + sum(blocked_*) "
        f"({blocked_total}) — an exit path is going uncounted. decision_stats={decision_stats}",
    )
    testcase.assertEqual(decision_stats.get('unaccounted'), 0)


class PaperTradeDecisionStatsInvariantTests(TestCase):
    """One scenario per sandbox: a mix of a no-signal candidate, a
    below-threshold candidate, an existing-position candidate, and a
    candidate that clears every gate and gets created — then assert the
    accounting invariant on the resulting decision_stats."""

    def _run(self, sandbox, prefix, signal_by_symbol, reinforce_signal_symbol=None):
        SandboxWatchlist.objects.update_or_create(
            sandbox=sandbox, defaults={'symbols': list(signal_by_symbol.keys())},
        )
        if reinforce_signal_symbol:
            PaperTrade.objects.create(
                ticker=reinforce_signal_symbol,
                sandbox=sandbox,
                entry_price=10.0,
                quantity=1,
                stop_loss=1.0,
                status='OPEN',
                broker='SIM',
                pnl=0,
            )

        captured: dict = {}

        def _capture_system_log(category, level, message, symbol=None, metadata=None):
            if metadata is not None and 'watchlist' in metadata:
                captured['decision_stats'] = metadata

        def _tracking_signal(symbol):
            value = signal_by_symbol.get(symbol)
            return 0.0 if value is None else value

        simple_mocks = {
            'get_latest_trade_price': lambda symbol: 10.0,
            'get_daily_bars': _fake_get_daily_bars,
            'get_intraday_context': lambda *a, **k: None,
            '_intraday_context_for_timeframe': lambda *a, **k: None,
            'get_market_sentiment': lambda: ('NEUTRAL', {}),
            '_market_sentiment_score': lambda: 0.5,
            'get_market_regime_context': lambda: {'risk_off': False},
            '_daily_equity_circuit_breaker': lambda sandbox, capital: {'triggered': False},
            '_weak_list_health': lambda: {'defensive': False},
            '_atr_spike': lambda symbol, use_alpaca=False: False,
            '_btc_trend_ok': lambda symbol: True,
            '_earnings_blackout': lambda symbol, days=2: (False, None),
            '_daily_trend_ok': lambda symbol, use_alpaca=False: True,
            '_correlation_blocked': lambda symbol, open_symbols: False,
            '_spread_too_wide': lambda symbol, max_pct: False,
            '_loss_blacklist': lambda symbol, sandbox: False,
            '_sector_trend_ok': lambda symbol: True,
            '_reddit_hype_risk': lambda symbol: False,
            '_pump_dump_risk': lambda ctx, sentiment: False,
            '_penny_breakout_score': lambda bars: (False, 0.0),
            '_finbert_recent_sentiment': lambda symbol, days=2: (0.0, []),
            '_model_signal': _model_signal_side_effect(signal_by_symbol),
            '_system_log': _capture_system_log,
            '_mean_reversion_score': lambda symbol: (
                _tracking_signal(symbol),
                None if signal_by_symbol.get(symbol) is None else 50.0,
                9.0,
            ),
            '_bluechip_fundamental_score': lambda symbol: (
                min(1.0, max(0.0, _tracking_signal(symbol))), {},
            ),
            '_index_correlation_score': lambda symbol, days=90: min(1.0, max(0.0, _tracking_signal(symbol))),
        }

        with ExitStack() as stack:
            for name, value in simple_mocks.items():
                stack.enter_context(patch.object(tasks, name, value))
            stack.enter_context(patch.object(tasks.yf, 'Ticker', _FakeTicker))
            stack.enter_context(patch(
                'portfolio.services.signal_engine_patches.should_trade_with_mtf',
                return_value=(True, {}),
            ))
            tasks._execute_paper_trades_for_sandbox(sandbox, prefix)

        self.assertIn('decision_stats', captured, "System log with decision_stats was never captured")
        return captured['decision_stats']

    def test_watchlist_sandbox_invariant(self):
        signal_by_symbol = {
            'NOSIGW': None,
            'LOWW': 0.01,
            'EXISTW': 0.70,
            'WINW': 0.99,
        }
        stats = self._run('WATCHLIST', 'PAPER', signal_by_symbol, reinforce_signal_symbol='EXISTW')
        _assert_invariant(self, stats)
        self.assertEqual(stats['watchlist'], 4)

    def test_ai_penny_sandbox_invariant(self):
        signal_by_symbol = {
            'NOSIGP': None,
            'LOWP': 0.01,
            'EXISTP': 0.50,
            'WINP': 0.99,
        }
        stats = self._run('AI_PENNY', 'AI_PENNY', signal_by_symbol, reinforce_signal_symbol='EXISTP')
        _assert_invariant(self, stats)
        self.assertEqual(stats['watchlist'], 4)

    def test_ai_bluechip_sandbox_invariant(self):
        # AI_BLUECHIP's composite signal is 0.2*base + 0.4*fundamentals + 0.4*corr;
        # _bluechip_fundamental_score/_index_correlation_score are mocked to
        # track the same per-symbol value as base_signal, so composite ends
        # up == that value (0.2 + 0.4 + 0.4 == a 1.0x multiplier).
        signal_by_symbol = {
            'LOWB': 0.01,
            'EXISTB': 0.82,
            'WINB': 0.99,
        }
        stats = self._run('AI_BLUECHIP', 'AI_BLUECHIP', signal_by_symbol, reinforce_signal_symbol='EXISTB')
        _assert_invariant(self, stats)
        self.assertEqual(stats['watchlist'], 3)
