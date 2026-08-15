"""Regression tests for item 17 (TECH_DEBT_NOTES.md): the confidence gates
in _execute_paper_trades_for_sandbox (SIM) and
_execute_alpaca_paper_trades_for_sandbox (Alpaca) used to be independent of
`buy_threshold` -- hardcoded 0.65 (SIM, non-AI_PENNY) / defaulted to 0.7
(Alpaca, all sandboxes). Harmless while buy_threshold stayed above those
values (AI_BLUECHIP/AI_PENNY's normal 0.80/0.82), but whenever buy_threshold
dropped below it -- WATCHLIST's own normal threshold (0.55) or the
exploration phase's override (0.55, docs/EXPLORATION_PHASE_2026-08.md) -- a
candidate could clear buy_threshold and still be silently rejected here,
with zero trades created despite the visible threshold saying it should
qualify. Confirmed live 2026-08-14: AI_BLUECHIP signal 0.6089 (> 0.55
buy_threshold) blocked by the old hardcoded 0.65 SIM floor.

Fix: cap each floor at buy_threshold (min(existing_default, buy_threshold)),
so the two gates can never disagree, in every mode.
"""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from django.test import TestCase

from portfolio import tasks
from portfolio.models import PaperTrade, SandboxWatchlist


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


class _FakeTicker:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def history(self, *args, **kwargs) -> pd.DataFrame:
        idx = pd.date_range('2024-01-01', periods=20, freq='D')
        return pd.DataFrame(
            {'Open': 10.0, 'High': 10.2, 'Low': 9.8, 'Close': 10.0}, index=idx,
        )


class _FakeDataFusionEngine:
    """DataFusionEngine(symbol, use_alpaca=True).fuse_all() -- an empty
    frame is handled gracefully by both consumers (_classify_tier falls
    back to T2, _atr_from_frame returns None), so this is enough."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def fuse_all(self) -> pd.DataFrame:
        return pd.DataFrame()


class ConfidenceFloorSimPathTests(TestCase):
    """_execute_paper_trades_for_sandbox: confidence_factor's floor for
    non-AI_PENNY sandboxes must never exceed the effective buy_threshold."""

    def _run(self, sandbox, prefix, signal_by_symbol, env=None):
        SandboxWatchlist.objects.update_or_create(
            sandbox=sandbox, defaults={'symbols': list(signal_by_symbol.keys())},
        )

        def _tracking_signal(symbol):
            value = signal_by_symbol.get(symbol)
            return 0.0 if value is None else value

        captured: dict = {}

        def _capture_system_log(category, level, message, symbol=None, metadata=None):
            if metadata is not None and 'watchlist' in metadata:
                captured['decision_stats'] = metadata

        simple_mocks = {
            '_system_log': _capture_system_log,
            'get_latest_trade_price': lambda symbol: 10.0,
            'get_daily_bars': _fake_get_daily_bars,
            'get_intraday_context': lambda *a, **k: None,
            '_intraday_context_for_timeframe': lambda *a, **k: None,
            'get_market_sentiment': lambda: ('NEUTRAL', {}),
            '_market_sentiment_score': lambda: 0.5,
            'get_market_regime_context': lambda: {'risk_off': False},
            '_daily_equity_circuit_breaker': lambda sandbox, capital, **kwargs: {'triggered': False},
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

        env = dict(env or {})
        with ExitStack() as stack:
            for name, value in simple_mocks.items():
                stack.enter_context(patch.object(tasks, name, value))
            stack.enter_context(patch.object(tasks.yf, 'Ticker', _FakeTicker))
            stack.enter_context(patch(
                'portfolio.services.signal_engine_patches.should_trade_with_mtf',
                return_value=(True, {}),
            ))
            stack.enter_context(patch.dict('os.environ', env))
            tasks._execute_paper_trades_for_sandbox(sandbox, prefix)
        return captured.get('decision_stats')

    def test_exploration_mode_unblocks_dead_zone_between_threshold_and_old_065_floor(self):
        """AI_BLUECHIP, exploration phase on (buy_threshold -> 0.55). A
        signal of 0.60 clears buy_threshold but sits below the old hardcoded
        0.65 confidence floor -- this is the exact scenario confirmed live
        on 2026-08-14 (signal 0.6089, created=0). Must now create a trade."""
        signal_by_symbol = {'DEADZONE': 0.60}
        stats = self._run(
            'AI_BLUECHIP', 'AI_BLUECHIP', signal_by_symbol,
            env={'EXPLORATION_PHASE_ENABLED': 'true', 'EXPLORATION_PHASE_BUY_THRESHOLD': '0.55'},
        )
        self.assertTrue(
            PaperTrade.objects.filter(sandbox='AI_BLUECHIP', ticker='DEADZONE', status='OPEN').exists(),
            "signal 0.60 cleared the 0.55 exploration buy_threshold but was still "
            f"blocked -- confidence floor is not tracking buy_threshold. decision_stats={stats}",
        )

    def test_normal_mode_bluechip_unaffected_signal_above_080_threshold(self):
        """AI_BLUECHIP, no exploration (buy_threshold stays 0.80). Old
        floor was 0.65 < 0.80, so it never bound in practice -- confirms
        min(0.65, 0.80) == 0.65 keeps this path working exactly as before."""
        signal_by_symbol = {'NORMALWIN': 0.90}
        stats = self._run('AI_BLUECHIP', 'AI_BLUECHIP', signal_by_symbol, env={'EXPLORATION_PHASE_ENABLED': 'false'})
        self.assertTrue(
            PaperTrade.objects.filter(sandbox='AI_BLUECHIP', ticker='NORMALWIN', status='OPEN').exists(),
            f"decision_stats={stats}",
        )

    def test_normal_mode_bluechip_still_blocks_below_threshold(self):
        """Sanity check: a signal below buy_threshold (0.80 normal) is still
        blocked by the threshold gate itself, unrelated to this fix."""
        signal_by_symbol = {'TOOLOW': 0.60}
        self._run('AI_BLUECHIP', 'AI_BLUECHIP', signal_by_symbol, env={'EXPLORATION_PHASE_ENABLED': 'false'})
        self.assertFalse(
            PaperTrade.objects.filter(sandbox='AI_BLUECHIP', ticker='TOOLOW').exists(),
        )


class ConfidenceFloorAlpacaPathTests(TestCase):
    """_execute_alpaca_paper_trades_for_sandbox: min_confidence must never
    exceed the effective buy_threshold. Uses ALPACA_SHADOW_TRADING=true
    (an existing test-friendly path already built into the function) to
    reach a real PaperTrade.objects.create call without touching the real
    Alpaca order-submission API."""

    def _run(self, sandbox, prefix, signal_by_symbol, env=None):
        simple_mocks = {
            '_get_watchlist': lambda sandbox, prefix, default: list(signal_by_symbol.keys()),
            '_midday_blackout': lambda: False,
            '_is_reentry_candidate': lambda symbol, sandbox: False,
            '_weak_list_health': lambda: {'defensive': False},
            '_earnings_blackout': lambda symbol, days=2: (False, None),
            '_daily_trend_ok': lambda symbol, use_alpaca=False: True,
            '_correlation_blocked': lambda symbol, open_symbols: False,
            '_symbol_currency': lambda symbol: 'USD',
            'get_market_sentiment': lambda: ('NEUTRAL', {}),
            '_market_sentiment_score': lambda: 0.5,
            'get_market_regime_context': lambda: {'risk_off': False},
            '_alpaca_buying_power': lambda: 50000.0,
            'get_account': lambda: SimpleNamespace(equity=50000.0),
            'get_open_positions': lambda: [],
            '_daily_equity_circuit_breaker': lambda sandbox, equity, **kwargs: {'triggered': False},
            '_model_signal': _model_signal_side_effect(signal_by_symbol),
            'get_intraday_context': lambda *a, **k: None,
            '_time_of_day_penalty': lambda *a, **k: 1.0,
            '_loss_blacklist': lambda symbol, sandbox: False,
            '_sector_trend_ok': lambda symbol: True,
            '_atr_spike': lambda symbol, use_alpaca=False: False,
            '_btc_trend_ok': lambda symbol: True,
            'get_latest_bid_ask_spread_pct': lambda symbol: 0.001,
            'get_order_book_imbalance': lambda symbol: None,
            'get_latest_trade_price': lambda symbol: 100.0,
            'DataFusionEngine': _FakeDataFusionEngine,
        }

        env = dict(env or {}, ALPACA_SHADOW_TRADING='true')
        with ExitStack() as stack:
            for name, value in simple_mocks.items():
                stack.enter_context(patch.object(tasks, name, value))
            stack.enter_context(patch.dict('os.environ', env))
            tasks._execute_alpaca_paper_trades_for_sandbox(sandbox, prefix)

    def test_exploration_mode_unblocks_dead_zone_between_threshold_and_old_07_floor(self):
        """AI_BLUECHIP, exploration phase on (buy_threshold -> 0.55). A
        signal of 0.60 clears buy_threshold but sits below the old default
        0.7 ALPACA_MIN_CONFIDENCE floor -- must now create a (shadow) trade."""
        signal_by_symbol = {'DEADZONEA': 0.60}
        self._run(
            'AI_BLUECHIP', 'AI_BLUECHIP', signal_by_symbol,
            env={'EXPLORATION_PHASE_ENABLED': 'true', 'EXPLORATION_PHASE_BUY_THRESHOLD': '0.55'},
        )
        self.assertTrue(
            PaperTrade.objects.filter(sandbox='AI_BLUECHIP', ticker='DEADZONEA', broker='SHADOW', status='OPEN').exists(),
            "signal 0.60 cleared the 0.55 exploration buy_threshold but was still "
            "blocked -- Alpaca min_confidence is not tracking buy_threshold.",
        )

    def test_normal_mode_unaffected_signal_above_default_thresholds(self):
        """No exploration, buy_threshold falls back to the hardcoded 0.82
        default (no {PREFIX}_BUY_THRESHOLD/PAPER_BUY_THRESHOLD set here) --
        min(0.7, 0.82) == 0.7, unchanged from before this fix."""
        signal_by_symbol = {'NORMALWINA': 0.90}
        self._run('AI_BLUECHIP', 'AI_BLUECHIP', signal_by_symbol, env={'EXPLORATION_PHASE_ENABLED': 'false'})
        self.assertTrue(
            PaperTrade.objects.filter(sandbox='AI_BLUECHIP', ticker='NORMALWINA', broker='SHADOW', status='OPEN').exists(),
        )
