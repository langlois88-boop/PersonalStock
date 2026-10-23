"""Volet A2 (2026-10) : vente partielle dans _execute_paper_trades_for_sandbox.

Vérifie le cas de test demandé explicitement : quantité 7, ratio 0,5 -> 3
actions vendues (floor, plus de minimum forcé à 1), 4 restantes, pnl partiel
calculé sur ces 3 actions et accumulé dans realized_partial_pnl (séparément
de pnl), initial_quantity/initial_stop_loss jamais modifiés -- y compris
après le recalcul du trailing stop qui suit dans le même cycle.

Watchlist vide (aucun nouveau candidat à l'achat) pour isoler le chemin de
gestion des positions ouvertes du reste de la fonction -- même famille de
mocks que tests_paper_trade_decision_stats.py (DataFusionEngine ajouté ici
pour _volume_z_negative_streak, appelée dès que profit_pct > 0).
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

from django.test import TestCase

from portfolio import tasks
from portfolio.models import PaperTrade
from portfolio.tests_paper_trade_decision_stats import (
    _FakeTicker,
    _fake_get_daily_bars,
    _model_signal_side_effect,
    _set_sandbox_watchlist,
)


class _FakeDataFusionEngine:
    """_volume_z_negative_streak attrape toute exception et retourne
    (False, None) -- pas besoin d'un fuse_all() réaliste, juste rapide et
    sans réseau."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def fuse_all(self):
        return None


class PartialTakeProfitFloorTests(TestCase):
    def test_qty7_ratio_half_sells_3_keeps_4_tracks_initial_and_partial_pnl(self):
        _set_sandbox_watchlist('AI_PENNY', [])

        trade = PaperTrade.objects.create(
            ticker='PARTP',
            sandbox='AI_PENNY',
            entry_price=10.0,
            quantity=7,
            initial_quantity=7,
            stop_loss=9.0,
            initial_stop_loss=9.0,
            status='OPEN',
            broker='SIM',
            model_name='PENNY',
            pnl=0,
        )

        simple_mocks = {
            'get_latest_trade_price': lambda symbol: 11.0,
            'get_daily_bars': _fake_get_daily_bars,
            'get_intraday_context': lambda *a, **k: None,
            '_intraday_context_for_timeframe': lambda *a, **k: None,
            'get_market_sentiment': lambda: ('NEUTRAL', {}),
            '_market_sentiment_score': lambda: 0.5,
            'get_market_regime_context': lambda: {'risk_off': False},
            '_daily_equity_circuit_breaker': lambda sandbox, capital, **kwargs: {'triggered': False},
            '_weak_list_health': lambda: {'defensive': False},
            '_model_signal': _model_signal_side_effect({'PARTP': 0.9}),
            '_system_log': lambda *a, **k: None,
            'DataFusionEngine': _FakeDataFusionEngine,
        }

        with ExitStack() as stack:
            for name, value in simple_mocks.items():
                stack.enter_context(patch.object(tasks, name, value))
            stack.enter_context(patch.object(tasks.yf, 'Ticker', _FakeTicker))
            tasks._execute_paper_trades_for_sandbox('AI_PENNY', 'AI_PENNY')

        trade.refresh_from_db()

        # Vente partielle : floor(7 * 0.5) = 3 vendues, 4 restantes.
        self.assertEqual(float(trade.quantity), 4.0)
        # pnl partiel sur ces 3 actions : (11 - 10) * 3 = 3.
        self.assertAlmostEqual(float(trade.pnl), 3.0, places=2)
        self.assertAlmostEqual(float(trade.realized_partial_pnl), 3.0, places=2)
        self.assertTrue(dict(trade.entry_features or {}).get('partial_taken'))

        # Jamais modifiés, y compris après le recalcul du trailing stop du
        # même cycle (stop_loss, lui, a bougé -- c'est attendu).
        self.assertEqual(float(trade.initial_quantity), 7.0)
        self.assertEqual(float(trade.initial_stop_loss), 9.0)

        # Le trade reste ouvert (pas de stop_hit/signal_exit/velocity_exit
        # à ce prix avec signal=0.9 > sell_threshold=0.4).
        self.assertEqual(trade.status, 'OPEN')

    def test_small_position_below_one_share_skips_partial_sale_entirely(self):
        """floor(1 * 0.5) == 0 -> aucune vente partielle (plus de
        max(1, ...) qui aurait forcé la vente de l'unique action)."""
        _set_sandbox_watchlist('AI_PENNY', [])

        trade = PaperTrade.objects.create(
            ticker='TINYP',
            sandbox='AI_PENNY',
            entry_price=10.0,
            quantity=1,
            initial_quantity=1,
            stop_loss=9.0,
            initial_stop_loss=9.0,
            status='OPEN',
            broker='SIM',
            model_name='PENNY',
            pnl=0,
        )

        simple_mocks = {
            'get_latest_trade_price': lambda symbol: 11.0,
            'get_daily_bars': _fake_get_daily_bars,
            'get_intraday_context': lambda *a, **k: None,
            '_intraday_context_for_timeframe': lambda *a, **k: None,
            'get_market_sentiment': lambda: ('NEUTRAL', {}),
            '_market_sentiment_score': lambda: 0.5,
            'get_market_regime_context': lambda: {'risk_off': False},
            '_daily_equity_circuit_breaker': lambda sandbox, capital, **kwargs: {'triggered': False},
            '_weak_list_health': lambda: {'defensive': False},
            '_model_signal': _model_signal_side_effect({'TINYP': 0.9}),
            '_system_log': lambda *a, **k: None,
            'DataFusionEngine': _FakeDataFusionEngine,
        }

        with ExitStack() as stack:
            for name, value in simple_mocks.items():
                stack.enter_context(patch.object(tasks, name, value))
            stack.enter_context(patch.object(tasks.yf, 'Ticker', _FakeTicker))
            tasks._execute_paper_trades_for_sandbox('AI_PENNY', 'AI_PENNY')

        trade.refresh_from_db()

        self.assertEqual(float(trade.quantity), 1.0)
        self.assertEqual(float(trade.pnl), 0.0)
        self.assertEqual(float(trade.realized_partial_pnl), 0.0)
        self.assertFalse(dict(trade.entry_features or {}).get('partial_taken'))
        self.assertEqual(float(trade.initial_quantity), 1.0)
