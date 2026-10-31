"""Régression directe du cas AXG/ATYR (2026-10-29) : deux reverse splits
(1-pour-200 et 1-pour-40) pendant des positions AI_PENNY ouvertes.
_execute_paper_trades_for_sandbox (WATCHLIST/AI_BLUECHIP/AI_PENNY) n'avait
aucune gestion de split -- contrairement à FUNDAMENTAL_LAB (voir
analysis/tasks.py::_apply_split_adjustment_if_needed, cas BRCC 2026-08-24),
d'où un faux gain de +13880%/+3872% et une vente partielle (floor(), volet
A2) exécutée sur ce prix fictif.
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import pandas as pd
from django.test import TestCase
from django.utils import timezone

from portfolio import tasks
from portfolio.models import PaperTrade


def _split_series(split_date, ratio):
    return pd.Series([ratio], index=pd.DatetimeIndex([split_date]))


class SandboxSplitAdjustmentTests(TestCase):
    def _make_trade(self, ticker, entry_date, entry_price, stop_loss, quantity,
                     initial_quantity=None, initial_stop_loss=None, take_profit=None):
        trade = PaperTrade.objects.create(
            ticker=ticker, sandbox='AI_PENNY', entry_price=entry_price,
            quantity=quantity, initial_quantity=initial_quantity,
            stop_loss=stop_loss, initial_stop_loss=initial_stop_loss,
            take_profit=take_profit, status='OPEN', broker='SIM',
            model_name='PENNY_RVOL_SENTIMENT',
        )
        PaperTrade.objects.filter(pk=trade.pk).update(entry_date=entry_date)
        trade.refresh_from_db()
        return trade

    def test_axg_like_reverse_split_1_for_200_adjusts_everything(self):
        entry_date = timezone.now() - timedelta(days=2)
        split_date = timezone.now() - timedelta(hours=10)
        trade = self._make_trade(
            'AXGLIKE', entry_date, entry_price=0.0299, stop_loss=0.02832,
            quantity=16722.0, initial_quantity=33444.0, initial_stop_loss=0.009918,
        )
        with patch.object(tasks.yf, 'Ticker') as mock_ticker_cls:
            mock_ticker_cls.return_value.splits = _split_series(split_date, 0.005)
            changed = tasks._apply_sandbox_split_adjustment_if_needed(trade)

        self.assertTrue(changed)
        trade.refresh_from_db()
        self.assertAlmostEqual(float(trade.entry_price), 5.98, places=3)
        self.assertAlmostEqual(float(trade.stop_loss), 5.664, places=3)
        self.assertAlmostEqual(float(trade.initial_stop_loss), 1.9836, places=3)
        self.assertAlmostEqual(float(trade.quantity), 83.61, places=2)
        self.assertAlmostEqual(float(trade.initial_quantity), 167.22, places=2)
        self.assertIn('Split détecté', trade.notes)

    def test_atyr_like_reverse_split_1_for_40_adjusts_everything(self):
        entry_date = timezone.now() - timedelta(days=2)
        split_date = timezone.now() - timedelta(hours=10)
        trade = self._make_trade(
            'ATYRLIKE', entry_date, entry_price=0.1269, stop_loss=0.121824,
            quantity=3940.0, initial_quantity=7880.0, initial_stop_loss=0.099675,
        )
        with patch.object(tasks.yf, 'Ticker') as mock_ticker_cls:
            mock_ticker_cls.return_value.splits = _split_series(split_date, 0.025)
            changed = tasks._apply_sandbox_split_adjustment_if_needed(trade)

        self.assertTrue(changed)
        trade.refresh_from_db()
        self.assertAlmostEqual(float(trade.entry_price), 5.076, places=3)
        self.assertAlmostEqual(float(trade.quantity), 98.5, places=2)
        self.assertAlmostEqual(float(trade.initial_quantity), 197.0, places=2)

    def test_split_before_entry_is_ignored(self):
        entry_date = timezone.now() - timedelta(hours=2)
        split_date = timezone.now() - timedelta(days=10)  # avant l'entrée -- non pertinent
        trade = self._make_trade(
            'NOSPLIT', entry_date, entry_price=1.0, stop_loss=0.95,
            quantity=100.0, initial_quantity=100.0, initial_stop_loss=0.95,
        )
        with patch.object(tasks.yf, 'Ticker') as mock_ticker_cls:
            mock_ticker_cls.return_value.splits = _split_series(split_date, 0.1)
            changed = tasks._apply_sandbox_split_adjustment_if_needed(trade)

        self.assertFalse(changed)
        trade.refresh_from_db()
        self.assertEqual(float(trade.entry_price), 1.0)
        self.assertEqual(float(trade.quantity), 100.0)

    def test_no_split_data_leaves_trade_untouched(self):
        entry_date = timezone.now() - timedelta(days=2)
        trade = self._make_trade(
            'PLAIN', entry_date, entry_price=1.0, stop_loss=0.95,
            quantity=100.0, initial_quantity=100.0, initial_stop_loss=0.95,
        )
        with patch.object(tasks.yf, 'Ticker') as mock_ticker_cls:
            mock_ticker_cls.return_value.splits = pd.Series(dtype='float64')
            changed = tasks._apply_sandbox_split_adjustment_if_needed(trade)

        self.assertFalse(changed)
        trade.refresh_from_db()
        self.assertEqual(float(trade.entry_price), 1.0)

    def test_fetch_failure_is_caught_and_leaves_trade_untouched(self):
        entry_date = timezone.now() - timedelta(days=2)
        trade = self._make_trade(
            'ERRTICK', entry_date, entry_price=1.0, stop_loss=0.95,
            quantity=100.0, initial_quantity=100.0, initial_stop_loss=0.95,
        )
        with patch.object(tasks.yf, 'Ticker', side_effect=RuntimeError('boom')):
            changed = tasks._apply_sandbox_split_adjustment_if_needed(trade)

        self.assertFalse(changed)
        trade.refresh_from_db()
        self.assertEqual(float(trade.entry_price), 1.0)

    def test_initial_fields_null_are_left_null_after_adjustment(self):
        """Trades créés avant le volet A2 (initial_quantity/initial_stop_loss
        NULL) : le split ajuste entry_price/stop_loss/quantity, pas de
        tentative de deviner une valeur initiale qui n'a jamais existé."""
        entry_date = timezone.now() - timedelta(days=2)
        split_date = timezone.now() - timedelta(hours=10)
        trade = self._make_trade(
            'OLDTRADE', entry_date, entry_price=0.0299, stop_loss=0.02832,
            quantity=16722.0, initial_quantity=None, initial_stop_loss=None,
        )
        with patch.object(tasks.yf, 'Ticker') as mock_ticker_cls:
            mock_ticker_cls.return_value.splits = _split_series(split_date, 0.005)
            changed = tasks._apply_sandbox_split_adjustment_if_needed(trade)

        self.assertTrue(changed)
        trade.refresh_from_db()
        self.assertAlmostEqual(float(trade.entry_price), 5.98, places=3)
        self.assertIsNone(trade.initial_quantity)
        self.assertIsNone(trade.initial_stop_loss)
