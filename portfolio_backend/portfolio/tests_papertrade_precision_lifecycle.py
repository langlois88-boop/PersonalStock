"""Volet A1 (2026-10) : la précision (14,6) sur entry_price/stop_loss/
exit_price doit survivre à un cycle complet -- création, mise à jour du
trailing stop, puis clôture -- sans jamais repasser par un round(..., 2)
ou round(..., 4) explicite entre-temps (tous retirés dans
_execute_paper_trades_for_sandbox, voir docs/TECH_DEBT_NOTES.md item 25).

Reproduit le cas SDST du diagnostic du 2026-10-22 en miniature : un prix
sous 0,05$ avec des mouvements de l'ordre de 0,001$, invisibles à 2
décimales (tout aurait arrondi à "0.02") mais significatifs à 6.
"""

from __future__ import annotations

from contextlib import ExitStack
from decimal import Decimal
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
from portfolio.tests_partial_take_profit import _FakeDataFusionEngine


class PaperTradePrecisionLifecycleTests(TestCase):
    def _run_cycle(self, price: float) -> None:
        simple_mocks = {
            'get_latest_trade_price': lambda symbol: price,
            'get_daily_bars': _fake_get_daily_bars,
            'get_intraday_context': lambda *a, **k: None,
            '_intraday_context_for_timeframe': lambda *a, **k: None,
            'get_market_sentiment': lambda: ('NEUTRAL', {}),
            '_market_sentiment_score': lambda: 0.5,
            'get_market_regime_context': lambda: {'risk_off': False},
            '_daily_equity_circuit_breaker': lambda sandbox, capital, **kwargs: {'triggered': False},
            '_weak_list_health': lambda: {'defensive': False},
            '_model_signal': _model_signal_side_effect({'PRECP': 0.9}),
            '_system_log': lambda *a, **k: None,
            'DataFusionEngine': _FakeDataFusionEngine,
        }
        with ExitStack() as stack:
            for name, value in simple_mocks.items():
                stack.enter_context(patch.object(tasks, name, value))
            stack.enter_context(patch.object(tasks.yf, 'Ticker', _FakeTicker))
            tasks._execute_paper_trades_for_sandbox('AI_PENNY', 'AI_PENNY')

    def test_precision_survives_entry_trailing_and_close(self):
        _set_sandbox_watchlist('AI_PENNY', [])

        entry_price = Decimal('0.023456')
        trade = PaperTrade.objects.create(
            ticker='PRECP',
            sandbox='AI_PENNY',
            entry_price=entry_price,
            quantity=1000,
            initial_quantity=1000,
            stop_loss=Decimal('0.021000'),
            initial_stop_loss=Decimal('0.021000'),
            status='OPEN',
            broker='SIM',
            model_name='PENNY',
            pnl=0,
        )
        trade.refresh_from_db()
        # Garanti par le champ lui-même (14,6) -- pas par le code -- mais
        # confirme qu'aucune migration/signal Django ne tronque à l'écriture.
        self.assertEqual(trade.entry_price, entry_price)

        # Cycle 1 : +2,01% (sous le seuil de prise de profit partielle à 3%,
        # pour isoler le trailing stop de la vente partielle) -- le
        # break-even (1,5%) se déclenche, new_stop = entry_price exactement.
        self._run_cycle(price=0.023928)
        trade.refresh_from_db()
        self.assertEqual(trade.status, 'OPEN')
        self.assertEqual(trade.quantity, 1000)  # pas de vente partielle ici
        self.assertEqual(trade.stop_loss, entry_price)
        # L'ancien schéma (10,2) aurait stocké "0.02" -- la 6e décimale
        # exacte prouve que plus aucun round(..., 2) n'intervient avant
        # l'écriture en base.
        self.assertEqual(str(trade.stop_loss), '0.023456')

        # Cycle 2 : chute sous le stop trailing du cycle 1 -> clôture.
        exit_price = Decimal('0.020789')
        self._run_cycle(price=float(exit_price))
        trade.refresh_from_db()
        self.assertEqual(trade.status, 'CLOSED')
        self.assertEqual(trade.exit_price, exit_price)
        self.assertEqual(str(trade.exit_price), '0.020789')
        # pnl sur 1000 actions : (0.020789 - 0.023456) * 1000.
        self.assertAlmostEqual(float(trade.pnl), -2.667, places=2)
        # initial_quantity/initial_stop_loss toujours inchangés après clôture.
        self.assertEqual(trade.initial_quantity, 1000)
        self.assertEqual(trade.initial_stop_loss, Decimal('0.021000'))
