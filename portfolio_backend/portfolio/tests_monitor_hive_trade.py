"""Régression : monitor_hive_trade plantait avec "The truth value of a
DataFrame is ambiguous" quand get_latest_trade_price() renvoyait None ET
que le fallback yfinance renvoyait un DataFrame vide.

Cause réelle (confirmée par lecture du code, pas une hypothèse) :
`latest_price` était réutilisée pour porter à la fois le DataFrame
intermédiaire de yfinance et le prix final (float) -- quand le DataFrame
était vide, la variable restait un DataFrame au lieu de retomber sur
None, et `latest_price >= limit_price` plus loin dans la fonction levait
l'erreur de vérité ambiguë au lieu du early-exit 'no_price' attendu.
Confirmé intermittent en production : ~10 échecs en rafale le
2026-08-10 puis plus aucun -- cohérent avec une combinaison rare
(Alpaca ET yfinance en échec pour le même appel).
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pandas as pd
from django.test import TestCase

from portfolio import tasks


def _weekday_market_hours() -> datetime:
    # Mardi 12h00 -- à l'intérieur de la fenêtre 9h30-16h00 (naive, comme
    # _ny_time_now() le renvoie réellement dans le code de production).
    return datetime(2026, 8, 11, 12, 0, 0)


class _FakeTicker:
    """Remplace market_data.Ticker (aliasé `yf` dans tasks.py)."""

    def __init__(self, symbol, history_result):
        self._history_result = history_result

    def history(self, *args, **kwargs):
        return self._history_result


class MonitorHiveTradeEmptyHistoryTests(TestCase):
    def _run_with_history(self, history_result):
        fake_ticker_cls = lambda symbol, _hr=history_result: _FakeTicker(symbol, _hr)  # noqa: E731
        with patch.object(tasks, '_ny_time_now', return_value=_weekday_market_hours()), \
             patch.object(tasks, 'get_latest_trade_price', return_value=None), \
             patch.object(tasks.yf, 'Ticker', fake_ticker_cls), \
             patch.object(tasks, '_send_telegram_alert'):
            return tasks.monitor_hive_trade()

    def test_empty_yfinance_history_falls_back_to_no_price_instead_of_crashing(self):
        result = self._run_with_history(pd.DataFrame())
        self.assertEqual(result.get('status'), 'no_price')

    def test_nonempty_yfinance_history_still_extracts_a_usable_price(self):
        df = pd.DataFrame({'Close': [2.50, 2.60, 2.70]})
        result = self._run_with_history(df)
        self.assertNotEqual(result.get('status'), 'no_price')
        self.assertNotEqual(result.get('status'), 'failed')
