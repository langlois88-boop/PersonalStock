"""Régression : monitor_my_portfolio échouait à 100% ("Second or Minute
units can only be used with amounts between 1-59") pour les symboles
"core" (interval='60m') -- confirmé en isolant et en exécutant l'appel
fautif en direct contre le vrai SDK Alpaca (pas juste une lecture de
code) :

    >>> from portfolio.market_data import _timeframe_from_interval
    >>> _timeframe_from_interval('60m')
    ValueError: Second or Minute units can only be used with amounts
    between 1-59.

Cause : `_timeframe_from_interval` construisait
`TimeFrame(60, TimeFrameUnit.Minute)`, qu'Alpaca refuse par construction
-- 60 minutes doit être exprimé en heures (`TimeFrame(1, TimeFrameUnit.
Hour)`), exactement comme la branche 'h' de la même fonction le fait déjà
pour les intervalles en heures.
"""

from __future__ import annotations

import unittest

from django.test import TestCase

from portfolio.market_data import TimeFrame, TimeFrameUnit, _timeframe_from_interval


@unittest.skipIf(TimeFrame is None, "alpaca-py non installé dans cet environnement")
class TimeframeFromIntervalTests(TestCase):
    def test_60m_no_longer_raises_and_maps_to_1_hour(self):
        # C'était le crash reproduit en direct ci-dessus avant le fix.
        tf = _timeframe_from_interval('60m')
        self.assertEqual(tf.amount, 1)
        self.assertEqual(tf.unit, TimeFrameUnit.Hour)

    def test_120m_maps_to_2_hours(self):
        tf = _timeframe_from_interval('120m')
        self.assertEqual(tf.amount, 2)
        self.assertEqual(tf.unit, TimeFrameUnit.Hour)

    def test_ordinary_minute_intervals_unaffected(self):
        for interval, expected_amount in [('1m', 1), ('5m', 5), ('15m', 15), ('30m', 30)]:
            tf = _timeframe_from_interval(interval)
            self.assertEqual(tf.amount, expected_amount, interval)
            self.assertEqual(tf.unit, TimeFrameUnit.Minute, interval)

    def test_hour_intervals_unaffected(self):
        tf = _timeframe_from_interval('1h')
        self.assertEqual(tf.amount, 1)
        self.assertEqual(tf.unit, TimeFrameUnit.Hour)

    def test_default_falls_back_to_day(self):
        tf = _timeframe_from_interval(None)
        self.assertIs(tf, TimeFrame.Day)
