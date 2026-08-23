"""Régression (2026-08-23) : en creusant l'ampleur réelle du bug
MultiIndex (voir tests_market_data_multiindex.py), vérifié en direct sur
le NAS pourquoi le chemin Alpaca échouait pour TOUS les symboles testés,
y compris SPY/QQQ qu'Alpaca couvre trivialement :

    >>> client.get_stock_bars(StockBarsRequest(..., end=datetime.now(timezone.utc)))
    APIError: {"message":"subscription does not permit querying recent
    SIP data"}

Le plan de données Alpaca de ce compte refuse toute requête dont la
fenêtre se termine dans les 15 dernières minutes -- confirmé précisément
en direct (-14min encore refusé, -16min accepté). Comme download() sans
`end` explicite (l'immense majorité des appels du projet -- juste
`period='1y'` etc.) utilisait `datetime.now()` par défaut, CHAQUE appel
tombait dans la zone interdite, échouait côté Alpaca, et retombait
TOUJOURS sur yfinance -- silencieusement (jamais d'exception qui remonte
à l'appelant), juste un aller-retour Alpaca gaspillé à chaque appel."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from django.test import TestCase

from portfolio import market_data


class DefaultEndRespectsAlpacaSipDelayTests(TestCase):
    def test_default_end_is_at_least_16_minutes_in_the_past(self):
        before = datetime.now(timezone.utc)
        end = market_data._default_end_for_alpaca_window()
        after = datetime.now(timezone.utc)

        # end == "now" (entre before et after) moins 16 minutes -- borné
        # des deux côtés par l'instant de l'appel, à la microseconde près.
        self.assertGreaterEqual(end, before - timedelta(minutes=16))
        self.assertLessEqual(end, after - timedelta(minutes=16))

    def test_download_without_explicit_end_uses_the_delayed_window(self):
        captured = {}

        def fake_bars_for_symbols(symbols, start, end, interval):
            captured['end'] = end
            return __import__('pandas').DataFrame()  # vide -> tombe sur le repli yfinance, sans importance ici

        with patch.object(market_data, "_bars_for_symbols", side_effect=fake_bars_for_symbols), \
                patch.object(market_data, "_allow_yf_price_fallback", return_value=False):
            market_data.download("SPY", period="1y", interval="1d")

        self.assertIn('end', captured)
        self.assertLessEqual(captured['end'], datetime.now(timezone.utc) - timedelta(minutes=16))

    def test_download_with_explicit_end_is_not_overridden(self):
        # Un `end` fourni explicitement par l'appelant est respecté tel
        # quel -- même s'il tombe dans la zone interdite Alpaca, auquel
        # cas ça échoue proprement côté Alpaca et retombe sur yfinance
        # comme avant, sans qu'on force une valeur à sa place.
        explicit_end = datetime(2026, 1, 1, tzinfo=timezone.utc)
        captured = {}

        def fake_bars_for_symbols(symbols, start, end, interval):
            captured['end'] = end
            return __import__('pandas').DataFrame()

        with patch.object(market_data, "_bars_for_symbols", side_effect=fake_bars_for_symbols), \
                patch.object(market_data, "_allow_yf_price_fallback", return_value=False):
            market_data.download("SPY", period="1y", interval="1d", end=explicit_end)

        self.assertEqual(captured['end'], explicit_end)
