"""Tests pour les helpers crypto d'alpaca_data.py (2026-08-18) --
submit_market_order (actions) utilise qty entier + TimeInForce.DAY ; le
marché crypto d'Alpaca rejette DAY (24/7, pas de session) et a besoin de
quantités fractionnaires -- deux différences faciles à rater en copiant
juste submit_market_order tel quel."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.test import TestCase

from portfolio import alpaca_data


class ToAlpacaCryptoSymbolTests(TestCase):
    def test_converts_dash_format_to_slash(self):
        self.assertEqual(alpaca_data._to_alpaca_crypto_symbol('BTC-USD'), 'BTC/USD')
        self.assertEqual(alpaca_data._to_alpaca_crypto_symbol('eth-usd'), 'ETH/USD')

    def test_leaves_already_slash_format_unchanged(self):
        self.assertEqual(alpaca_data._to_alpaca_crypto_symbol('BTC/USD'), 'BTC/USD')

    def test_empty_symbol_returns_empty(self):
        self.assertEqual(alpaca_data._to_alpaca_crypto_symbol(''), '')
        self.assertEqual(alpaca_data._to_alpaca_crypto_symbol(None), '')


class SubmitCryptoMarketOrderTests(TestCase):
    def test_uses_gtc_not_day_and_fractional_qty(self):
        """C'est exactement le bug qu'un copier-coller de submit_market_order
        aurait introduit silencieusement -- DAY est rejeté par l'API Alpaca
        pour les ordres crypto (marché 24/7), et qty=0.0023 BTC arrondi à
        l'entier le plus proche donnerait qty=0, un ordre qui ne fait rien."""
        mock_client = MagicMock()
        with patch.object(alpaca_data, '_alpaca_trading_client', return_value=mock_client):
            alpaca_data.submit_crypto_market_order('BTC-USD', 0.0023, 'buy')

        mock_client.submit_order.assert_called_once()
        request = mock_client.submit_order.call_args[0][0]
        self.assertEqual(request.symbol, 'BTC/USD')
        self.assertEqual(request.time_in_force, alpaca_data.TimeInForce.GTC)
        self.assertAlmostEqual(float(request.qty), 0.0023)

    def test_sell_side_maps_correctly(self):
        mock_client = MagicMock()
        with patch.object(alpaca_data, '_alpaca_trading_client', return_value=mock_client):
            alpaca_data.submit_crypto_market_order('ETH-USD', 0.5, 'sell')

        request = mock_client.submit_order.call_args[0][0]
        self.assertEqual(request.side, alpaca_data.OrderSide.SELL)

    def test_returns_none_without_raising_when_no_client(self):
        with patch.object(alpaca_data, '_alpaca_trading_client', return_value=None):
            result = alpaca_data.submit_crypto_market_order('BTC-USD', 0.001, 'buy')
        self.assertIsNone(result)

    def test_returns_none_for_zero_or_negative_qty(self):
        mock_client = MagicMock()
        with patch.object(alpaca_data, '_alpaca_trading_client', return_value=mock_client):
            self.assertIsNone(alpaca_data.submit_crypto_market_order('BTC-USD', 0, 'buy'))
            self.assertIsNone(alpaca_data.submit_crypto_market_order('BTC-USD', -0.001, 'buy'))
        mock_client.submit_order.assert_not_called()

    def test_swallows_api_exception_returns_none(self):
        mock_client = MagicMock()
        mock_client.submit_order.side_effect = RuntimeError('Alpaca API 422')
        with patch.object(alpaca_data, '_alpaca_trading_client', return_value=mock_client):
            result = alpaca_data.submit_crypto_market_order('BTC-USD', 0.001, 'buy')
        self.assertIsNone(result)
