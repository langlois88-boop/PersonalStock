from unittest.mock import patch

from rest_framework.test import APITestCase

from . import market_data


class MarketDataTickerInfoTests(APITestCase):
	"""
	Regression: market_data.Ticker.info called get_latest_trade_price without
	ever importing it -- a NameError on every single call. Introduced by
	commit ff798abb (2026-02-22, an import-cleanup that dropped it along with
	two genuinely-unused imports on the same line) and silently swallowed by
	every one of the ~20 call sites across tasks.py/views.py/ai_scout.py/
	ml_engine (all wrap .info in `except Exception`), degrading sector
	classification, correlation grouping, and bid-ask-spread fallback
	features for about 5.5 months before being caught by a dry run of the
	analysis module (fixed in commit 330c3c8). This module is imported as
	`yf` throughout the ML pipeline (tasks.py, views.py, ai_scout.py,
	consumers.py, ml_engine/processor.py) -- this test guards the shared
	entry point directly rather than one specific caller.
	"""

	def test_ticker_info_does_not_raise(self):
		with patch.object(market_data, '_yfinance') as mock_yf, \
				patch.object(market_data, 'get_latest_trade_price', return_value=None):
			mock_yf.Ticker.return_value.info = {'sector': 'Technology', 'regularMarketPrice': 123.45}
			info = market_data.Ticker('AAPL').info

		self.assertEqual(info.get('sector'), 'Technology')
		self.assertEqual(info.get('regularMarketPrice'), 123.45)

	def test_ticker_info_uses_alpaca_price_override_when_available(self):
		with patch.object(market_data, '_yfinance') as mock_yf, \
				patch.object(market_data, 'get_latest_trade_price', return_value=99.9):
			mock_yf.Ticker.return_value.info = {'regularMarketPrice': 1.0}
			info = market_data.Ticker('AAPL').info

		self.assertEqual(info.get('regularMarketPrice'), 99.9)


class HealthEndpointTests(APITestCase):
	def test_health_endpoint_includes_tasks(self):
		response = self.client.get('/api/health/')
		self.assertEqual(response.status_code, 200)
		self.assertIn('tasks', response.data)
		self.assertIn('auto_rollback_models_daily', response.data['tasks'])


class MonitoringEndpointTests(APITestCase):
	def test_monitoring_summary_has_results(self):
		response = self.client.get('/api/models/monitoring/')
		self.assertEqual(response.status_code, 200)
		self.assertIn('results', response.data)
		self.assertTrue(isinstance(response.data['results'], list))


class AccountDashboardTests(APITestCase):
	def test_account_dashboard_returns_accounts(self):
		response = self.client.get('/api/dashboard/accounts/')
		self.assertEqual(response.status_code, 200)
		self.assertIn('accounts', response.data)
		self.assertIn('top_movers', response.data)
