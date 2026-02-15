from rest_framework.test import APITestCase


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
