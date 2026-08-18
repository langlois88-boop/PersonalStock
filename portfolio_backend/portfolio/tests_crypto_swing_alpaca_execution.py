"""Tests pour _execute_alpaca_paper_trades_for_crypto_swing et sa tâche
Celery (2026-08-18) -- première exécution réelle jamais câblée pour
AI_CRYPTO (le chemin SIM existant, _execute_paper_trades_for_crypto_
sandbox, n'a jamais été appelé par aucune tâche Beat)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import TestCase

from portfolio import tasks
from portfolio.models import PaperTrade, SandboxWatchlist


class ExecuteAlpacaCryptoSwingTests(TestCase):
    def _fake_account(self, equity=100000.0):
        return SimpleNamespace(equity=equity, cash=equity)

    def test_disabled_by_default(self):
        result = tasks._execute_alpaca_paper_trades_for_crypto_swing()
        self.assertEqual(result['status'], 'disabled')
        self.assertEqual(PaperTrade.objects.count(), 0)

    @patch.dict("os.environ", {"AI_CRYPTO_SWING_ENABLED": "true"})
    def test_no_account_returns_early(self):
        with patch.object(tasks, "get_account", return_value=None):
            result = tasks._execute_alpaca_paper_trades_for_crypto_swing()
        self.assertEqual(result['status'], 'no_account')

    @patch.dict("os.environ", {"AI_CRYPTO_SWING_ENABLED": "true"})
    def test_circuit_breaker_triggered_blocks_trading(self):
        with patch.object(tasks, "get_account", return_value=self._fake_account()), \
                patch.object(tasks, "_daily_equity_circuit_breaker", return_value={"triggered": True}), \
                patch("portfolio.crypto_processor.scan_crypto_swing") as mock_scan:
            result = tasks._execute_alpaca_paper_trades_for_crypto_swing()
        self.assertEqual(result['status'], 'circuit_breaker_triggered')
        mock_scan.assert_not_called()

    @patch.dict("os.environ", {"AI_CRYPTO_SWING_ENABLED": "true"})
    def test_real_pullback_with_successful_order_creates_paper_trade(self):
        scan_result = [{
            'symbol': 'BTC-USD', 'price': 50000.0, 'pullback_pct': -20.0,
            'rsi': 25.0, 'oversold': True, 'btc_panic': False,
            'panic_verdict': 'ROTATION', 'btc_corr': 0.0, 'blocked': False,
            'score': None, 'bb_pct_b_20': 0.1,
            'pattern_hammer': True, 'pattern_morning_star': False,
        }]
        fake_order = MagicMock(id='order-123')
        with patch.object(tasks, "get_account", return_value=self._fake_account(equity=100000.0)), \
                patch.object(tasks, "_daily_equity_circuit_breaker", return_value={"triggered": False}), \
                patch("portfolio.crypto_processor.scan_crypto_swing", return_value=scan_result), \
                patch.object(tasks, "submit_crypto_market_order", return_value=fake_order) as mock_order:
            result = tasks._execute_alpaca_paper_trades_for_crypto_swing()

        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['created'], 1)
        trade = PaperTrade.objects.get(ticker='BTC-USD', sandbox='AI_CRYPTO')
        self.assertEqual(trade.broker, 'ALPACA')
        self.assertEqual(trade.model_name, 'CRYPTO_SWING')
        self.assertEqual(trade.status, 'OPEN')
        self.assertEqual(trade.broker_status, 'submitted')
        # equity 100000 * position_cap_pct 0.05 par defaut / prix 50000 = 0.1 BTC
        self.assertAlmostEqual(float(trade.quantity), 0.1, places=4)
        mock_order.assert_called_once_with('BTC-USD', trade.quantity, 'buy')

    @patch.dict("os.environ", {"AI_CRYPTO_SWING_ENABLED": "true"})
    def test_failed_alpaca_order_does_not_create_phantom_trade(self):
        scan_result = [{
            'symbol': 'BTC-USD', 'price': 50000.0, 'pullback_pct': -20.0,
            'rsi': 25.0, 'oversold': True, 'blocked': False, 'score': None,
        }]
        with patch.object(tasks, "get_account", return_value=self._fake_account()), \
                patch.object(tasks, "_daily_equity_circuit_breaker", return_value={"triggered": False}), \
                patch("portfolio.crypto_processor.scan_crypto_swing", return_value=scan_result), \
                patch.object(tasks, "submit_crypto_market_order", return_value=None):
            result = tasks._execute_alpaca_paper_trades_for_crypto_swing()

        self.assertEqual(result['created'], 0)
        self.assertEqual(PaperTrade.objects.count(), 0)

    @patch.dict("os.environ", {"AI_CRYPTO_SWING_ENABLED": "true"})
    def test_blocked_by_systemic_panic_is_skipped(self):
        scan_result = [{
            'symbol': 'ETH-USD', 'price': 3000.0, 'pullback_pct': -20.0,
            'rsi': 20.0, 'oversold': True, 'blocked': True, 'score': None,
        }]
        with patch.object(tasks, "get_account", return_value=self._fake_account()), \
                patch.object(tasks, "_daily_equity_circuit_breaker", return_value={"triggered": False}), \
                patch("portfolio.crypto_processor.scan_crypto_swing", return_value=scan_result), \
                patch.object(tasks, "submit_crypto_market_order") as mock_order:
            result = tasks._execute_alpaca_paper_trades_for_crypto_swing()

        self.assertEqual(result['created'], 0)
        mock_order.assert_not_called()

    @patch.dict("os.environ", {"AI_CRYPTO_SWING_ENABLED": "true"})
    def test_not_oversold_is_skipped_by_default(self):
        scan_result = [{
            'symbol': 'BTC-USD', 'price': 50000.0, 'pullback_pct': -16.0,
            'rsi': 55.0, 'oversold': False, 'blocked': False, 'score': None,
        }]
        with patch.object(tasks, "get_account", return_value=self._fake_account()), \
                patch.object(tasks, "_daily_equity_circuit_breaker", return_value={"triggered": False}), \
                patch("portfolio.crypto_processor.scan_crypto_swing", return_value=scan_result), \
                patch.object(tasks, "submit_crypto_market_order") as mock_order:
            result = tasks._execute_alpaca_paper_trades_for_crypto_swing()

        self.assertEqual(result['created'], 0)
        mock_order.assert_not_called()

    @patch.dict("os.environ", {"AI_CRYPTO_SWING_ENABLED": "true"})
    def test_ml_score_below_buy_threshold_is_skipped(self):
        scan_result = [{
            'symbol': 'BTC-USD', 'price': 50000.0, 'pullback_pct': -20.0,
            'rsi': 25.0, 'oversold': True, 'blocked': False, 'score': 0.30,
        }]
        with patch.object(tasks, "get_account", return_value=self._fake_account()), \
                patch.object(tasks, "_daily_equity_circuit_breaker", return_value={"triggered": False}), \
                patch("portfolio.crypto_processor.scan_crypto_swing", return_value=scan_result), \
                patch.object(tasks, "submit_crypto_market_order") as mock_order:
            result = tasks._execute_alpaca_paper_trades_for_crypto_swing()

        self.assertEqual(result['created'], 0)
        mock_order.assert_not_called()

    @patch.dict("os.environ", {"AI_CRYPTO_SWING_ENABLED": "true"})
    def test_already_open_position_is_not_duplicated(self):
        PaperTrade.objects.create(
            ticker='BTC-USD', sandbox='AI_CRYPTO', entry_price=48000, quantity=0.1,
            stop_loss=43200, status='OPEN', pnl=0, broker='ALPACA', model_name='CRYPTO_SWING',
        )
        scan_result = [{
            'symbol': 'BTC-USD', 'price': 50000.0, 'pullback_pct': -20.0,
            'rsi': 25.0, 'oversold': True, 'blocked': False, 'score': None,
        }]
        with patch.object(tasks, "get_account", return_value=self._fake_account()), \
                patch.object(tasks, "_daily_equity_circuit_breaker", return_value={"triggered": False}), \
                patch.object(tasks, "_crypto_latest_price", return_value=50000.0), \
                patch("portfolio.crypto_processor.scan_crypto_swing", return_value=scan_result), \
                patch.object(tasks, "submit_crypto_market_order") as mock_order:
            result = tasks._execute_alpaca_paper_trades_for_crypto_swing()

        self.assertEqual(result['created'], 0)
        mock_order.assert_not_called()

    @patch.dict("os.environ", {"AI_CRYPTO_SWING_ENABLED": "true"})
    def test_stop_loss_hit_closes_position_and_submits_sell_order(self):
        PaperTrade.objects.create(
            ticker='BTC-USD', sandbox='AI_CRYPTO', entry_price=50000, quantity=0.1,
            stop_loss=45000, status='OPEN', pnl=0, broker='ALPACA', model_name='CRYPTO_SWING',
        )
        with patch.object(tasks, "get_account", return_value=self._fake_account()), \
                patch.object(tasks, "_daily_equity_circuit_breaker", return_value={"triggered": False}), \
                patch.object(tasks, "_crypto_latest_price", return_value=44000.0), \
                patch("portfolio.crypto_processor.scan_crypto_swing", return_value=[]), \
                patch.object(tasks, "submit_crypto_market_order", return_value=MagicMock()) as mock_order:
            result = tasks._execute_alpaca_paper_trades_for_crypto_swing()

        self.assertEqual(result['closed'], 1)
        trade = PaperTrade.objects.get(ticker='BTC-USD')
        self.assertEqual(trade.status, 'CLOSED')
        self.assertEqual(trade.outcome, 'LOSS')
        mock_order.assert_called_once_with('BTC-USD', 0.1, 'sell')

    @patch.dict("os.environ", {"AI_CRYPTO_SWING_ENABLED": "true"})
    def test_uses_sandboxwatchlist_when_configured(self):
        SandboxWatchlist.objects.update_or_create(
            sandbox='AI_CRYPTO', defaults={'symbols': ['ETH-USD']},
        )
        with patch.object(tasks, "get_account", return_value=self._fake_account()), \
                patch.object(tasks, "_daily_equity_circuit_breaker", return_value={"triggered": False}), \
                patch("portfolio.crypto_processor.scan_crypto_swing", return_value=[]) as mock_scan:
            tasks._execute_alpaca_paper_trades_for_crypto_swing()

        mock_scan.assert_called_once_with(symbols=['ETH-USD'])


class CryptoSwingPaperTradeTaskTests(TestCase):
    def test_logs_success_when_disabled(self):
        from portfolio.models import TaskRunLog
        result = tasks.crypto_swing_paper_trade_task()
        self.assertEqual(result['status'], 'disabled')
        log = TaskRunLog.objects.filter(task_name='crypto_swing_paper_trade_task').latest('started_at')
        self.assertEqual(log.status, 'SUCCESS')
