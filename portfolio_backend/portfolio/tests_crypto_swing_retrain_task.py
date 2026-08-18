"""Tests pour retrain_crypto_swing_model_weekly (2026-08-18, tasks.py) --
la tâche Beat hebdomadaire qui réentraîne le pipeline SWING crypto. Un
rejet par les portes de qualité est un résultat SUCCESS attendu (pas une
erreur de tâche) ; seule une vraie exception inattendue doit logger ERROR.
"""

from __future__ import annotations

from unittest.mock import patch

from django.test import TestCase

from portfolio import tasks
from portfolio.ml_engine import crypto_training
from portfolio.models import TaskRunLog


class RetrainCryptoSwingModelWeeklyTests(TestCase):
    def test_uses_btc_eth_only_by_default(self):
        with patch.object(crypto_training, "train_crypto_swing_model", return_value={
            "n_samples": 100, "cv_mean": 0.6, "walk_forward": [{"f1": 0.6}],
        }) as mock_train, \
                patch.object(crypto_training, "save_crypto_swing_model", return_value="/fake/path.pkl"), \
                patch.object(tasks, "_send_telegram_alert"):
            tasks.retrain_crypto_swing_model_weekly()

        called_symbols = mock_train.call_args[0][0]
        self.assertEqual(called_symbols, ["BTC-USD", "ETH-USD"])

    def test_saved_model_sends_telegram_alert_and_logs_success(self):
        payload = {"n_samples": 4344, "cv_mean": 0.62, "walk_forward": [{"f1": 0.55}, {"f1": 0.58}]}
        with patch.object(crypto_training, "train_crypto_swing_model", return_value=payload), \
                patch.object(crypto_training, "save_crypto_swing_model", return_value="/fake/crypto_swing_brain_v1.pkl") as mock_save, \
                patch.object(tasks, "_send_telegram_alert") as mock_alert:
            result = tasks.retrain_crypto_swing_model_weekly()

        mock_save.assert_called_once_with(payload)
        self.assertEqual(result["status"], "deployed")
        self.assertAlmostEqual(result["walk_forward_f1"], 0.565)
        mock_alert.assert_called_once()
        self.assertIn("Nouveau modèle SWING", mock_alert.call_args[0][0])
        log = TaskRunLog.objects.filter(task_name="retrain_crypto_swing_model_weekly").latest("started_at")
        self.assertEqual(log.status, "SUCCESS")

    def test_gate_rejection_is_success_not_error_and_no_alert(self):
        """Rejet par les portes de qualité (cv_mean/walk-forward trop bas) --
        comportement attendu, pas une panne de tâche. Reproduit ce qui a
        été observé en direct le 2026-08-18 (BTC/ETH, walk-forward F1
        0.170-0.301, tous rejetés)."""
        payload = {"n_samples": 4344, "cv_mean": 0.52, "walk_forward": [{"f1": 0.30}]}
        with patch.object(crypto_training, "train_crypto_swing_model", return_value=payload), \
                patch.object(
                    crypto_training, "save_crypto_swing_model",
                    side_effect=ValueError("Walk-forward F1 0.300 below threshold 0.50"),
                ), \
                patch.object(tasks, "_send_telegram_alert") as mock_alert:
            result = tasks.retrain_crypto_swing_model_weekly()

        self.assertEqual(result["status"], "rejected")
        self.assertIn("Walk-forward", result["reason"])
        mock_alert.assert_not_called()
        log = TaskRunLog.objects.filter(task_name="retrain_crypto_swing_model_weekly").latest("started_at")
        self.assertEqual(log.status, "SUCCESS")  # rejet = résultat valide, pas un échec de tâche

    def test_unexpected_training_exception_logs_error(self):
        with patch.object(crypto_training, "train_crypto_swing_model", side_effect=RuntimeError("yfinance down")), \
                patch.object(tasks, "_send_telegram_alert") as mock_alert:
            result = tasks.retrain_crypto_swing_model_weekly()

        self.assertEqual(result["status"], "error")
        mock_alert.assert_not_called()
        log = TaskRunLog.objects.filter(task_name="retrain_crypto_swing_model_weekly").latest("started_at")
        self.assertEqual(log.status, "ERROR")

    def test_symbols_overridable_via_env(self):
        with patch.dict("os.environ", {"CRYPTO_SWING_SYMBOLS": "BTC-USD,ETH-USD,SOL-USD"}), \
                patch.object(crypto_training, "train_crypto_swing_model", return_value={
                    "n_samples": 1, "cv_mean": 0.6, "walk_forward": [{"f1": 0.6}],
                }) as mock_train, \
                patch.object(crypto_training, "save_crypto_swing_model", return_value="/fake/path.pkl"), \
                patch.object(tasks, "_send_telegram_alert"):
            tasks.retrain_crypto_swing_model_weekly()

        called_symbols = mock_train.call_args[0][0]
        self.assertEqual(called_symbols, ["BTC-USD", "ETH-USD", "SOL-USD"])
