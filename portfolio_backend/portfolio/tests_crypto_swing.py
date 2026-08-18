"""Tests pour scan_crypto_swing (2026-08-18, crypto_processor.py) --
équivalent SWING (bougies journalières, déclenché sur un recul depuis un
sommet récent) de scan_crypto_drip (intraday, déclenché sur une chute
ponctuelle en 15 min). Ni l'un ni l'autre n'avait de couverture de test
avant cette suite."""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
from django.test import TestCase

from portfolio import crypto_processor


def _flat_ohlcv(n: int = 250, price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "timestamp": idx,
        "open": [price] * n,
        "high": [price * 1.005] * n,
        "low": [price * 0.995] * n,
        "close": [price] * n,
        "volume": [1_000_000.0] * n,
    })


def _pulled_back_ohlcv(n: int = 250, high: float = 100.0, drop_pct: float = 0.25) -> pd.DataFrame:
    """n-10 jours plats au sommet, puis une chute de drop_pct sur les 10
    derniers jours -- simule un vrai recul depuis un sommet récent (ce que
    _swing_pullback_pct doit détecter)."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    flat = [high] * (n - 10)
    falling = list(np.linspace(high, high * (1 - drop_pct), 10))
    close = flat + falling
    return pd.DataFrame({
        "timestamp": idx,
        "open": close,
        "high": [c * 1.005 for c in close],
        "low": [c * 0.995 for c in close],
        "close": close,
        "volume": [1_000_000.0] * n,
    })


class ScanCryptoSwingTests(TestCase):
    def setUp(self):
        # _crypto_panic_verdict retombe sur 'SYSTEMIC' par défaut si aucune
        # URL Ollama/DeepSeek n'est configurée -- déterministe sans mock
        # réseau, mais on force explicitement pour ne pas dépendre de l'env
        # ambiant du poste qui fait tourner les tests.
        patcher = patch.dict(os.environ, {
            "DANAS_CHAT_BASE_URL": "", "OLLAMA_CHAT_BASE_URL": "", "OLLAMA_BASE_URL": "",
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        crypto_processor._CRYPTO_PANIC_CACHE = {"ts": 0.0, "verdict": None, "reason": None}

    def test_no_pullback_no_signal(self):
        """Un titre plat (aucun recul depuis son sommet de 20j) ne doit
        jamais être évalué -- la question 'la chute est-elle finie ?' n'a
        pas de sens sans chute."""
        with patch.object(crypto_processor, "fetch_crypto_history_daily", return_value=_flat_ohlcv()):
            results = crypto_processor.scan_crypto_swing(symbols=["ETH-USD"])
        self.assertEqual(results, [])

    def test_real_pullback_is_detected_and_scored(self):
        pulled_back = _pulled_back_ohlcv(drop_pct=0.25)  # -25%, au-delà du seuil par défaut (15%)

        def fake_fetch(symbol, years=2):
            return pulled_back if symbol == "ETH-USD" else _flat_ohlcv()

        with patch.object(crypto_processor, "fetch_crypto_history_daily", side_effect=fake_fetch):
            results = crypto_processor.scan_crypto_swing(symbols=["ETH-USD"])

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result["symbol"], "ETH-USD")
        self.assertLess(result["pullback_pct"], -15.0)
        self.assertIsNone(result["score"])  # aucun modèle swing entraîné dans l'environnement de test
        self.assertIn("rsi", result)
        self.assertIn("bb_pct_b_20", result)

    def test_pullback_threshold_is_configurable_via_env(self):
        """Un recul de -10% ne déclenche rien au seuil par défaut (15%),
        mais doit déclencher si le seuil est abaissé via env var."""
        mild_pullback = _pulled_back_ohlcv(drop_pct=0.10)

        def fake_fetch(symbol, years=2):
            return mild_pullback if symbol == "ETH-USD" else _flat_ohlcv()

        with patch.object(crypto_processor, "fetch_crypto_history_daily", side_effect=fake_fetch):
            results_default = crypto_processor.scan_crypto_swing(symbols=["ETH-USD"])
            with patch.dict(os.environ, {"CRYPTO_SWING_PULLBACK_THRESHOLD": "0.05"}):
                results_lowered = crypto_processor.scan_crypto_swing(symbols=["ETH-USD"])

        self.assertEqual(results_default, [])
        self.assertEqual(len(results_lowered), 1)

    def test_score_computed_when_swing_model_present(self):
        """Si un modèle swing entraîné existe, son score doit être calculé
        et injecté dans le résultat -- pas juste laissé à None."""
        pulled_back = _pulled_back_ohlcv(drop_pct=0.25)

        class _FakeModel:
            def predict_proba(self, X):
                return [[0.35, 0.65]]  # classe positive = 0.65

        fake_payload = {"model": _FakeModel(), "features": ["rsi_14", "bb_pct_b_20"]}

        def fake_fetch(symbol, years=2):
            return pulled_back if symbol == "ETH-USD" else _flat_ohlcv()

        with patch.object(crypto_processor, "fetch_crypto_history_daily", side_effect=fake_fetch), \
                patch.object(crypto_processor, "_load_crypto_swing_model", return_value=fake_payload):
            results = crypto_processor.scan_crypto_swing(symbols=["ETH-USD"])

        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]["score"], 0.65)

    def test_btc_itself_never_blocked_by_its_own_panic(self):
        pulled_back = _pulled_back_ohlcv(drop_pct=0.30)
        with patch.object(crypto_processor, "fetch_crypto_history_daily", return_value=pulled_back), \
                patch.object(crypto_processor, "_btc_panic", return_value=(True, 0.9)), \
                patch.object(crypto_processor, "_crypto_panic_verdict", return_value="SYSTEMIC"):
            results = crypto_processor.scan_crypto_swing(symbols=["BTC-USD"])

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["blocked"])  # BTC-USD est explicitement exempté de son propre signal de panique

    def test_altcoin_blocked_during_systemic_btc_panic(self):
        pulled_back = _pulled_back_ohlcv(drop_pct=0.30)
        with patch.object(crypto_processor, "fetch_crypto_history_daily", return_value=pulled_back), \
                patch.object(crypto_processor, "_btc_panic", return_value=(True, 0.9)), \
                patch.object(crypto_processor, "_crypto_panic_verdict", return_value="SYSTEMIC"):
            results = crypto_processor.scan_crypto_swing(symbols=["ETH-USD"])

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["blocked"])

    def test_empty_history_skips_symbol_without_crashing(self):
        with patch.object(crypto_processor, "fetch_crypto_history_daily", return_value=pd.DataFrame()):
            results = crypto_processor.scan_crypto_swing(symbols=["DOGE-USD"])
        self.assertEqual(results, [])
