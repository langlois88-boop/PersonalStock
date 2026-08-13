"""
Régression pour l'item 9 de TECH_DEBT_NOTES.md (2026-08-13) :
_daily_equity_circuit_breaker utilisait le cache Django par défaut
(LocMemCache, en mémoire du process) -- un redémarrage du worker/NAS
perdait silencieusement l'état "baseline"/"déjà déclenché aujourd'hui",
confirmé en conditions réelles le 2026-08-13 (double déclenchement du
breaker pour AI_BLUECHIP/AI_PENNY avec un `capital` identique aux deux
occasions, corrélé à un redémarrage complet du NAS entre les deux).

Teste contre le vrai client Redis (déjà utilisé comme broker Celery dans
ce projet, disponible sur le réseau docker de test) plutôt qu'un mock --
le but précis de ce test est de vérifier une PERSISTANCE réelle, un mock
en mémoire masquerait exactement le bug qu'on corrige.
"""
from __future__ import annotations

from django.test import TestCase

from . import tasks


class DailyEquityCircuitBreakerRedisPersistenceTests(TestCase):
    def setUp(self):
        # Etat propre avant/après chaque test -- namespace de sandbox
        # dédié pour ne jamais toucher un vrai déclenchement en cours si
        # ce test tournait par erreur contre un Redis partagé.
        self.sandbox = "TEST_CB_SANDBOX"
        tasks.reset_daily_equity_breaker(sandbox=self.sandbox)

    def tearDown(self):
        tasks.reset_daily_equity_breaker(sandbox=self.sandbox)

    def test_first_call_sets_baseline_not_triggered(self):
        result = tasks._daily_equity_circuit_breaker(self.sandbox, 10000.0)
        self.assertFalse(result["triggered"])
        self.assertEqual(result["baseline"], 10000.0)

    def test_small_drawdown_does_not_trigger(self):
        tasks._daily_equity_circuit_breaker(self.sandbox, 10000.0)
        result = tasks._daily_equity_circuit_breaker(self.sandbox, 9900.0)  # -1%
        self.assertFalse(result["triggered"])

    def test_large_drawdown_triggers_once(self):
        tasks._daily_equity_circuit_breaker(self.sandbox, 10000.0)
        result = tasks._daily_equity_circuit_breaker(self.sandbox, 9600.0)  # -4%, seuil par défaut 3%
        self.assertTrue(result["triggered"])
        self.assertTrue(result["first_trigger"])

    def test_state_survives_a_fresh_client_instance(self):
        """
        Le coeur de la régression : simule un redémarrage de worker en
        forçant la recréation du client Redis module-level (jamais
        possible avec l'ancien LocMemCache, qui perdait tout). L'état
        (baseline puis déclenchement) doit survivre.
        """
        tasks._daily_equity_circuit_breaker(self.sandbox, 10000.0)

        # Simule un nouveau process (le client lazy-initialisé est
        # recréé, mais Redis lui-même -- un service séparé -- ne l'est
        # pas, contrairement à LocMemCache qui vit dans le process).
        tasks._circuit_breaker_redis = None

        result = tasks._daily_equity_circuit_breaker(self.sandbox, 9600.0)  # -4%
        self.assertTrue(result["triggered"])
        self.assertEqual(result["baseline"], 10000.0)  # baseline d'origine bien conservée

        # Nouveau "redémarrage" -- doit rester déclenché pour le reste de
        # la journée (comportement voulu, pas un bug).
        tasks._circuit_breaker_redis = None
        result2 = tasks._daily_equity_circuit_breaker(self.sandbox, 9900.0)
        self.assertTrue(result2["triggered"])

    def test_reset_clears_both_keys(self):
        tasks._daily_equity_circuit_breaker(self.sandbox, 10000.0)
        tasks._daily_equity_circuit_breaker(self.sandbox, 9600.0)  # déclenche

        reset_result = tasks.reset_daily_equity_breaker(sandbox=self.sandbox)
        self.assertIn(self.sandbox, reset_result["cleared"])

        # Après reset, un nouvel appel doit repartir de zéro (nouvelle baseline).
        result = tasks._daily_equity_circuit_breaker(self.sandbox, 5000.0)
        self.assertFalse(result["triggered"])
        self.assertEqual(result["baseline"], 5000.0)
