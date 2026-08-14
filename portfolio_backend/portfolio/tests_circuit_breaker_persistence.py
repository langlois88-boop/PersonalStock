"""
Régression pour l'item 9 de TECH_DEBT_NOTES.md (2026-08-13) :
_daily_equity_circuit_breaker utilisait le cache Django par défaut
(LocMemCache, en mémoire du process) -- un redémarrage du worker/NAS
perdait silencieusement l'état "baseline"/"déjà déclenché aujourd'hui",
confirmé en conditions réelles le 2026-08-13 (double déclenchement du
breaker pour AI_BLUECHIP/AI_PENNY avec un `capital` identique aux deux
occasions, corrélé à un redémarrage complet du NAS entre les deux).

ET pour l'item 16 (2026-08-14) : le breaker est appelé depuis 2 chemins
pour le même sandbox (SIM ~10-25k$, Alpaca réel ~100k$ partagé) qui
partageaient la même clé de cache -- confirmé en direct le 2026-08-14
qu'une baseline fixée par un chemin (ex. Alpaca, ~100k$) faisait
apparaître l'autre chemin (SIM, ~24k$) comme un "drawdown" de ~-76%,
déclenchant le breaker presque tous les jours sans rapport avec une
vraie perte -- explique très probablement l'essentiel du silence de
AI_BLUECHIP/AI_PENNY documenté dans docs/ML_PIPELINE_AUDIT.md.

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
        tasks._daily_equity_circuit_breaker(self.sandbox, 10000.0, broker_path="SIM")
        tasks._daily_equity_circuit_breaker(self.sandbox, 9600.0, broker_path="SIM")  # déclenche

        reset_result = tasks.reset_daily_equity_breaker(sandbox=self.sandbox)
        self.assertIn(f"{self.sandbox}:SIM", reset_result["cleared"])

        # Après reset, un nouvel appel doit repartir de zéro (nouvelle baseline).
        result = tasks._daily_equity_circuit_breaker(self.sandbox, 5000.0, broker_path="SIM")
        self.assertFalse(result["triggered"])
        self.assertEqual(result["baseline"], 5000.0)


class DailyEquityCircuitBreakerBrokerPathIsolationTests(TestCase):
    """
    Item 16 (2026-08-14) : SIM et Alpaca ne doivent plus jamais partager
    une baseline, même pour le même sandbox -- c'est le coeur du bug
    trouvé en conditions réelles ce soir-là.
    """

    def setUp(self):
        self.sandbox = "TEST_CB_SANDBOX"
        tasks.reset_daily_equity_breaker(sandbox=self.sandbox)

    def tearDown(self):
        tasks.reset_daily_equity_breaker(sandbox=self.sandbox)

    def test_sim_and_alpaca_baselines_are_independent(self):
        # Le chemin Alpaca fixe sa baseline en premier, à une échelle
        # ~10x plus grande que le SIM (reproduit exactement le scénario
        # réel : compte partagé ~100k$ vs comptabilité SIM ~10k$).
        alpaca_result = tasks._daily_equity_circuit_breaker(self.sandbox, 100000.0, broker_path="ALPACA")
        self.assertFalse(alpaca_result["triggered"])
        self.assertEqual(alpaca_result["baseline"], 100000.0)

        # Le chemin SIM, qui n'a JAMAIS vu l'équité Alpaca, ne doit pas
        # être affecté par cette baseline -- avant le fix, ce même appel
        # aurait été comparé à 100000.0 et serait apparu comme un
        # drawdown de ~-90%, déclenchant le breaker à tort.
        sim_result = tasks._daily_equity_circuit_breaker(self.sandbox, 10000.0, broker_path="SIM")
        self.assertFalse(sim_result["triggered"])
        self.assertEqual(sim_result["baseline"], 10000.0)

    def test_sim_trigger_does_not_affect_alpaca_path(self):
        tasks._daily_equity_circuit_breaker(self.sandbox, 10000.0, broker_path="SIM")
        sim_triggered = tasks._daily_equity_circuit_breaker(self.sandbox, 9600.0, broker_path="SIM")  # -4%
        self.assertTrue(sim_triggered["triggered"])

        # Le chemin Alpaca, indépendant, doit pouvoir démarrer normalement.
        alpaca_result = tasks._daily_equity_circuit_breaker(self.sandbox, 100000.0, broker_path="ALPACA")
        self.assertFalse(alpaca_result["triggered"])

    def test_default_broker_path_is_sim(self):
        """Compatibilité : un appel sans broker_path explicite (ancien
        code éventuel) reste sur le chemin SIM, pas un comportement
        ambigu/partagé."""
        result = tasks._daily_equity_circuit_breaker(self.sandbox, 10000.0)
        self.assertFalse(result["triggered"])
        r = tasks._circuit_breaker_redis_client()
        day_key = tasks._ny_time_now().strftime("%Y%m%d")
        self.assertIsNotNone(r.get(f"daily_equity_base:{self.sandbox}:SIM:{day_key}"))
