"""Regression tests (2026-08-20) : la phase d'exploration
(EXPLORATION_PHASE_ENABLED, docs/EXPLORATION_PHASE_2026-08.md) écrasait
INCONDITIONNELLEMENT buy_threshold à 0.55, y compris pour des sandboxes
dont le seuil "normal" est déjà PLUS BAS -- WATCHLIST (0.50, petite
hausse non voulue) et surtout AI_PENNY (0.20 -> 0.55, +175%).

Trouvé en diagnostiquant pourquoi AI_PENNY restait bloqué à 1 seul trade
fermé en 10 jours malgré un modèle qui produit des signaux valides pour
la quasi-totalité de sa watchlist (vérifié en direct via _model_signal()) :
sur 100 cycles/10 jours de logs réels, le signal max observé plafonne à
0.4756 -- sous le seuil de 0.55 imposé par l'exploration à CHAQUE cycle,
alors que le seuil réellement calibré pour AI_PENNY (0.20) l'aurait
laissé passer la quasi-totalité du temps.

Fix : min(seuil_normal, seuil_exploration) au lieu d'un écrasement --
l'exploration ne doit qu'ABAISSER un seuil trop strict (AI_BLUECHIP
0.80 -> 0.55), jamais RELEVER un seuil déjà plus permissif.
"""

from __future__ import annotations

from unittest.mock import patch

from django.test import TestCase

from portfolio import tasks
from portfolio.models import PaperTrade
from portfolio.tests_confidence_floor import ConfidenceFloorAlpacaPathTests, ConfidenceFloorSimPathTests


class ExplorationPhaseNeverRaisesASandboxOwnThresholdTests(ConfidenceFloorSimPathTests):
    """Réutilise le harnais de mock de ConfidenceFloorSimPathTests (_run) --
    même chemin SIM, bug différent (buy_threshold lui-même, pas le plancher
    de confiance qui le suit)."""

    def test_ai_penny_exploration_does_not_raise_its_own_lower_020_threshold(self):
        """AI_PENNY normal = 0.20. Signal 0.30 clear largement 0.20 mais
        PAS 0.55 -- avant le fix, l'exploration relevait le seuil à 0.55 et
        bloquait ce trade malgré un seuil normal bien plus permissif."""
        signal_by_symbol = {'PENNYWIN': 0.30}
        stats = self._run(
            'AI_PENNY', 'AI_PENNY', signal_by_symbol,
            env={'EXPLORATION_PHASE_ENABLED': 'true', 'EXPLORATION_PHASE_BUY_THRESHOLD': '0.55'},
        )
        self.assertTrue(
            PaperTrade.objects.filter(sandbox='AI_PENNY', ticker='PENNYWIN', status='OPEN').exists(),
            f"signal 0.30 clear le seuil normal AI_PENNY (0.20) mais l'exploration "
            f"l'a quand même bloqué en relevant le seuil à 0.55. decision_stats={stats}",
        )

    def test_watchlist_exploration_does_not_raise_its_own_050_threshold(self):
        """WATCHLIST normal = 0.50. Signal 0.52 clear 0.50 mais pas 0.55."""
        signal_by_symbol = {'WATCHWIN': 0.52}
        stats = self._run(
            'WATCHLIST', 'WATCHLIST', signal_by_symbol,
            env={'EXPLORATION_PHASE_ENABLED': 'true', 'EXPLORATION_PHASE_BUY_THRESHOLD': '0.55'},
        )
        self.assertTrue(
            PaperTrade.objects.filter(sandbox='WATCHLIST', ticker='WATCHWIN', status='OPEN').exists(),
            f"signal 0.52 clear le seuil normal WATCHLIST (0.50) mais l'exploration "
            f"l'a quand même bloqué en relevant le seuil à 0.55. decision_stats={stats}",
        )

    def test_ai_penny_signal_still_blocked_below_its_own_020_threshold(self):
        """Non-régression : le seuil de 0.20 reste un vrai seuil, pas
        désactivé -- un signal en dessous doit toujours être bloqué."""
        signal_by_symbol = {'PENNYLOSE': 0.10}
        self._run(
            'AI_PENNY', 'AI_PENNY', signal_by_symbol,
            env={'EXPLORATION_PHASE_ENABLED': 'true', 'EXPLORATION_PHASE_BUY_THRESHOLD': '0.55'},
        )
        self.assertFalse(
            PaperTrade.objects.filter(sandbox='AI_PENNY', ticker='PENNYLOSE').exists(),
        )

    def test_ai_bluechip_exploration_still_lowers_its_080_threshold(self):
        """Non-régression directe de l'item 17/confidence floor : l'usage
        principal de l'exploration (abaisser AI_BLUECHIP 0.80 -> 0.55)
        continue de fonctionner exactement comme avant ce fix."""
        signal_by_symbol = {'BLUEWIN': 0.60}
        stats = self._run(
            'AI_BLUECHIP', 'AI_BLUECHIP', signal_by_symbol,
            env={'EXPLORATION_PHASE_ENABLED': 'true', 'EXPLORATION_PHASE_BUY_THRESHOLD': '0.55'},
        )
        self.assertTrue(
            PaperTrade.objects.filter(sandbox='AI_BLUECHIP', ticker='BLUEWIN', status='OPEN').exists(),
            f"decision_stats={stats}",
        )


class ExplorationPhaseAlpacaPathNeverRaisesWatchlistThresholdTests(ConfidenceFloorAlpacaPathTests):
    """Même bug, chemin Alpaca -- WATCHLIST est le seul avec un seuil dédié
    plus bas que 0.55 sur ce chemin (AI_PENNY y retombe sur le défaut 0.82,
    donc l'exploration l'abaisse toujours correctement, voir commentaire
    dans _execute_alpaca_paper_trades_for_sandbox)."""

    def test_watchlist_alpaca_exploration_does_not_raise_its_own_050_threshold(self):
        signal_by_symbol = {'WATCHWINA': 0.52}
        self._run(
            'WATCHLIST', 'WATCHLIST', signal_by_symbol,
            env={'EXPLORATION_PHASE_ENABLED': 'true', 'EXPLORATION_PHASE_BUY_THRESHOLD': '0.55'},
        )
        self.assertTrue(
            PaperTrade.objects.filter(sandbox='WATCHLIST', ticker='WATCHWINA', broker='SHADOW', status='OPEN').exists(),
            "signal 0.52 clear le seuil normal WATCHLIST (0.50) sur le chemin Alpaca "
            "mais l'exploration l'a quand même bloqué en relevant le seuil à 0.55.",
        )


class EffectiveSandboxThresholdsMirrorTests(TestCase):
    """_effective_sandbox_thresholds (RiskSnapshotView, lecture seule) doit
    refléter EXACTEMENT le même comportement que le chemin de trading réel
    -- les deux sont explicitement commentés comme devant rester en
    synchronisation."""

    def test_ai_penny_mirror_matches_real_path_during_exploration(self):
        with patch.dict('os.environ', {'EXPLORATION_PHASE_ENABLED': 'true', 'EXPLORATION_PHASE_BUY_THRESHOLD': '0.55'}):
            result = tasks._effective_sandbox_thresholds('AI_PENNY', 'AI_PENNY')
        self.assertEqual(result['buy_threshold'], 0.20)
        self.assertTrue(result['exploration_override_active'])

    def test_watchlist_mirror_matches_real_path_during_exploration(self):
        with patch.dict('os.environ', {'EXPLORATION_PHASE_ENABLED': 'true', 'EXPLORATION_PHASE_BUY_THRESHOLD': '0.55'}):
            result = tasks._effective_sandbox_thresholds('WATCHLIST', 'WATCHLIST')
        self.assertEqual(result['buy_threshold'], 0.50)

    def test_ai_bluechip_mirror_still_gets_lowered_during_exploration(self):
        with patch.dict('os.environ', {'EXPLORATION_PHASE_ENABLED': 'true', 'EXPLORATION_PHASE_BUY_THRESHOLD': '0.55'}):
            result = tasks._effective_sandbox_thresholds('AI_BLUECHIP', 'AI_BLUECHIP')
        self.assertEqual(result['buy_threshold'], 0.55)
