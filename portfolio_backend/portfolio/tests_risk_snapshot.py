"""Regression tests for RiskSnapshotView -- the read-only replacement for
RiskControlCenter.js's fake localStorage-only sliders (2026-08-16, Eric
explicitly chose this over building a real write path, for zero risk on
the live trading path). Confirms it reports the real, currently-active
per-sandbox thresholds (including the exploration phase override), not
placeholder values.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from rest_framework.test import APITestCase


class RiskSnapshotViewTests(APITestCase):
    def test_normal_mode_reports_real_per_sandbox_thresholds(self):
        with patch.dict(os.environ, {'EXPLORATION_PHASE_ENABLED': 'false'}):
            response = self.client.get('/api/risk-snapshot/')

        self.assertEqual(response.status_code, 200)
        by_sandbox = {row['sandbox']: row for row in response.data['sandboxes']}
        self.assertEqual(by_sandbox['WATCHLIST']['buy_threshold'], 0.50)
        self.assertEqual(by_sandbox['WATCHLIST']['sell_threshold'], 0.30)
        self.assertEqual(by_sandbox['AI_BLUECHIP']['buy_threshold'], 0.80)
        self.assertEqual(by_sandbox['AI_PENNY']['buy_threshold'], 0.20)
        self.assertFalse(by_sandbox['WATCHLIST']['exploration_override_active'])

    def test_exploration_mode_only_lowers_thresholds_never_raises_them(self):
        """Corrigé le 2026-08-20 : l'exploration écrasait INCONDITIONNELLEMENT
        buy_threshold à 0.55 pour toutes les sandboxes, y compris celles dont
        le seuil normal est déjà plus bas (WATCHLIST 0.50, AI_PENNY 0.20) --
        les relevant sans raison au lieu de les laisser telles quelles.
        min(seuil_normal, 0.55) : AI_BLUECHIP (0.80) est abaissé comme prévu,
        WATCHLIST/AI_PENNY restent à leur propre seuil, déjà plus permissif."""
        with patch.dict(os.environ, {
            'EXPLORATION_PHASE_ENABLED': 'true',
            'EXPLORATION_PHASE_BUY_THRESHOLD': '0.55',
        }):
            response = self.client.get('/api/risk-snapshot/')

        by_sandbox = {row['sandbox']: row for row in response.data['sandboxes']}
        self.assertEqual(by_sandbox['WATCHLIST']['buy_threshold'], 0.50)
        self.assertEqual(by_sandbox['AI_PENNY']['buy_threshold'], 0.20)
        self.assertEqual(by_sandbox['AI_BLUECHIP']['buy_threshold'], 0.55)
        for sandbox in ('WATCHLIST', 'AI_BLUECHIP', 'AI_PENNY'):
            self.assertTrue(by_sandbox[sandbox]['exploration_override_active'])

    def test_global_risk_params_present(self):
        response = self.client.get('/api/risk-snapshot/')
        self.assertIn('daily_drawdown_circuit_breaker_pct', response.data)
        self.assertIn('dynamic_position_min', response.data)
        self.assertIn('dynamic_position_max', response.data)
