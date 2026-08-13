"""Regression tests for the temporary exploration-phase helpers
(_exploration_phase_enabled / _exploration_position_value), added
2026-08-12 -- see docs/EXPLORATION_PHASE_2026-08.md. Pure functions, no
network/DB mocking needed.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from django.test import TestCase

from portfolio import tasks


class ExplorationPhaseFlagTests(TestCase):
    def test_disabled_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('EXPLORATION_PHASE_ENABLED', None)
            self.assertFalse(tasks._exploration_phase_enabled())

    def test_enabled_when_flag_set_true(self) -> None:
        with patch.dict(os.environ, {'EXPLORATION_PHASE_ENABLED': 'true'}):
            self.assertTrue(tasks._exploration_phase_enabled())

    def test_disabled_when_flag_set_false(self) -> None:
        with patch.dict(os.environ, {'EXPLORATION_PHASE_ENABLED': 'false'}):
            self.assertFalse(tasks._exploration_phase_enabled())


class ExplorationPositionValueTests(TestCase):
    def test_defaults_to_1000(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('EXPLORATION_PHASE_POSITION_SIZE', None)
            self.assertEqual(tasks._exploration_position_value(50000.0), 1000.0)

    def test_never_exceeds_available_capital(self) -> None:
        """The one hard guardrail explicitly requested: exploration sizing
        must never exceed the sandbox's real available capital, even when
        the configured target ($1000 or an override) is larger."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('EXPLORATION_PHASE_POSITION_SIZE', None)
            self.assertEqual(tasks._exploration_position_value(500.0), 500.0)
            self.assertEqual(tasks._exploration_position_value(0.0), 0.0)
            self.assertEqual(tasks._exploration_position_value(-50.0), 0.0)

    def test_respects_override(self) -> None:
        with patch.dict(os.environ, {'EXPLORATION_PHASE_POSITION_SIZE': '250'}):
            self.assertEqual(tasks._exploration_position_value(50000.0), 250.0)
