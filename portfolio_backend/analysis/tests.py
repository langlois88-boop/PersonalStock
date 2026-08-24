import json
from datetime import timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd
from django.utils import timezone
from rest_framework.test import APITestCase

from portfolio.models import PaperTrade, SandboxWatchlist

from .models import (
    ScreenerPreset, ScanRun, ScanResult, FundamentalLabPosition,
    RejectionLog, RejectionOutcome, ManualTickerCheck, LabWeeklySuggestion,
)
from . import tasks
from .services import quant_filter, manual_check
from .services.deepseek_triage import DeepSeekResult
from .services.claude_verifier import ClaudeResult
from .services.quant_filter import QuantResult
from .services.sanity_check import check_sanity


def _make_preset(**kwargs):
    defaults = dict(
        slug="undervalued-without-reason",
        name="Undervalued Without Reason",
        requires_news_check=True,
        is_active=True,
    )
    defaults.update(kwargs)
    return ScreenerPreset.objects.create(**defaults)


def _make_quant_result(ticker="WINW", score=3.5):
    return QuantResult(
        ticker=ticker,
        sector="Technology",
        passed=True,
        score=score,
        price=12.5,
        details={"pe_ratio": 8.2, "peg_ratio": 0.9, "roe": 18.0, "sector": "Technology"},
        reasons_failed=[],
    )


class GetUniverseTests(APITestCase):
    """
    TODO #1 du brief : _get_universe doit lire les vrais sandboxes
    (SandboxWatchlist.symbols, un JSON par sandbox), pas un modèle
    'Sandbox' par ticker qui n'existe pas dans ce repo.
    """

    def test_universe_merges_watchlist_bluechip_penny_only(self):
        SandboxWatchlist.objects.create(sandbox="WATCHLIST", symbols=["AAA", "bbb"])
        SandboxWatchlist.objects.create(sandbox="AI_BLUECHIP", symbols=["BBB", "CCC"])
        SandboxWatchlist.objects.create(sandbox="AI_PENNY", symbols=["DDD"])
        # AI_CRYPTO n'est pas dans l'univers V1 (cf. brief : sandboxes existants
        # uniquement, pas d'élargissement pour l'instant) — doit être ignoré.
        SandboxWatchlist.objects.create(sandbox="AI_CRYPTO", symbols=["BTC-USD"])

        universe = tasks._get_universe("sandboxes")

        self.assertEqual(universe, ["AAA", "BBB", "CCC", "DDD"])

    def test_universe_empty_when_no_sandboxes(self):
        self.assertEqual(tasks._get_universe("sandboxes"), [])

    def test_unknown_source_returns_empty(self):
        self.assertEqual(tasks._get_universe("custom"), [])


class CreateLabPositionTests(APITestCase):
    """
    TODO #2 du brief : connecter FundamentalLabPosition à un vrai PaperTrade
    (sandbox FUNDAMENTAL_LAB), sans réintroduire le bug NameError/silencieux
    déjà corrigé une fois sur ce projet (commit 3c01b62).
    """

    def setUp(self):
        self.preset = _make_preset()
        self.scan_run = ScanRun.objects.create(preset=self.preset, universe_source="sandboxes")

    def _make_scan_result(self, price="12.5000", verdict="confirmed"):
        return ScanResult.objects.create(
            scan_run=self.scan_run,
            ticker="WINW",
            sector="Technology",
            quant_score=3.5,
            price_at_scan=Decimal(price),
            final_verdict=verdict,
        )

    def _fake_alpaca_order(self):
        """
        _create_lab_position (Partie B, 2026-08-11) place maintenant un vrai
        ordre Alpaca via submit_lab_market_order pour tout ticker non-TSX --
        mocké ici plutôt que de laisser l'appel réseau réel se produire en
        test (retournerait None hors environnement configuré, faisant
        échouer silencieusement toute la branche avant même d'atteindre
        PaperTrade.objects.create).
        """
        order = MagicMock()
        order.id = "fake-order-id-123"
        order.status = "accepted"
        return order

    def test_creates_lab_position_and_real_paper_trade(self):
        scan_result = self._make_scan_result()

        with patch.object(tasks, "submit_lab_market_order", return_value=self._fake_alpaca_order()):
            tasks._create_lab_position(scan_result, self.preset, Decimal("21500.00"))

        lab_position = FundamentalLabPosition.objects.get(scan_result=scan_result)
        self.assertEqual(lab_position.ticker, "WINW")
        self.assertFalse(lab_position.manually_selected)
        self.assertIsNotNone(lab_position.paper_trade_id)

        trade = PaperTrade.objects.get(id=lab_position.paper_trade_id)
        self.assertEqual(trade.sandbox, "FUNDAMENTAL_LAB")
        self.assertEqual(trade.ticker, "WINW")
        self.assertGreater(trade.quantity, 0)
        self.assertLess(float(trade.stop_loss), float(trade.entry_price))
        # Ne doit jamais hériter du défaut 'BLUECHIP' du champ model_name —
        # sinon un futur groupby par model_name (sans regarder le sandbox)
        # pourrait le confondre avec le vrai modèle ML BLUECHIP.
        self.assertEqual(trade.model_name, "FUNDAMENTAL_LAB")

    def test_invalid_price_keeps_lab_position_without_paper_trade(self):
        """
        Un prix d'entrée invalide (0/absent) ne doit jamais faire disparaître
        le pick silencieusement : la position lab existe quand même, sans
        ordre, et rien ne plante.
        """
        scan_result = self._make_scan_result(price="0")

        tasks._create_lab_position(scan_result, self.preset, Decimal("21500.00"))

        lab_position = FundamentalLabPosition.objects.get(scan_result=scan_result)
        self.assertIsNone(lab_position.paper_trade_id)
        self.assertEqual(PaperTrade.objects.filter(sandbox="FUNDAMENTAL_LAB").count(), 0)

    @patch.object(tasks.PaperTrade.objects, "create", side_effect=RuntimeError("boom"))
    def test_paper_trade_creation_failure_is_caught_and_logged(self, mock_create):
        """Une exception pendant la création du PaperTrade ne doit jamais remonter."""
        scan_result = self._make_scan_result()

        with patch.object(tasks, "submit_lab_market_order", return_value=self._fake_alpaca_order()):
            tasks._create_lab_position(scan_result, self.preset, Decimal("21500.00"))

        lab_position = FundamentalLabPosition.objects.get(scan_result=scan_result)
        self.assertIsNone(lab_position.paper_trade_id)
        mock_create.assert_called_once()

    def test_reconfirming_ticker_with_open_position_does_not_duplicate(self):
        """
        Régression : 3 scans le même soir (2026-08-10, 20h26/20h59/21h28
        UTC) ont chacun créé une nouvelle FundamentalLabPosition + un
        nouveau PaperTrade pour BTO.TO, faute de vérifier une position déjà
        ouverte -- ~296$ engagés au lieu de ~99$. Un ScanResult qui
        reconfirme un ticker déjà en position ouverte ne doit plus créer de
        doublon.
        """
        first_scan_result = self._make_scan_result()
        with patch.object(tasks, "submit_lab_market_order", return_value=self._fake_alpaca_order()):
            tasks._create_lab_position(first_scan_result, self.preset, Decimal("21500.00"))
        self.assertEqual(FundamentalLabPosition.objects.filter(ticker="WINW").count(), 1)
        self.assertEqual(PaperTrade.objects.filter(sandbox="FUNDAMENTAL_LAB").count(), 1)

        second_scan_result = ScanResult.objects.create(
            scan_run=self.scan_run, ticker="WINW", sector="Technology",
            quant_score=3.4, price_at_scan=Decimal("12.6000"), final_verdict="confirmed",
        )
        tasks._create_lab_position(second_scan_result, self.preset, Decimal("21500.00"))

        # Toujours une seule position/ordre -- le 2e scan n'a rien créé.
        self.assertEqual(FundamentalLabPosition.objects.filter(ticker="WINW").count(), 1)
        self.assertEqual(PaperTrade.objects.filter(sandbox="FUNDAMENTAL_LAB").count(), 1)
        self.assertFalse(FundamentalLabPosition.objects.filter(scan_result=second_scan_result).exists())

    def test_reconfirming_ticker_after_position_closed_allows_new_entry(self):
        """Une position déjà fermée (is_open=False) n'empêche pas une nouvelle entrée."""
        first_scan_result = self._make_scan_result()
        tasks._create_lab_position(first_scan_result, self.preset, Decimal("21500.00"))
        FundamentalLabPosition.objects.filter(scan_result=first_scan_result).update(is_open=False)

        second_scan_result = ScanResult.objects.create(
            scan_run=self.scan_run, ticker="WINW", sector="Technology",
            quant_score=3.4, price_at_scan=Decimal("12.6000"), final_verdict="confirmed",
        )
        tasks._create_lab_position(second_scan_result, self.preset, Decimal("21500.00"))

        self.assertEqual(FundamentalLabPosition.objects.filter(ticker="WINW").count(), 2)
        self.assertTrue(FundamentalLabPosition.objects.filter(scan_result=second_scan_result).exists())

    def test_same_ticker_different_preset_is_not_deduplicated(self):
        """Le dédoublonnage est scopé par ticker+preset, pas juste par ticker."""
        first_scan_result = self._make_scan_result()
        tasks._create_lab_position(first_scan_result, self.preset, Decimal("21500.00"))

        other_preset = _make_preset(slug="value-catalyst-other", name="Autre preset")
        other_scan_run = ScanRun.objects.create(preset=other_preset, universe_source="sandboxes")
        second_scan_result = ScanResult.objects.create(
            scan_run=other_scan_run, ticker="WINW", sector="Technology",
            quant_score=3.4, price_at_scan=Decimal("12.6000"), final_verdict="confirmed",
        )
        tasks._create_lab_position(second_scan_result, other_preset, Decimal("21500.00"))

        self.assertEqual(FundamentalLabPosition.objects.filter(ticker="WINW").count(), 2)
        self.assertTrue(FundamentalLabPosition.objects.filter(scan_result=second_scan_result).exists())


class SanityCheckThresholdTests(APITestCase):
    """
    Partie 1.3 (2026-08-14) : régression exacte du cas HAIN qui a motivé le
    garde-fou -- ROE -113%, FCF Yield 234%, Dette/EBITDA 7.12x, un verdict
    Claude 'confirmed'/'uncertain' qui aurait ouvert une position sans ce
    garde-fou. Teste la fonction pure d'abord (aucune dépendance DB/pipeline).
    """

    def test_hain_case_triggers_multiple_thresholds(self):
        hain_like_details = {
            "roe": -113.0,
            "fcf_yield": 234.0,
            "debt_to_ebitda": 7.12,
            "pe_ratio": None,
            "thresholds_used": {"debt_ebitda_max": 3},  # DEFAULT sector
        }
        result = check_sanity(hain_like_details)
        self.assertFalse(result.passed)
        # Les 3 seuils indépendants doivent tous se déclencher en même temps
        # sur ce cas précis (pas juste un seul par coïncidence).
        self.assertEqual(len(result.reasons), 3)
        self.assertTrue(any("ROE" in r for r in result.reasons))
        self.assertTrue(any("FCF Yield" in r for r in result.reasons))
        self.assertTrue(any("Dette/EBITDA" in r for r in result.reasons))

    def test_normal_ratios_pass(self):
        normal_details = {
            "roe": 18.0,
            "fcf_yield": 6.0,
            "debt_to_ebitda": 1.2,
            "pe_ratio": 12.0,
            "thresholds_used": {"debt_ebitda_max": 3},
        }
        result = check_sanity(normal_details)
        self.assertTrue(result.passed)
        self.assertEqual(result.reasons, [])

    def test_missing_data_signal_triggers_alone(self):
        details = {
            "roe": -5.0,  # négatif mais pas assez pour ROE_ANOMALY_THRESHOLD seul
            "fcf_yield": None,
            "pe_ratio": None,
            "debt_to_ebitda": None,
            "thresholds_used": {"debt_ebitda_max": 3},
        }
        result = check_sanity(details)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.reasons), 1)
        self.assertIn("données fondamentales probablement non fiables", result.reasons[0])

    def test_debt_ebitda_uses_sector_threshold_not_fixed_value(self):
        # 6x est normal pour Utilities (debt_ebitda_max=6) mais serait un
        # levier extrême pour Technology (debt_ebitda_max=1.5, 2x = 3).
        utilities_details = {
            "roe": 10.0, "fcf_yield": 5.0, "debt_to_ebitda": 6.0, "pe_ratio": 15.0,
            "thresholds_used": {"debt_ebitda_max": 6},
        }
        self.assertTrue(check_sanity(utilities_details).passed)

        tech_details = dict(utilities_details, thresholds_used={"debt_ebitda_max": 1.5})
        result = check_sanity(tech_details)
        self.assertFalse(result.passed)
        self.assertTrue(any("Dette/EBITDA" in r for r in result.reasons))


class CreateLabPositionSanityGuardrailTests(APITestCase):
    """
    Partie 1.2/1.3 : _create_lab_position doit bloquer la création, peu
    importe le verdict LLM déjà stocké sur le ScanResult (confirmed OU
    uncertain) -- reproduit le cas HAIN exact au niveau du point de passage
    unique (avant PaperTrade.objects.create).
    """

    def setUp(self):
        self.preset = _make_preset()
        self.scan_run = ScanRun.objects.create(preset=self.preset, universe_source="sandboxes")

    def _make_hain_like_scan_result(self, verdict):
        return ScanResult.objects.create(
            scan_run=self.scan_run,
            ticker="HAIN",
            sector="Consumer Defensive",
            quant_score=2.0,
            price_at_scan=Decimal("9.50"),
            quant_details={
                "roe": -113.0,
                "fcf_yield": 234.0,
                "debt_to_ebitda": 7.12,
                "pe_ratio": None,
                "thresholds_used": {"debt_ebitda_max": 3},
            },
            claude_verdict=verdict,
            claude_reasoning="Ratios statistiquement suspects mais signal mixte.",
            final_verdict=verdict,
        )

    def test_confirmed_verdict_still_blocked(self):
        scan_result = self._make_hain_like_scan_result("confirmed")

        tasks._create_lab_position(scan_result, self.preset, Decimal("21500.00"))

        scan_result.refresh_from_db()
        self.assertEqual(scan_result.final_verdict, "flagged_anomaly")
        self.assertIn("ROE", scan_result.anomaly_reason)
        # Le verdict LLM d'origine reste tracé, seul final_verdict change.
        self.assertEqual(scan_result.claude_verdict, "confirmed")
        self.assertFalse(FundamentalLabPosition.objects.filter(scan_result=scan_result).exists())
        self.assertEqual(PaperTrade.objects.filter(sandbox="FUNDAMENTAL_LAB").count(), 0)

    def test_uncertain_verdict_still_blocked(self):
        scan_result = self._make_hain_like_scan_result("uncertain")

        tasks._create_lab_position(scan_result, self.preset, Decimal("21500.00"))

        scan_result.refresh_from_db()
        self.assertEqual(scan_result.final_verdict, "flagged_anomaly")
        self.assertFalse(FundamentalLabPosition.objects.filter(scan_result=scan_result).exists())
        self.assertEqual(PaperTrade.objects.filter(sandbox="FUNDAMENTAL_LAB").count(), 0)

    def test_full_pipeline_blocks_hain_case_end_to_end(self):
        """
        Bout en bout via _process_candidate (pas juste _create_lab_position
        directement) -- confirme que le garde-fou s'applique aussi quand
        déclenché depuis le vrai chemin d'exécution du scan.
        """
        quant_result = QuantResult(
            ticker="HAIN", sector="Consumer Defensive", passed=True, score=2.0, price=9.50,
            details={
                "roe": -113.0, "fcf_yield": 234.0, "debt_to_ebitda": 7.12, "pe_ratio": None,
                "sector": "Consumer Defensive", "thresholds_used": {"debt_ebitda_max": 3},
            },
            reasons_failed=[],
        )
        with patch.object(tasks, "triage_ticker") as mock_deepseek, \
                patch.object(tasks, "verify_ticker") as mock_claude, \
                patch.object(tasks, "submit_lab_market_order") as mock_order:
            mock_deepseek.return_value = DeepSeekResult(verdict="uncertain", reasoning="Signal mixte.")
            mock_claude.return_value = ClaudeResult(
                verdict="confirmed", reasoning="Ratios suspects mais je penche pour une inefficience.",
                confidence="low",
            )

            tasks._process_candidate(self.scan_run, self.preset, quant_result, Decimal("21500.00"))

            mock_order.assert_not_called()

        scan_result = ScanResult.objects.get(scan_run=self.scan_run, ticker="HAIN")
        self.assertEqual(scan_result.final_verdict, "flagged_anomaly")
        self.assertEqual(scan_result.claude_verdict, "confirmed")
        self.assertEqual(scan_result.claude_confidence, "low")
        self.assertFalse(FundamentalLabPosition.objects.filter(scan_result=scan_result).exists())


class RejectionOutcomeBenchmarkTests(APITestCase):
    """
    TODO #4 du brief : benchmark_return_pct doit être calculé à partir du
    prix benchmark réellement capturé au moment du rejet
    (RejectionLog.benchmark_price_at_rejection), pas laissé à 0.0.
    """

    def setUp(self):
        self.preset = _make_preset()
        self.scan_run = ScanRun.objects.create(preset=self.preset, universe_source="sandboxes")
        self.scan_result = ScanResult.objects.create(
            scan_run=self.scan_run, ticker="ONCY", final_verdict="rejected_deepseek",
            price_at_scan=Decimal("2.00"),
        )
        self.log = RejectionLog.objects.create(
            scan_result=self.scan_result,
            ticker="ONCY",
            preset=self.preset,
            rejected_by="deepseek",
            reasoning_text="Guidance coupée.",
            price_at_rejection=Decimal("2.00"),
            benchmark_price_at_rejection=Decimal("1000.00"),
        )

    def test_alpha_uses_real_benchmark_return(self):
        with patch.object(tasks.market_data, "Ticker") as MockTicker:
            MockTicker.return_value.info = {"regularMarketPrice": 3.00}  # ticker +50%
            tasks._check_single_outcome(self.log, 30, Decimal("1100.00"))  # benchmark +10%

        outcome = RejectionOutcome.objects.get(rejection_log=self.log, days_since_rejection=30)
        self.assertEqual(outcome.return_pct, 50.0)
        self.assertEqual(outcome.benchmark_return_pct, 10.0)
        self.assertEqual(outcome.alpha, 40.0)
        self.assertTrue(outcome.possible_missed_winner)

    def test_missing_benchmark_price_falls_back_without_crashing(self):
        self.log.benchmark_price_at_rejection = None
        self.log.save(update_fields=["benchmark_price_at_rejection"])

        with patch.object(tasks.market_data, "Ticker") as MockTicker:
            MockTicker.return_value.info = {"regularMarketPrice": 2.10}
            tasks._check_single_outcome(self.log, 30, Decimal("1100.00"))

        outcome = RejectionOutcome.objects.get(rejection_log=self.log, days_since_rejection=30)
        self.assertEqual(outcome.benchmark_return_pct, 0.0)


class ProcessCandidatePipelineTests(APITestCase):
    """Bout en bout des 3 étapes (mockées, aucun appel réseau)."""

    def setUp(self):
        self.preset = _make_preset()
        self.scan_run = ScanRun.objects.create(preset=self.preset, universe_source="sandboxes")

    def test_deepseek_rejection_logs_and_skips_claude(self):
        quant_result = _make_quant_result()
        with patch.object(tasks, "triage_ticker") as mock_deepseek, \
                patch.object(tasks, "verify_ticker") as mock_claude:
            mock_deepseek.return_value = DeepSeekResult(verdict="negative_reason", reasoning="Fraude alléguée.")

            tasks._process_candidate(self.scan_run, self.preset, quant_result, Decimal("21500.00"))

            mock_claude.assert_not_called()

        scan_result = ScanResult.objects.get(scan_run=self.scan_run, ticker="WINW")
        self.assertEqual(scan_result.final_verdict, "rejected_deepseek")
        self.assertFalse(FundamentalLabPosition.objects.filter(scan_result=scan_result).exists())

        rejection = RejectionLog.objects.get(scan_result=scan_result)
        self.assertEqual(rejection.rejected_by, "deepseek")
        self.assertEqual(rejection.benchmark_price_at_rejection, Decimal("21500.00"))

    def test_claude_confirmed_creates_lab_position(self):
        quant_result = _make_quant_result(ticker="ATD")
        fake_order = MagicMock()
        fake_order.id = "fake-order-id-456"
        fake_order.status = "accepted"
        with patch.object(tasks, "triage_ticker") as mock_deepseek, \
                patch.object(tasks, "verify_ticker") as mock_claude, \
                patch.object(tasks, "submit_lab_market_order", return_value=fake_order):
            mock_deepseek.return_value = DeepSeekResult(verdict="no_reason", reasoning="Rien de notable.")
            mock_claude.return_value = ClaudeResult(verdict="confirmed", reasoning="Inefficience de marché.")

            tasks._process_candidate(self.scan_run, self.preset, quant_result, Decimal("21500.00"))

        scan_result = ScanResult.objects.get(scan_run=self.scan_run, ticker="ATD")
        self.assertEqual(scan_result.final_verdict, "confirmed")
        lab_position = FundamentalLabPosition.objects.get(scan_result=scan_result)
        self.assertIsNotNone(lab_position.paper_trade_id)

    def test_quant_only_preset_skips_llm_steps(self):
        preset = _make_preset(slug="quality-compounder", name="Quality Compounder", requires_news_check=False)
        scan_run = ScanRun.objects.create(preset=preset, universe_source="sandboxes")
        quant_result = _make_quant_result(ticker="RY")

        with patch.object(tasks, "triage_ticker") as mock_deepseek, \
                patch.object(tasks, "verify_ticker") as mock_claude:
            tasks._process_candidate(scan_run, preset, quant_result, Decimal("21500.00"))
            mock_deepseek.assert_not_called()
            mock_claude.assert_not_called()

        scan_result = ScanResult.objects.get(scan_run=scan_run, ticker="RY")
        self.assertEqual(scan_result.final_verdict, "confirmed")
        self.assertTrue(FundamentalLabPosition.objects.filter(scan_result=scan_result).exists())


class MonitorLabPositionsCloseFieldsTests(APITestCase):
    """
    Partie 2.2 (2026-08-14) : monitor_lab_positions doit renseigner
    exit_price/exit_date/close_reason/realized_return_pct sur
    FundamentalLabPosition au moment de la fermeture stop-loss -- avant ça
    seul is_open passait à False, obligeant à passer par PaperTrade pour
    retrouver ces informations (et sans raison de fermeture machine-lisible
    du tout). Futur feature/label set pour un modèle ML, voir
    docs/ML_LAB_FUTURE_MODEL.md.
    """

    def setUp(self):
        self.preset = _make_preset()
        self.scan_run = ScanRun.objects.create(preset=self.preset, universe_source="sandboxes")
        self.scan_result = ScanResult.objects.create(
            scan_run=self.scan_run, ticker="ATD", final_verdict="confirmed",
            price_at_scan=Decimal("10.00"), quant_score=3.0,
        )
        self.trade = PaperTrade.objects.create(
            ticker="ATD", sandbox="FUNDAMENTAL_LAB", entry_price=Decimal("10.00"),
            quantity=10, stop_loss=Decimal("9.50"), status="OPEN",
            broker="ALPACA_LAB", broker_order_id="order-abc", broker_status="new",
        )
        self.lab_position = FundamentalLabPosition.objects.create(
            scan_result=self.scan_result, ticker="ATD", preset=self.preset,
            entry_price=Decimal("10.00"), entry_date=timezone.now(),
            paper_trade=self.trade,
        )

    def test_stop_loss_close_populates_lab_position_close_fields(self):
        filled_order = MagicMock()
        filled_order.status = "filled"
        filled_order.filled_qty = "10"
        filled_order.filled_avg_price = "10.00"

        sell_order = MagicMock()
        sell_order.id = "sell-order-xyz"
        sell_order.status = "filled"

        with patch.object(tasks, "get_lab_order_by_id", return_value=filled_order), \
                patch.object(tasks, "fetch_ticker_data", return_value={"price": 9.00}), \
                patch.object(tasks, "submit_lab_market_order", return_value=sell_order):
            tasks.monitor_lab_positions()

        self.lab_position.refresh_from_db()
        self.trade.refresh_from_db()

        self.assertFalse(self.lab_position.is_open)
        self.assertEqual(self.trade.status, "CLOSED")
        self.assertEqual(float(self.lab_position.exit_price), 9.00)
        self.assertIsNotNone(self.lab_position.exit_date)
        self.assertEqual(self.lab_position.close_reason, "stop_loss")
        self.assertAlmostEqual(self.lab_position.realized_return_pct, -10.0, places=2)


class MonitorLabPositionsTakeProfitAndMaxHoldTests(APITestCase):
    """
    2026-08-23 : avant ça, monitor_lab_positions n'avait qu'un stop_loss --
    une position gagnante ne se fermait JAMAIS d'elle-même (aucune prise de
    profit, aucune limite de temps), biaisant structurellement tout
    échantillon de positions fermées vers les pertes. Couvre les deux
    nouvelles conditions de sortie, pour ALPACA_LAB ET SIM (TSX)."""

    def setUp(self):
        self.preset = _make_preset()

    def _make_position(self, ticker, broker, entry_date=None, stop_loss=None, take_profit=None):
        scan_run = ScanRun.objects.create(preset=self.preset, universe_source="sandboxes")
        scan_result = ScanResult.objects.create(
            scan_run=scan_run, ticker=ticker, final_verdict="confirmed",
            price_at_scan=Decimal("10.00"), quant_score=3.0,
        )
        trade = PaperTrade.objects.create(
            ticker=ticker, sandbox="FUNDAMENTAL_LAB", entry_price=Decimal("10.00"),
            quantity=10, stop_loss=stop_loss or Decimal("9.50"), take_profit=take_profit,
            status="OPEN", broker=broker,
            broker_order_id="order-abc" if broker == "ALPACA_LAB" else "",
            broker_status="new" if broker == "ALPACA_LAB" else "",
        )
        if entry_date is not None:
            # entry_date a auto_now_add=True -- ignore toute valeur passée à
            # create()/save(), doit être forcé via .update() (bypass
            # auto_now_add) pour simuler une position ouverte depuis
            # longtemps dans ces tests de max_hold_time.
            PaperTrade.objects.filter(pk=trade.pk).update(entry_date=entry_date)
            trade.refresh_from_db()
        lab_position = FundamentalLabPosition.objects.create(
            scan_result=scan_result, ticker=ticker, preset=self.preset,
            entry_price=Decimal("10.00"), entry_date=entry_date or timezone.now(),
            paper_trade=trade,
        )
        return trade, lab_position

    def test_alpaca_take_profit_closes_with_a_real_sell_order(self):
        trade, lab_position = self._make_position("ATD", "ALPACA_LAB", take_profit=Decimal("12.00"))
        filled_order = MagicMock(status="filled", filled_qty="10", filled_avg_price="10.00")
        sell_order = MagicMock(id="sell-order-xyz", status="filled")

        with patch.object(tasks, "get_lab_order_by_id", return_value=filled_order), \
                patch.object(tasks, "fetch_ticker_data", return_value={"price": 12.50}), \
                patch.object(tasks, "submit_lab_market_order", return_value=sell_order) as mock_sell:
            tasks.monitor_lab_positions()

        mock_sell.assert_called_once()
        lab_position.refresh_from_db()
        trade.refresh_from_db()
        self.assertFalse(lab_position.is_open)
        self.assertEqual(lab_position.close_reason, "take_profit")
        self.assertEqual(trade.outcome, "WIN")
        self.assertAlmostEqual(lab_position.realized_return_pct, 25.0, places=2)

    def test_alpaca_max_hold_time_closes_regardless_of_price(self):
        old_entry = timezone.now() - timedelta(days=61)  # au-delà du défaut de 60j
        trade, lab_position = self._make_position("ATD", "ALPACA_LAB", entry_date=old_entry, take_profit=Decimal("12.00"))
        filled_order = MagicMock(status="filled", filled_qty="10", filled_avg_price="10.00")
        sell_order = MagicMock(id="sell-order-xyz", status="filled")

        with patch.object(tasks, "get_lab_order_by_id", return_value=filled_order), \
                patch.object(tasks, "fetch_ticker_data", return_value={"price": 10.20}), \
                patch.object(tasks, "submit_lab_market_order", return_value=sell_order):
            tasks.monitor_lab_positions()

        lab_position.refresh_from_db()
        self.assertFalse(lab_position.is_open)
        self.assertEqual(lab_position.close_reason, "max_hold_time")

    def test_alpaca_neither_condition_stays_open(self):
        trade, lab_position = self._make_position("ATD", "ALPACA_LAB", take_profit=Decimal("12.00"))
        filled_order = MagicMock(status="filled", filled_qty="10", filled_avg_price="10.00")

        with patch.object(tasks, "get_lab_order_by_id", return_value=filled_order), \
                patch.object(tasks, "fetch_ticker_data", return_value={"price": 10.50}), \
                patch.object(tasks, "submit_lab_market_order") as mock_sell:
            tasks.monitor_lab_positions()

        mock_sell.assert_not_called()
        lab_position.refresh_from_db()
        self.assertTrue(lab_position.is_open)

    def test_sim_tsx_take_profit_closes_without_any_real_order(self):
        # Régression directe : sans ça, TOUTE position TSX (broker='SIM')
        # restait ouverte pour toujours -- ni stop_loss (jamais eu, par
        # design) ni aucune autre sortie n'existait avant ce correctif.
        trade, lab_position = self._make_position("K.TO", "SIM", take_profit=Decimal("12.00"))

        with patch.object(tasks, "fetch_ticker_data", return_value={"price": 13.00}), \
                patch.object(tasks, "submit_lab_market_order") as mock_sell, \
                patch.object(tasks, "get_lab_order_by_id") as mock_get_order:
            tasks.monitor_lab_positions()

        mock_sell.assert_not_called()  # SIM -- jamais d'ordre réel, ni à l'achat ni à la sortie
        mock_get_order.assert_not_called()
        lab_position.refresh_from_db()
        trade.refresh_from_db()
        self.assertFalse(lab_position.is_open)
        self.assertEqual(lab_position.close_reason, "take_profit")
        self.assertEqual(trade.status, "CLOSED")

    def test_sim_tsx_max_hold_time_closes(self):
        old_entry = timezone.now() - timedelta(days=61)
        trade, lab_position = self._make_position("K.TO", "SIM", entry_date=old_entry, take_profit=Decimal("12.00"))

        with patch.object(tasks, "fetch_ticker_data", return_value={"price": 10.10}):
            tasks.monitor_lab_positions()

        lab_position.refresh_from_db()
        self.assertFalse(lab_position.is_open)
        self.assertEqual(lab_position.close_reason, "max_hold_time")

    def test_sim_tsx_stays_open_without_take_profit_or_max_hold(self):
        trade, lab_position = self._make_position("K.TO", "SIM", take_profit=Decimal("12.00"))

        with patch.object(tasks, "fetch_ticker_data", return_value={"price": 10.50}):
            tasks.monitor_lab_positions()

        lab_position.refresh_from_db()
        self.assertTrue(lab_position.is_open)


class SplitAdjustmentTests(APITestCase):
    """
    2026-08-24 : régression directe du cas BRCC -- reverse split 1-pour-10
    pendant qu'une position FUNDAMENTAL_LAB était ouverte. Comparer
    entry_price (avant le split) au prix courant (après) donnait un faux
    mouvement de +1000%, qui a déclenché une VRAIE prise de profit (ordre
    de vente réel exécuté sur le compte paper Alpaca dédié) -- confirmé
    par le relevé de transactions Alpaca lui-même (achats ~0,84$ le
    18 août, vente ~9,18$ le 24 août, exactement le ratio 0.1 du split).
    """

    def setUp(self):
        self.preset = _make_preset()

    def _make_position(self, ticker, broker, entry_date, entry_price, stop_loss, take_profit):
        scan_run = ScanRun.objects.create(preset=self.preset, universe_source="sandboxes")
        scan_result = ScanResult.objects.create(
            scan_run=scan_run, ticker=ticker, final_verdict="uncertain",
            price_at_scan=Decimal(str(entry_price)), quant_score=1.0,
        )
        trade = PaperTrade.objects.create(
            ticker=ticker, sandbox="FUNDAMENTAL_LAB", entry_price=Decimal(str(entry_price)),
            quantity=121, stop_loss=Decimal(str(stop_loss)), take_profit=Decimal(str(take_profit)),
            status="OPEN", broker=broker,
            broker_order_id="order-brcc" if broker == "ALPACA_LAB" else "",
            broker_status="new" if broker == "ALPACA_LAB" else "",
        )
        PaperTrade.objects.filter(pk=trade.pk).update(entry_date=entry_date)
        trade.refresh_from_db()
        lab_position = FundamentalLabPosition.objects.create(
            scan_result=scan_result, ticker=ticker, preset=self.preset,
            entry_price=Decimal(str(entry_price)), entry_date=entry_date, paper_trade=trade,
        )
        return trade, lab_position

    def _brcc_split_series(self, split_date):
        return pd.Series([0.1], index=pd.DatetimeIndex([split_date]))

    def test_split_after_entry_prevents_a_false_take_profit(self):
        entry_date = timezone.now() - timedelta(days=6)  # avant le split simulé
        split_date = timezone.now() - timedelta(hours=10)
        trade, lab_position = self._make_position(
            "BRCC", "ALPACA_LAB", entry_date, entry_price=0.84, stop_loss=0.80, take_profit=1.01,
        )
        filled_order = MagicMock(status="filled", filled_qty="121", filled_avg_price="0.84")

        with patch.object(tasks, "get_lab_order_by_id", return_value=filled_order), \
                patch.object(tasks, "fetch_ticker_data", return_value={"price": 9.26}), \
                patch.object(tasks.market_data, "Ticker") as mock_ticker_cls, \
                patch.object(tasks, "submit_lab_market_order") as mock_sell:
            mock_ticker_cls.return_value.splits = self._brcc_split_series(split_date)
            tasks.monitor_lab_positions()

        mock_sell.assert_not_called()
        trade.refresh_from_db()
        lab_position.refresh_from_db()
        self.assertTrue(lab_position.is_open, "la fausse prise de profit ne doit pas fermer la position")
        # entry_price/take_profit ajustés par le ratio (0.1) pour rester
        # comparables au prix courant post-split.
        self.assertAlmostEqual(float(trade.entry_price), 8.4, places=2)
        self.assertAlmostEqual(float(trade.take_profit), 10.1, places=2)
        self.assertAlmostEqual(float(trade.quantity), 12.1, places=2)

    def test_split_before_entry_is_ignored(self):
        entry_date = timezone.now() - timedelta(hours=2)
        split_date = timezone.now() - timedelta(days=10)  # bien avant l'entrée -- non pertinent
        trade, lab_position = self._make_position(
            "BRCC", "ALPACA_LAB", entry_date, entry_price=8.4, stop_loss=8.0, take_profit=10.1,
        )
        filled_order = MagicMock(status="filled", filled_qty="12", filled_avg_price="8.4")

        with patch.object(tasks, "get_lab_order_by_id", return_value=filled_order), \
                patch.object(tasks, "fetch_ticker_data", return_value={"price": 9.0}), \
                patch.object(tasks.market_data, "Ticker") as mock_ticker_cls, \
                patch.object(tasks, "submit_lab_market_order") as mock_sell:
            mock_ticker_cls.return_value.splits = self._brcc_split_series(split_date)
            tasks.monitor_lab_positions()

        mock_sell.assert_not_called()
        trade.refresh_from_db()
        self.assertEqual(float(trade.entry_price), 8.4)  # inchangé -- le split est hors fenêtre

    def test_no_splits_at_all_leaves_fields_untouched(self):
        entry_date = timezone.now() - timedelta(days=1)
        trade, lab_position = self._make_position(
            "GOOD", "SIM", entry_date, entry_price=10.0, stop_loss=9.0, take_profit=12.0,
        )

        with patch.object(tasks, "fetch_ticker_data", return_value={"price": 10.5}), \
                patch.object(tasks.market_data, "Ticker") as mock_ticker_cls:
            mock_ticker_cls.return_value.splits = pd.Series(dtype="float64")
            tasks.monitor_lab_positions()

        trade.refresh_from_db()
        self.assertEqual(float(trade.entry_price), 10.0)

    def test_real_take_profit_after_split_adjustment_still_closes(self):
        # Le titre continue de monter APRÈS le split, dépassant vraiment le
        # take_profit ajusté (10.1$) -- doit se fermer normalement, la
        # protection contre les splits ne doit pas bloquer une vraie sortie.
        entry_date = timezone.now() - timedelta(days=6)
        split_date = timezone.now() - timedelta(hours=10)
        trade, lab_position = self._make_position(
            "BRCC", "SIM", entry_date, entry_price=0.84, stop_loss=0.80, take_profit=1.01,
        )

        with patch.object(tasks, "fetch_ticker_data", return_value={"price": 11.0}), \
                patch.object(tasks.market_data, "Ticker") as mock_ticker_cls:
            mock_ticker_cls.return_value.splits = self._brcc_split_series(split_date)
            tasks.monitor_lab_positions()

        lab_position.refresh_from_db()
        self.assertFalse(lab_position.is_open)
        self.assertEqual(lab_position.close_reason, "take_profit")


class SyncWatchlistFromFundamentalLabTests(APITestCase):
    """
    2026-08-23 : remplace, pour WATCHLIST, l'ancien mécanisme d'ajout
    (fusion du scanner penny 0,50$-10$) -- un décalage prix/style trouvé
    en discutant pourquoi WATCHLIST ne contenait que des titres sous 10$
    alors qu'il partage son modèle (FUSION) avec AI_BLUECHIP."""

    def setUp(self):
        self.preset = _make_preset()

    def _make_confirmed_position(self, ticker, is_open=True):
        scan_run = ScanRun.objects.create(preset=self.preset, universe_source="sandboxes")
        scan_result = ScanResult.objects.create(
            scan_run=scan_run, ticker=ticker, final_verdict="confirmed",
            price_at_scan=Decimal("10.00"), quant_score=3.0,
        )
        return FundamentalLabPosition.objects.create(
            scan_result=scan_result, ticker=ticker, preset=self.preset,
            entry_price=Decimal("10.00"), entry_date=timezone.now(), is_open=is_open,
        )

    def test_adds_confirmed_open_tickers_to_an_empty_watchlist(self):
        self._make_confirmed_position("ADBE")
        self._make_confirmed_position("K.TO")

        result = tasks.sync_watchlist_from_fundamental_lab("WATCHLIST")

        watch = SandboxWatchlist.objects.get(sandbox="WATCHLIST")
        self.assertEqual(set(watch.symbols), {"ADBE", "K.TO"})
        self.assertEqual(watch.source, "fundamental_lab_confirmed")
        self.assertEqual(set(result["added"]), {"ADBE", "K.TO"})

    def test_never_removes_existing_symbols_only_adds(self):
        SandboxWatchlist.objects.create(sandbox="WATCHLIST", symbols=["EXISTING"], source="manual")
        self._make_confirmed_position("ADBE")

        tasks.sync_watchlist_from_fundamental_lab("WATCHLIST")

        watch = SandboxWatchlist.objects.get(sandbox="WATCHLIST")
        self.assertIn("EXISTING", watch.symbols)
        self.assertIn("ADBE", watch.symbols)

    def test_closed_confirmed_positions_are_ignored(self):
        self._make_confirmed_position("ADBE", is_open=False)

        result = tasks.sync_watchlist_from_fundamental_lab("WATCHLIST")

        self.assertEqual(result["status"], "no_data")
        self.assertFalse(SandboxWatchlist.objects.filter(sandbox="WATCHLIST").exists())

    def test_uncertain_verdict_is_never_synced_only_confirmed(self):
        scan_run = ScanRun.objects.create(preset=self.preset, universe_source="sandboxes")
        scan_result = ScanResult.objects.create(
            scan_run=scan_run, ticker="BRCC", final_verdict="uncertain",
            price_at_scan=Decimal("0.80"), quant_score=1.0,
        )
        FundamentalLabPosition.objects.create(
            scan_result=scan_result, ticker="BRCC", preset=self.preset,
            entry_price=Decimal("0.80"), entry_date=timezone.now(), is_open=True,
        )

        result = tasks.sync_watchlist_from_fundamental_lab("WATCHLIST")
        self.assertEqual(result["status"], "no_data")

    def test_respects_the_room_limit(self):
        SandboxWatchlist.objects.create(sandbox="WATCHLIST", symbols=[], source="manual")
        self._make_confirmed_position("ADBE")
        self._make_confirmed_position("K.TO")

        real_getenv = tasks.os.getenv
        with patch.object(
            tasks.os, "getenv",
            side_effect=lambda k, d=None: "1" if k == "WATCHLIST_FROM_LAB_LIMIT" else real_getenv(k, d),
        ):
            result = tasks.sync_watchlist_from_fundamental_lab("WATCHLIST")

        self.assertEqual(len(result["added"]), 1)

    def test_duplicate_ticker_across_multiple_positions_added_once(self):
        self._make_confirmed_position("ADBE")
        # Deuxième position confirmée sur le même ticker (ex. reconfirmée
        # par un autre preset) -- ne doit pas être ajoutée deux fois.
        scan_run2 = ScanRun.objects.create(preset=self.preset, universe_source="sandboxes")
        scan_result2 = ScanResult.objects.create(
            scan_run=scan_run2, ticker="ADBE", final_verdict="confirmed",
            price_at_scan=Decimal("11.00"), quant_score=2.0,
        )
        FundamentalLabPosition.objects.create(
            scan_result=scan_result2, ticker="ADBE", preset=self.preset,
            entry_price=Decimal("11.00"), entry_date=timezone.now(), is_open=True,
        )

        result = tasks.sync_watchlist_from_fundamental_lab("WATCHLIST")
        self.assertEqual(result["added"], ["ADBE"])


class LabPerformanceViewTests(APITestCase):
    """
    Confirmed live 2026-08-15 (session investigating a user-reported
    duplicate-looking ticker list on /lab): LabPerformanceView counted and
    averaged in 9 dead rows left over from the item 8 incident
    (TECH_DEBT_NOTES.md, 2026-08-10) -- closed directly in DB with
    is_open=False but no exit_price/exit_date/realized_return_pct, since
    they were duplicate-scan cleanup, never a real trade outcome. Also,
    current_return_pct recomputed against *today's* live price even for
    genuinely closed positions, so a position closed weeks ago silently
    reported "what it would be worth if you'd kept holding it" instead of
    what it actually realized at exit.
    """

    def setUp(self):
        self.preset = _make_preset()
        self.scan_run = ScanRun.objects.create(preset=self.preset, universe_source="sandboxes")

        # Dead duplicate-cleanup row: is_open=False, no exit_date -- must
        # never be counted or averaged.
        dud_result = ScanResult.objects.create(
            scan_run=self.scan_run, ticker="DUPX", final_verdict="confirmed",
            price_at_scan=Decimal("10.00"), quant_score=3.0,
        )
        FundamentalLabPosition.objects.create(
            scan_result=dud_result, ticker="DUPX", preset=self.preset,
            entry_price=Decimal("10.00"), entry_date=timezone.now(),
            is_open=False,
        )

        # Genuinely closed position: +20% realized at exit, but the live
        # market has since moved to a price that would compute a very
        # different return -- current_return_pct must reflect the +20%
        # realized outcome, not a fresh live-price recomputation.
        closed_result = ScanResult.objects.create(
            scan_run=self.scan_run, ticker="REAL", final_verdict="confirmed",
            price_at_scan=Decimal("10.00"), quant_score=3.0,
        )
        self.closed_position = FundamentalLabPosition.objects.create(
            scan_result=closed_result, ticker="REAL", preset=self.preset,
            entry_price=Decimal("10.00"), entry_date=timezone.now(),
            is_open=False, exit_price=Decimal("12.00"), exit_date=timezone.now(),
            close_reason="signal", realized_return_pct=20.0,
        )

        # Still-open position: current_return_pct must keep using the live
        # price (unchanged behavior).
        open_result = ScanResult.objects.create(
            scan_run=self.scan_run, ticker="OPEN", final_verdict="uncertain",
            price_at_scan=Decimal("10.00"), quant_score=3.0,
        )
        self.open_position = FundamentalLabPosition.objects.create(
            scan_result=open_result, ticker="OPEN", preset=self.preset,
            entry_price=Decimal("10.00"), entry_date=timezone.now(), is_open=True,
        )

    def test_dead_duplicate_rows_excluded_from_positions_and_counts(self):
        # LabPerformanceView imports fetch_ticker_data locally from
        # quant_filter (not via `tasks`) -- patch the source it actually
        # resolves against at call time.
        with patch.object(quant_filter, "fetch_ticker_data", return_value={"price": 11.0}):
            response = self.client.get("/api/analysis/lab/performance/")

        self.assertEqual(response.status_code, 200)
        tickers = {p["ticker"] for p in response.data["positions"]}
        self.assertNotIn("DUPX", tickers, "dead duplicate-cleanup row leaked into the positions list")
        self.assertEqual(tickers, {"REAL", "OPEN"})
        # DUPX had final_verdict='confirmed' just like REAL -- if it leaked
        # through, count would be 2 instead of 1.
        self.assertEqual(response.data["confirmed"]["count"], 1)

    def test_closed_position_uses_realized_return_not_live_price(self):
        # Live price implies -50% ((5-10)/10) -- must NOT show up anywhere;
        # the closed position's realized +20% must win instead.
        with patch.object(quant_filter, "fetch_ticker_data", return_value={"price": 5.0}):
            response = self.client.get("/api/analysis/lab/performance/")

        closed = next(p for p in response.data["positions"] if p["ticker"] == "REAL")
        self.assertEqual(closed["current_return_pct"], 20.0)
        self.assertEqual(response.data["confirmed"]["avg_return_pct"], 20.0)

    def test_open_position_still_uses_live_price(self):
        with patch.object(quant_filter, "fetch_ticker_data", return_value={"price": 11.0}):
            response = self.client.get("/api/analysis/lab/performance/")

        opened = next(p for p in response.data["positions"] if p["ticker"] == "OPEN")
        self.assertEqual(opened["current_return_pct"], 10.0)

    def test_same_ticker_across_multiple_positions_fetched_only_once(self):
        # Régression (2026-08-24) : FundamentalLabPosition.Meta.ordering =
        # ['-entry_date'] s'applique implicitement -- combiné à .distinct()
        # sur un values_list('ticker'), ça ne dédupliquait pas réellement
        # (même piège Django que sync_watchlist_from_fundamental_lab le
        # même jour). Confirmé en direct : 33 "tickers distincts" pour
        # ~18 titres réels -- jusqu'à 2x plus d'appels réseau que
        # nécessaire, contribuant au dépassement de délai du frontend.
        second_result = ScanResult.objects.create(
            scan_run=self.scan_run, ticker="OPEN", final_verdict="uncertain",
            price_at_scan=Decimal("9.00"), quant_score=2.0,
        )
        FundamentalLabPosition.objects.create(
            scan_result=second_result, ticker="OPEN", preset=self.preset,
            entry_price=Decimal("9.00"), entry_date=timezone.now(), is_open=True,
        )

        with patch.object(quant_filter, "fetch_ticker_data", return_value={"price": 11.0}) as mock_fetch:
            response = self.client.get("/api/analysis/lab/performance/")

        self.assertEqual(response.status_code, 200)
        open_calls = [c for c in mock_fetch.call_args_list if c.args == ("OPEN",)]
        self.assertEqual(len(open_calls), 1, "OPEN a été fetché plus d'une fois malgré la déduplication")


class GenerateLabWeeklySuggestionsTests(APITestCase):
    """
    Partie 3 (2026-08-14) : generate_lab_weekly_suggestions doit rester
    silencieux (aucun crash) sur un volume de données faible ou nul --
    scénario le plus probable dans les premières semaines de ce garde-fou --
    et ne détecter un pattern que si LAB_SUGGESTION_MIN_OCCURRENCES
    positions au moins partagent le même facteur ET que l'écart de
    rendement dépasse LAB_SUGGESTION_DEVIATION_THRESHOLD_PCT.
    """

    def setUp(self):
        self.preset = _make_preset()
        self.scan_run = ScanRun.objects.create(preset=self.preset, universe_source="sandboxes")

    def _make_closed_position(self, ticker, realized_return_pct, sector="Technology", claude_confidence="high", roe=15.0, days_ago=2):
        scan_result = ScanResult.objects.create(
            scan_run=self.scan_run, ticker=ticker, sector=sector,
            price_at_scan=Decimal("10.00"), quant_score=3.0,
            final_verdict="confirmed", claude_verdict="confirmed", claude_confidence=claude_confidence,
            quant_details={"roe": roe, "fcf_yield": 5.0, "sector": sector},
        )
        exit_date = timezone.now() - timedelta(days=days_ago)
        return FundamentalLabPosition.objects.create(
            scan_result=scan_result, ticker=ticker, preset=self.preset,
            entry_price=Decimal("10.00"), entry_date=exit_date - timedelta(days=3),
            is_open=False, exit_date=exit_date,
            exit_price=Decimal(str(10.0 * (1 + realized_return_pct / 100))),
            close_reason="stop_loss", realized_return_pct=realized_return_pct,
        )

    def test_no_closed_positions_does_not_crash(self):
        result = tasks.generate_lab_weekly_suggestions()

        self.assertEqual(result["status"], "no_data")
        suggestion = LabWeeklySuggestion.objects.get()
        self.assertFalse(suggestion.has_pattern)
        self.assertEqual(suggestion.positions_closed_analyzed, 0)

    def test_few_closed_positions_below_min_occurrences_no_pattern(self):
        # Seulement 2 positions dans le même bucket "Confiance Claude=low" --
        # sous LAB_SUGGESTION_MIN_OCCURRENCES (5), aucun pattern signalé
        # même si l'écart de rendement est énorme.
        self._make_closed_position("A", -50.0, claude_confidence="low")
        self._make_closed_position("B", -45.0, claude_confidence="low")
        self._make_closed_position("C", 5.0, claude_confidence="high")

        result = tasks.generate_lab_weekly_suggestions()

        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["has_pattern"])
        suggestion = LabWeeklySuggestion.objects.get()
        self.assertFalse(suggestion.has_pattern)
        self.assertEqual(suggestion.positions_closed_analyzed, 3)

    def test_clear_pattern_detected_above_min_occurrences_and_deviation(self):
        # 5 positions "Confiance Claude=low" qui sous-performent nettement
        # (>= LAB_SUGGESTION_DEVIATION_THRESHOLD_PCT sous la moyenne globale)
        # -- doit être détecté et résumé.
        for i in range(5):
            self._make_closed_position(f"LOW{i}", -40.0, claude_confidence="low")
        for i in range(3):
            self._make_closed_position(f"HIGH{i}", 10.0, claude_confidence="high")

        result = tasks.generate_lab_weekly_suggestions()

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["has_pattern"])
        suggestion = LabWeeklySuggestion.objects.get()
        self.assertTrue(suggestion.has_pattern)
        self.assertIn("Confiance Claude=low", suggestion.summary)
        self.assertEqual(suggestion.positions_closed_analyzed, 8)

    def test_positions_outside_the_week_window_are_ignored(self):
        self._make_closed_position("OLD", -80.0, days_ago=30)

        result = tasks.generate_lab_weekly_suggestions()

        self.assertEqual(result["status"], "no_data")
        self.assertEqual(result["positions_analyzed"], 0)

    def test_never_writes_outside_labweeklysuggestion_table(self):
        """
        Garde-fou explicite du brief : cette tâche ne doit JAMAIS modifier
        un seuil ou un comportement automatiquement -- vérifie qu'aucun
        ScreenerPreset (là où vivent les seuils réels) n'est touché.
        """
        for i in range(6):
            self._make_closed_position(f"LOW{i}", -40.0, claude_confidence="low")
        before = ScreenerPreset.objects.get(pk=self.preset.pk)
        before_thresholds = dict(before.thresholds)

        tasks.generate_lab_weekly_suggestions()

        after = ScreenerPreset.objects.get(pk=self.preset.pk)
        self.assertEqual(dict(after.thresholds), before_thresholds)


class ScreenerApiTests(APITestCase):
    """
    TODO #3 du brief : ScanResultSerializer doit exposer lab_position_id
    (via SerializerMethodField), dont AnalysisScreener.jsx a besoin pour
    le bouton "Je crois en celle-là " (handleMarkManual).
    """

    def setUp(self):
        self.preset = _make_preset()
        self.scan_run = ScanRun.objects.create(
            preset=self.preset, universe_source="sandboxes", status="completed",
            finished_at=timezone.now(), tickers_scanned=5, candidates_after_quant_filter=2,
        )
        self.confirmed_result = ScanResult.objects.create(
            scan_run=self.scan_run, ticker="WINW", final_verdict="confirmed",
            price_at_scan=Decimal("12.50"), quant_score=3.5,
        )
        self.lab_position = FundamentalLabPosition.objects.create(
            scan_result=self.confirmed_result, ticker="WINW", preset=self.preset,
            entry_price=Decimal("12.50"), entry_date=timezone.now(),
        )
        self.rejected_result = ScanResult.objects.create(
            scan_run=self.scan_run, ticker="XYZ", final_verdict="rejected_claude",
            price_at_scan=Decimal("4.00"), quant_score=1.0,
        )

    def test_results_endpoint_excludes_rejections_and_exposes_lab_position_id(self):
        response = self.client.get(f"/api/analysis/screener/{self.preset.slug}/results/")
        self.assertEqual(response.status_code, 200)
        tickers = [r["ticker"] for r in response.data["results"]]
        self.assertIn("WINW", tickers)
        self.assertNotIn("XYZ", tickers)

        confirmed = next(r for r in response.data["results"] if r["ticker"] == "WINW")
        self.assertEqual(confirmed["lab_position_id"], self.lab_position.id)

    def test_mark_manual_endpoint(self):
        response = self.client.post(
            f"/api/analysis/lab/{self.lab_position.id}/mark-manual/", {"note": "J'y crois"}, format="json",
        )
        self.assertEqual(response.status_code, 200)
        self.lab_position.refresh_from_db()
        self.assertTrue(self.lab_position.manually_selected)
        self.assertEqual(self.lab_position.manual_note, "J'y crois")

    def test_mark_manual_endpoint_keeps_existing_note_when_none_provided(self):
        # Régression : un ré-appel sans "note" (ou avec une note vide/blanche)
        # écrasait silencieusement une note de traçabilité déjà posée.
        self.lab_position.manual_note = "Note de nettoyage à ne pas perdre"
        self.lab_position.save(update_fields=["manual_note"])

        response = self.client.post(
            f"/api/analysis/lab/{self.lab_position.id}/mark-manual/", {}, format="json",
        )
        self.assertEqual(response.status_code, 200)
        self.lab_position.refresh_from_db()
        self.assertTrue(self.lab_position.manually_selected)
        self.assertEqual(self.lab_position.manual_note, "Note de nettoyage à ne pas perdre")

        # Une note blanche (espaces seulement) ne doit pas non plus écraser.
        response = self.client.post(
            f"/api/analysis/lab/{self.lab_position.id}/mark-manual/", {"note": "   "}, format="json",
        )
        self.assertEqual(response.status_code, 200)
        self.lab_position.refresh_from_db()
        self.assertEqual(self.lab_position.manual_note, "Note de nettoyage à ne pas perdre")

    def test_mark_manual_unknown_position_404(self):
        response = self.client.post("/api/analysis/lab/999999/mark-manual/", {}, format="json")
        self.assertEqual(response.status_code, 404)

    def test_run_scan_rejects_inactive_preset(self):
        inactive = _make_preset(slug="value-catalyst", name="Value + Catalyst", is_active=False)
        response = self.client.post(f"/api/analysis/screener/{inactive.slug}/run/")
        self.assertEqual(response.status_code, 404)


class QuantFilterMarginTrendTests(APITestCase):
    """
    Régression trouvée lors d'un scan à blanc (avant push) : pandas .loc[...]
    renvoie des scalaires numpy (float64), donc les comparaisons chaînées
    dans _detect_margin_inflection produisaient un numpy.bool_ plutôt qu'un
    bool Python — invisible dans les tests mockés au niveau réseau, mais
    ScanResult.quant_details (JSONField) refusait de le sérialiser dès qu'un
    vrai ticker avec quarterly_financials passait par evaluate_ticker().
    """

    def _quarterly_financials_with_inflection(self):
        # yfinance ordonne les colonnes du plus récent au plus ancien ; le code
        # les relit dans cet ordre puis les inverse pour obtenir la chronologie.
        # Colonnes Q1(récent)..Q4(ancien) avec marges 42%,32%,30%,40% -> une fois
        # inversé (chronologique) : [40,30,32,42] -> creux à 30 puis 32 puis 42,
        # ce qui déclenche bien _detect_margin_inflection (et donc le chemin
        # numpy.bool_ qu'on veut exercer).
        return pd.DataFrame(
            {
                "Q1": [1000.0, 420.0],
                "Q2": [1000.0, 320.0],
                "Q3": [1000.0, 300.0],
                "Q4": [1000.0, 400.0],
            },
            index=["Total Revenue", "Gross Profit"],
        )

    def test_margin_trend_values_are_native_python_floats(self):
        with patch.object(quant_filter.yf, "Ticker") as MockYfTicker:
            MockYfTicker.return_value.quarterly_financials = self._quarterly_financials_with_inflection()
            trend = quant_filter._fetch_margin_trend("AAPL")

        self.assertTrue(len(trend) > 0)
        for value in trend:
            self.assertIs(type(value), float)  # pas numpy.float64

    def test_margin_trend_skips_nan_gross_profit_instead_of_producing_nan(self):
        """
        Confirmé en direct le 2026-08-17 : OGC.TO (nouveau ticker via
        universe_source='combined') a un Gross Profit NaN sur un trimestre --
        `if rev and gp` seul ne le filtre pas (bool(float('nan')) est True en
        Python), donc un NaN atterrissait dans margin_trend puis faisait
        planter ScanResult.objects.create() (Postgres jsonb rejette le token
        NaN, invalide en JSON strict). A fait échouer 2 scans réels le jour
        même de l'activation de l'univers combiné.
        """
        financials_with_nan_quarter = pd.DataFrame(
            {
                "Q1": [1000.0, 420.0],
                "Q2": [1000.0, float("nan")],  # trimestre avec Gross Profit manquant
                "Q3": [1000.0, 300.0],
                "Q4": [1000.0, 400.0],
            },
            index=["Total Revenue", "Gross Profit"],
        )
        with patch.object(quant_filter.yf, "Ticker") as MockYfTicker:
            MockYfTicker.return_value.quarterly_financials = financials_with_nan_quarter
            trend = quant_filter._fetch_margin_trend("OGC.TO")

        self.assertEqual(len(trend), 3)  # le trimestre NaN est sauté, pas les 4
        for value in trend:
            self.assertFalse(value != value, f"NaN trouvé dans margin_trend: {trend}")
        json.dumps(trend)  # doit rester sérialisable en JSON strict (Postgres jsonb)

    def test_evaluate_ticker_details_are_json_serializable(self):
        with patch.object(quant_filter, "market_data") as mock_market_data, \
                patch.object(quant_filter.yf, "Ticker") as mock_yf_ticker:
            mock_market_data.Ticker.return_value.info = {
                "regularMarketPrice": 12.5,
                "sector": "Technology",
                "trailingPE": 8.2,
                "returnOnEquity": 0.18,
                "totalDebt": 100.0,
                "ebitda": 200.0,
                "numberOfAnalystOpinions": 3,
            }
            mock_yf_ticker.return_value.quarterly_financials = self._quarterly_financials_with_inflection()

            result = quant_filter.evaluate_ticker("AAPL", preset_thresholds={})

        self.assertIsNotNone(result)
        self.assertTrue(result.details.get("margin_inflection_detected"))
        # Ne doit jamais planter — c'est exactement ce que ScanResult.objects.create()
        # fait via le JSONField quant_details.
        json.dumps(result.details)


class QuantFilterValueCatalystTests(APITestCase):
    """
    Preset "Value + Catalyst" (2026-08-16) : thresholds['require_catalyst']
    ajoute un garde-fou positif (inflexion de marge OU achat d'initiés
    significatif requis) au-dessus du filtre quantitatif standard, sans
    changer le comportement de "Undervalued Without Reason" (thresholds={},
    donc require_catalyst absent).
    """

    def _mock_info(self, **overrides):
        base = {
            "regularMarketPrice": 12.5,
            "sector": "Technology",
            "trailingPE": 8.2,
            "returnOnEquity": 0.18,
            "totalDebt": 100.0,
            "ebitda": 200.0,
            "numberOfAnalystOpinions": 3,
        }
        base.update(overrides)
        return base

    def _flat_margin_trend(self):
        # Pas d'inflexion : marges constantes.
        return pd.DataFrame(
            {"Q1": [1000.0, 350.0], "Q2": [1000.0, 350.0], "Q3": [1000.0, 350.0], "Q4": [1000.0, 350.0]},
            index=["Total Revenue", "Gross Profit"],
        )

    def test_require_catalyst_rejects_when_no_margin_inflection_and_no_insider_buying(self):
        with patch.object(quant_filter, "market_data") as mock_market_data, \
                patch.object(quant_filter.yf, "Ticker") as mock_yf_ticker, \
                patch.object(quant_filter, "_fetch_insider_buying", return_value={"insiders_buying": False, "insider_buy_total": 0.0}) as mock_insider:
            mock_market_data.Ticker.return_value.info = self._mock_info()
            mock_yf_ticker.return_value.quarterly_financials = self._flat_margin_trend()

            result = quant_filter.evaluate_ticker("AAPL", preset_thresholds={"require_catalyst": True})

        mock_insider.assert_called_once_with("AAPL")
        self.assertFalse(result.passed)
        self.assertTrue(any("catalyseur" in r for r in result.reasons_failed))

    def test_require_catalyst_passes_on_insider_buying_alone(self):
        with patch.object(quant_filter, "market_data") as mock_market_data, \
                patch.object(quant_filter.yf, "Ticker") as mock_yf_ticker, \
                patch.object(quant_filter, "_fetch_insider_buying", return_value={"insiders_buying": True, "insider_buy_total": 75000.0}):
            mock_market_data.Ticker.return_value.info = self._mock_info()
            mock_yf_ticker.return_value.quarterly_financials = self._flat_margin_trend()

            result = quant_filter.evaluate_ticker("AAPL", preset_thresholds={"require_catalyst": True})

        self.assertTrue(result.passed)
        self.assertTrue(result.details.get("insiders_buying"))
        self.assertEqual(result.details.get("insider_buy_total"), 75000.0)
        json.dumps(result.details)  # JSONField-serializable, même contrainte que le test ci-dessus

    def test_require_catalyst_passes_on_margin_inflection_alone(self):
        with patch.object(quant_filter, "market_data") as mock_market_data, \
                patch.object(quant_filter.yf, "Ticker") as mock_yf_ticker, \
                patch.object(quant_filter, "_fetch_insider_buying", return_value={"insiders_buying": False, "insider_buy_total": 0.0}):
            mock_market_data.Ticker.return_value.info = self._mock_info()
            mock_yf_ticker.return_value.quarterly_financials = self._quarterly_financials_with_inflection_for_catalyst()

            result = quant_filter.evaluate_ticker("AAPL", preset_thresholds={"require_catalyst": True})

        self.assertTrue(result.passed)
        self.assertTrue(result.details.get("margin_inflection_detected"))

    def _quarterly_financials_with_inflection_for_catalyst(self):
        return pd.DataFrame(
            {"Q1": [1000.0, 420.0], "Q2": [1000.0, 320.0], "Q3": [1000.0, 300.0], "Q4": [1000.0, 400.0]},
            index=["Total Revenue", "Gross Profit"],
        )

    def test_default_preset_never_calls_insider_fetch(self):
        """thresholds={} (Undervalued Without Reason) -- require_catalyst absent,
        aucun appel réseau supplémentaire, comportement inchangé pour ce preset."""
        with patch.object(quant_filter, "market_data") as mock_market_data, \
                patch.object(quant_filter.yf, "Ticker") as mock_yf_ticker, \
                patch.object(quant_filter, "_fetch_insider_buying") as mock_insider:
            mock_market_data.Ticker.return_value.info = self._mock_info()
            mock_yf_ticker.return_value.quarterly_financials = self._flat_margin_trend()

            result = quant_filter.evaluate_ticker("AAPL", preset_thresholds={})

        mock_insider.assert_not_called()
        self.assertTrue(result.passed)
        self.assertNotIn("insiders_buying", result.details)


def _make_ohlc_frame(rows: int = 30) -> pd.DataFrame:
    """OHLC + volume factices avec un DatetimeIndex nommé 'Date', pour
    simuler ce que market_data.Ticker(...).history() renvoie réellement."""
    dates = pd.date_range("2026-01-01", periods=rows, freq="D")
    base = 10.0
    opens, highs, lows, closes, volumes = [], [], [], [], []
    for i in range(rows):
        o = base + i * 0.05
        c = o + (0.3 if i % 3 == 0 else -0.1)
        h = max(o, c) + 0.2
        low = min(o, c) - 0.2
        opens.append(o)
        highs.append(h)
        lows.append(low)
        closes.append(c)
        volumes.append(100000 + i * 1000)
    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=pd.DatetimeIndex(dates, name="Date"),
    )
    return df


class TickerChartViewTests(APITestCase):
    """
    Régression : deux bugs trouvés en testant l'endpoint en direct contre le
    NAS avant d'écrire le frontend (2026-08-10), pas en local seulement.
    """

    def test_handles_multiindex_columns_from_market_data(self):
        """
        market_data.Ticker(...).history() renvoie parfois des colonnes en
        MultiIndex (ex. ('BTO.TO', 'Open')) plutôt que 'Open' simple --
        confirmé en direct. Sans l'aplatissement, l'endpoint renvoyait
        silencieusement candles=[] pour un ticker pourtant valide.
        """
        frame = _make_ohlc_frame(30)
        frame.columns = pd.MultiIndex.from_product([["BTO.TO"], frame.columns])

        with patch("portfolio.market_data.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = frame
            response = self.client.get("/api/analysis/ticker/BTO.TO/chart/?period=3mo")

        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(response.data["candles"]), 0)
        self.assertGreater(len(response.data["ema20"]), 0)

    def test_response_is_valid_json_despite_nan_rvol(self):
        """
        rvol est NaN pour les toutes premières bougies d'une fenêtre (pas
        assez d'historique pour sa moyenne mobile) -- confirmé en direct :
        ça faisait planter le rendu JSON de DRF avec "Out of range float
        values are not JSON compliant: nan" avant le fix.
        """
        frame = _make_ohlc_frame(30)  # colonnes simples, pas de MultiIndex ici

        with patch("portfolio.market_data.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = frame
            response = self.client.get("/api/analysis/ticker/BTO.TO/chart/?period=3mo")

        self.assertEqual(response.status_code, 200)
        # response.data (avant rendu) peut encore contenir NaN -- le vrai
        # test est que le RENDU json ne plante pas, comme un vrai client HTTP.
        rendered = response.render()
        json.loads(rendered.content)
        patterns = response.data.get("patterns", [])
        for p in patterns:
            if p.get("rvol") is not None:
                self.assertFalse(pd.isna(p["rvol"]))

    def test_invalid_ticker_returns_empty_shape_not_error(self):
        with patch("portfolio.market_data.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = pd.DataFrame()
            response = self.client.get("/api/analysis/ticker/ZZZINVALIDXYZ/chart/?period=3mo")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["candles"], [])
        self.assertEqual(response.data["patterns"], [])

    def test_unknown_period_falls_back_to_3mo(self):
        frame = _make_ohlc_frame(10)
        with patch("portfolio.market_data.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = frame
            response = self.client.get("/api/analysis/ticker/BTO.TO/chart/?period=garbage")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["period"], "3mo")


class RunManualCheckTests(APITestCase):
    """
    services.manual_check.run_manual_check : miroir en lecture seule de
    _process_candidate (mêmes fonctions réutilisées), mais ne doit RIEN
    écrire en DB -- c'est ManualTickerCheckView qui se charge de la trace
    optionnelle, pas cette fonction.
    """

    def test_insufficient_data_returns_error_dict_not_exception(self):
        with patch.object(manual_check, "evaluate_ticker", return_value=None):
            result = manual_check.run_manual_check("ZZZINVALIDXYZ")
        self.assertIn("error", result)
        self.assertEqual(result["ticker"], "ZZZINVALIDXYZ")

    def test_deepseek_rejection_skips_claude(self):
        quant_result = _make_quant_result(ticker="DAR")
        with patch.object(manual_check, "evaluate_ticker", return_value=quant_result), \
                patch.object(manual_check, "triage_ticker") as mock_deepseek, \
                patch.object(manual_check, "verify_ticker") as mock_claude:
            mock_deepseek.return_value = DeepSeekResult(verdict="negative_reason", reasoning="Fraude alléguée.")

            result = manual_check.run_manual_check("dar")  # minuscule -> doit être normalisé

            mock_claude.assert_not_called()

        self.assertEqual(result["ticker"], "DAR")
        self.assertEqual(result["final_verdict"], "rejected_deepseek")
        self.assertEqual(result["claude_verdict"], "")

    def test_claude_confirmed_verdict_propagates(self):
        quant_result = _make_quant_result(ticker="DAR")
        with patch.object(manual_check, "evaluate_ticker", return_value=quant_result), \
                patch.object(manual_check, "triage_ticker") as mock_deepseek, \
                patch.object(manual_check, "verify_ticker") as mock_claude:
            mock_deepseek.return_value = DeepSeekResult(verdict="no_reason", reasoning="Rien de notable.")
            mock_claude.return_value = ClaudeResult(verdict="confirmed", reasoning="Bilan solide.", tokens_used=1500)

            result = manual_check.run_manual_check("DAR")

        self.assertEqual(result["final_verdict"], "confirmed")
        self.assertEqual(result["claude_tokens_used"], 1500)

    def test_claude_rejected_maps_to_rejected_claude(self):
        quant_result = _make_quant_result(ticker="DAR")
        with patch.object(manual_check, "evaluate_ticker", return_value=quant_result), \
                patch.object(manual_check, "triage_ticker") as mock_deepseek, \
                patch.object(manual_check, "verify_ticker") as mock_claude:
            mock_deepseek.return_value = DeepSeekResult(verdict="no_reason", reasoning="Rien de notable.")
            mock_claude.return_value = ClaudeResult(verdict="rejected", reasoning="Trop cher.")

            result = manual_check.run_manual_check("DAR")

        self.assertEqual(result["final_verdict"], "rejected_claude")

    def test_runs_even_when_quant_filter_fails(self):
        """Le bouton doit donner un avis complet même sur un ticker qui ne
        passerait pas le filtre quant du pipeline auto -- c'est justement
        le cas d'usage (vérifier un ticker hors de la présélection habituelle)."""
        failing_quant = QuantResult(
            ticker="DAR", sector="Energy", passed=False, score=0.0, price=5.0,
            details={"pe_ratio": None, "sector": "Energy"},
            reasons_failed=["ROE 4.0% < seuil 10%"],
        )
        with patch.object(manual_check, "evaluate_ticker", return_value=failing_quant), \
                patch.object(manual_check, "triage_ticker") as mock_deepseek, \
                patch.object(manual_check, "verify_ticker") as mock_claude:
            mock_deepseek.return_value = DeepSeekResult(verdict="no_reason", reasoning="")
            mock_claude.return_value = ClaudeResult(verdict="uncertain", reasoning="")

            result = manual_check.run_manual_check("DAR")

        mock_deepseek.assert_called_once()
        mock_claude.assert_called_once()
        self.assertFalse(result["quant_passed"])
        self.assertEqual(result["final_verdict"], "uncertain")


class ManualTickerCheckViewTests(APITestCase):
    """L'endpoint doit tracer dans ManualTickerCheck uniquement -- jamais
    ScanResult/FundamentalLabPosition/CustomWatchlistTicker/PaperTrade."""

    def test_creates_manual_check_record_only(self):
        quant_result = _make_quant_result(ticker="DAR")
        with patch.object(manual_check, "evaluate_ticker", return_value=quant_result), \
                patch.object(manual_check, "triage_ticker") as mock_deepseek, \
                patch.object(manual_check, "verify_ticker") as mock_claude:
            mock_deepseek.return_value = DeepSeekResult(verdict="no_reason", reasoning="Rien de notable.")
            mock_claude.return_value = ClaudeResult(verdict="confirmed", reasoning="Bilan solide.", tokens_used=1200)

            response = self.client.post("/api/analysis/ticker/DAR/manual-check/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["final_verdict"], "confirmed")
        self.assertEqual(ManualTickerCheck.objects.count(), 1)
        record = ManualTickerCheck.objects.get()
        self.assertEqual(record.ticker, "DAR")
        self.assertEqual(record.claude_tokens_used, 1200)

        # Aucune trace ailleurs -- le point central de cette fonctionnalité.
        self.assertEqual(ScanResult.objects.count(), 0)
        self.assertEqual(FundamentalLabPosition.objects.count(), 0)

    def test_invalid_ticker_returns_404_and_no_record(self):
        with patch.object(manual_check, "evaluate_ticker", return_value=None):
            response = self.client.post("/api/analysis/ticker/ZZZINVALIDXYZ/manual-check/")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(ManualTickerCheck.objects.count(), 0)

    def test_daily_limit_enforced(self):
        for i in range(20):
            ManualTickerCheck.objects.create(ticker=f"T{i}", final_verdict="confirmed")

        response = self.client.post("/api/analysis/ticker/DAR/manual-check/")

        self.assertEqual(response.status_code, 429)
        self.assertEqual(ManualTickerCheck.objects.count(), 20)


class LabWeeklySuggestionsViewTests(APITestCase):
    """Partie 3.2 : endpoint lecture seule pour l'encart frontend."""

    def test_empty_when_no_suggestions_yet(self):
        response = self.client.get("/api/analysis/lab/suggestions/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["suggestions"], [])

    def test_returns_latest_suggestions_most_recent_first(self):
        LabWeeklySuggestion.objects.create(
            week_start="2026-08-01", positions_closed_analyzed=0, has_pattern=False,
            summary="Aucune position fermée cette semaine.",
        )
        LabWeeklySuggestion.objects.create(
            week_start="2026-08-08", positions_closed_analyzed=6, has_pattern=True,
            summary="Confiance Claude=low : 6 positions, sous-performe nettement.",
        )

        response = self.client.get("/api/analysis/lab/suggestions/")

        self.assertEqual(response.status_code, 200)
        suggestions = response.data["suggestions"]
        self.assertEqual(len(suggestions), 2)
        self.assertEqual(suggestions[0]["week_start"], "2026-08-08")
        self.assertTrue(suggestions[0]["has_pattern"])


class RunScanUsesCombinedUniverseTests(APITestCase):
    """
    2026-08-16 (Eric, explicit decision) : _run_scan_for_preset était codé
    en dur sur universe_source='sandboxes' (~23 tickers des watchlists ML,
    pas un vrai univers de titres sous-évalués) -- passé à 'combined'
    (sandboxes + CustomWatchlistTicker actifs, alimenté par
    run_broad_index_scan) maintenant que la condition d'activation
    documentée dans docs/TECH_DEBT_NOTES.md item 11 est remplie (un cycle
    réel a tourné avec succès le 2026-08-11).

    Univers vide dans ce test (aucune SandboxWatchlist/CustomWatchlistTicker
    fixture) -- _run_scan_for_preset retourne tôt sans appeler le filtre
    quant ni aucun service réseau/LLM, donc ce test vérifie uniquement le
    universe_source enregistré sur le ScanRun, sans mock nécessaire.
    """

    def test_scan_run_records_combined_universe_source(self):
        preset = _make_preset()
        tasks._run_scan_for_preset(preset)

        scan_run = ScanRun.objects.get(preset=preset)
        self.assertEqual(scan_run.universe_source, "combined")
        self.assertEqual(scan_run.status, "completed")
