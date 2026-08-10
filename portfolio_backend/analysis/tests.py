import json
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
from django.utils import timezone
from rest_framework.test import APITestCase

from portfolio.models import PaperTrade, SandboxWatchlist

from .models import (
    ScreenerPreset, ScanRun, ScanResult, FundamentalLabPosition,
    RejectionLog, RejectionOutcome,
)
from . import tasks
from .services import quant_filter
from .services.deepseek_triage import DeepSeekResult
from .services.claude_verifier import ClaudeResult
from .services.quant_filter import QuantResult


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

    def test_creates_lab_position_and_real_paper_trade(self):
        scan_result = self._make_scan_result()

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

        tasks._create_lab_position(scan_result, self.preset, Decimal("21500.00"))

        lab_position = FundamentalLabPosition.objects.get(scan_result=scan_result)
        self.assertIsNone(lab_position.paper_trade_id)
        mock_create.assert_called_once()


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
        with patch.object(tasks, "triage_ticker") as mock_deepseek, \
                patch.object(tasks, "verify_ticker") as mock_claude:
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
