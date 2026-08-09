"""
Tâches Celery pour le module d'analyse fondamentale. Voir
CELERY_BEAT_SCHEDULE dans portfolio_backend/settings.py pour le
scan quotidien (fermeture ET) et le suivi hebdomadaire des rejets.

IMPORTANT — leçon apprise sur ce projet (cf. commit 3c01b62, incident
"zéro trade") : un `NameError` silencieux dans la création d'une
PaperTrade a déjà fait planter la création de trade dans
_execute_paper_trades_for_sandbox sans qu'aucun trade ne soit jamais créé
ni qu'aucune erreur ne remonte. Ce module suit le même garde-fou : la
création de PaperTrade est isolée dans son propre try/except avec
logger.exception explicite, jamais un `except: pass` nu.

Ce module utilise ses propres tables (ScanRun, ScanResult, etc.) — aucun
chevauchement avec les jobs Celery Beat existants (refresh AI_BLUECHIP,
etc.), qui vivent dans portfolio.tasks et ne touchent jamais ces tables.
La seule table partagée est portfolio.PaperTrade (sandbox='FUNDAMENTAL_LAB',
valeur dédiée qui n'existe dans aucun autre pipeline) donc aucune race
condition possible sur les sandboxes ML existants.
"""

from datetime import timedelta
from decimal import Decimal
import logging
import os

from celery import shared_task
from django.utils import timezone

from portfolio import market_data
from portfolio.models import PaperTrade, SandboxWatchlist

from .models import (
    ScreenerPreset, ScanRun, ScanResult, FundamentalLabPosition,
    RejectionLog, RejectionOutcome,
)
from .services.quant_filter import run_quant_filter
from .services.deepseek_triage import triage_ticker
from .services.claude_verifier import verify_ticker

logger = logging.getLogger(__name__)

MISSED_WINNER_ALPHA_THRESHOLD = 15.0  # %, cf. discussion "occasion manquée"

# Univers de départ V1 : sandboxes existants uniquement (pas d'ajout de
# tickers custom ni d'élargissement pour l'instant, cf. brief).
SANDBOX_UNIVERSE_SOURCES = ["WATCHLIST", "AI_BLUECHIP", "AI_PENNY"]


def _get_universe(universe_source: str) -> list:
    """
    Récupère la liste de tickers à scanner.

    Le repo n'a pas de modèle `Sandbox` par ticker : les sandboxes existants
    (WATCHLIST/AI_BLUECHIP/AI_PENNY) sont stockés dans SandboxWatchlist, une
    ligne par sandbox avec un champ JSON `symbols` (liste de tickers) — pas
    une table avec un enregistrement par ticker.
    """
    if universe_source == "sandboxes":
        rows = SandboxWatchlist.objects.filter(sandbox__in=SANDBOX_UNIVERSE_SOURCES)
        tickers: set = set()
        for row in rows:
            for symbol in (row.symbols or []):
                if symbol:
                    tickers.add(str(symbol).strip().upper())
        return sorted(tickers)
    return []


def _get_benchmark_price(symbol: str = "^GSPTSE") -> Decimal:
    """
    Niveau du benchmark TSX, via portfolio.market_data.Ticker (pas yfinance
    direct, cohérent avec le reste du module). Vérifié par un scan à blanc :
    Ticker.info ne résout SYMBOL_ALIASES (^GSPTSE -> XIU.TO) que pour le
    fallback prix Alpaca (get_latest_trade_price sur le symbole mappé) —
    Alpaca n'a pas de données pour un titre coté TSX, donc cette branche ne
    matche jamais ici et .info retombe sur yfinance brut avec le symbole
    d'origine, qui renvoie le niveau de l'indice ^GSPTSE (points, pas $).
    Sans incidence sur l'alpha calculé plus loin (return_pct/benchmark_return_pct
    sont des variations en %, peu importe l'unité, tant que la même source
    est utilisée à chaque appel — ce qui est le cas ici).
    """
    try:
        price = market_data.Ticker(symbol).info.get("regularMarketPrice")
        return Decimal(str(price)) if price else Decimal("0")
    except Exception:
        logger.exception("Erreur fetch prix benchmark %s", symbol)
        return Decimal("0")


@shared_task(bind=True, max_retries=2)
def run_daily_scan(self, preset_slug: str = None):
    """
    Point d'entrée principal. Si preset_slug est None, tourne pour tous
    les presets actifs. Orchestration complète : filtre quant -> DeepSeek
    -> Claude -> écriture en DB (sandbox ou log de rejet).
    """
    presets = ScreenerPreset.objects.filter(is_active=True)
    if preset_slug:
        presets = presets.filter(slug=preset_slug)

    for preset in presets:
        _run_scan_for_preset(preset)


def _run_scan_for_preset(preset: ScreenerPreset):
    scan_run = ScanRun.objects.create(preset=preset, universe_source="sandboxes")

    try:
        tickers = _get_universe(scan_run.universe_source)
        scan_run.tickers_scanned = len(tickers)
        scan_run.save(update_fields=["tickers_scanned"])

        if not tickers:
            logger.warning("Univers vide pour le preset %s, scan annulé.", preset.name)
            scan_run.status = "completed"
            scan_run.finished_at = timezone.now()
            scan_run.save()
            return

        quant_results = run_quant_filter(tickers, preset.thresholds)
        scan_run.candidates_after_quant_filter = len(quant_results)
        scan_run.save(update_fields=["candidates_after_quant_filter"])

        benchmark_price = _get_benchmark_price()

        for qr in quant_results:
            _process_candidate(scan_run, preset, qr, benchmark_price)

        scan_run.status = "completed"
        scan_run.finished_at = timezone.now()
        scan_run.save()
        logger.info(
            "Scan %s terminé : %d tickers scannés, %d candidats après filtre quant.",
            preset.name, scan_run.tickers_scanned, scan_run.candidates_after_quant_filter,
        )

    except Exception:
        scan_run.status = "failed"
        scan_run.error_message = "Voir logs Celery pour le traceback complet."
        scan_run.finished_at = timezone.now()
        scan_run.save()
        logger.exception("Échec du scan pour le preset %s", preset.name)


def _process_candidate(scan_run: ScanRun, preset: ScreenerPreset, quant_result, benchmark_price: Decimal):
    """Fait passer un candidat par les étapes 2 et 3, puis écrit le résultat final."""
    ticker = quant_result.ticker

    scan_result = ScanResult.objects.create(
        scan_run=scan_run,
        ticker=ticker,
        sector=quant_result.sector,
        quant_score=quant_result.score,
        quant_details=quant_result.details,
        price_at_scan=quant_result.price,
        final_verdict="uncertain",  # placeholder, mis à jour ci-dessous
    )

    if not preset.requires_news_check:
        # Preset purement quantitatif (ex. Quality Compounder) — pas de LLM
        scan_result.final_verdict = "confirmed"
        scan_result.save(update_fields=["final_verdict"])
        _create_lab_position(scan_result, preset, benchmark_price)
        return

    # --- Étape 2 : DeepSeek ---
    deepseek_result = triage_ticker(ticker)
    scan_result.deepseek_verdict = deepseek_result.verdict
    scan_result.deepseek_reasoning = deepseek_result.reasoning
    scan_result.save(update_fields=["deepseek_verdict", "deepseek_reasoning"])

    if deepseek_result.verdict == "negative_reason":
        scan_result.final_verdict = "rejected_deepseek"
        scan_result.save(update_fields=["final_verdict"])
        _create_rejection_log(scan_result, preset, "deepseek", deepseek_result.reasoning, benchmark_price)
        return

    # --- Étape 3 : Claude (seulement si DeepSeek n'a pas rejeté) ---
    claude_result = verify_ticker(ticker, quant_result.details, deepseek_result.reasoning)
    scan_result.claude_verdict = claude_result.verdict
    scan_result.claude_reasoning = claude_result.reasoning
    scan_result.claude_tokens_used = claude_result.tokens_used
    scan_result.save(update_fields=["claude_verdict", "claude_reasoning", "claude_tokens_used"])

    if claude_result.verdict == "rejected":
        scan_result.final_verdict = "rejected_claude"
        scan_result.save(update_fields=["final_verdict"])
        _create_rejection_log(scan_result, preset, "claude", claude_result.reasoning, benchmark_price)
        return

    # confirmed ou uncertain -> entre dans le sandbox
    scan_result.final_verdict = claude_result.verdict  # "confirmed" ou "uncertain"
    scan_result.save(update_fields=["final_verdict"])
    _create_lab_position(scan_result, preset, benchmark_price)


def _create_lab_position(scan_result: ScanResult, preset: ScreenerPreset, benchmark_price: Decimal):
    """
    Crée l'enregistrement FundamentalLabPosition puis l'ordre paper trade
    Alpaca réel correspondant (portfolio.PaperTrade, sandbox='FUNDAMENTAL_LAB').

    La position lab est toujours créée en premier et conservée même si la
    création du PaperTrade échoue plus loin — sinon un pick confirmé
    disparaîtrait sans laisser de trace, ce qui est exactement le genre
    d'échec silencieux que ce projet a déjà dû diagnostiquer une fois
    (cf. incident "zéro trade", commit 3c01b62). L'échec est loggé
    explicitement et n'interrompt pas le scan des autres candidats.
    """
    lab_position = FundamentalLabPosition.objects.create(
        scan_result=scan_result,
        ticker=scan_result.ticker,
        preset=preset,
        entry_price=scan_result.price_at_scan,
        entry_date=timezone.now(),
        benchmark_price_at_entry=benchmark_price,
    )

    try:
        position_size = float(os.getenv("FUNDAMENTAL_LAB_POSITION_SIZE", "100"))
        stop_pct = float(os.getenv("FUNDAMENTAL_LAB_STOP_PCT", "0.05"))
        entry_price = float(scan_result.price_at_scan)

        if entry_price <= 0:
            logger.error(
                "Impossible de créer le PaperTrade FUNDAMENTAL_LAB pour %s : prix d'entrée invalide (%s).",
                scan_result.ticker, entry_price,
            )
            return

        quantity = max(1, int(position_size / entry_price))
        stop_loss = round(entry_price * (1 - stop_pct), 2)

        paper_trade = PaperTrade.objects.create(
            ticker=scan_result.ticker,
            sandbox="FUNDAMENTAL_LAB",
            entry_price=round(entry_price, 4),
            quantity=quantity,
            stop_loss=stop_loss,
            entry_signal=scan_result.quant_score,
            # model_name par défaut sur PaperTrade est 'BLUECHIP' (valeur du
            # champ) — ce pick vient du screener d'analyse fondamentale, pas
            # du modèle ML BLUECHIP ; explicite ici pour ne jamais être
            # confondu avec ce modèle dans une future requête qui grouperait
            # par model_name sans regarder le sandbox.
            model_name="FUNDAMENTAL_LAB",
            broker="SIM",
            notes=f"Analyse fondamentale — verdict {scan_result.final_verdict}, preset {preset.slug}",
        )
        lab_position.paper_trade = paper_trade
        lab_position.save(update_fields=["paper_trade"])
    except Exception:
        logger.exception(
            "Échec de la création du PaperTrade FUNDAMENTAL_LAB pour %s (lab_position=%s) — "
            "position lab conservée sans ordre, à examiner.",
            scan_result.ticker, lab_position.id,
        )


def _create_rejection_log(
    scan_result: ScanResult, preset: ScreenerPreset, rejected_by: str, reasoning: str, benchmark_price: Decimal,
):
    RejectionLog.objects.create(
        scan_result=scan_result,
        ticker=scan_result.ticker,
        preset=preset,
        rejected_by=rejected_by,
        reasoning_text=reasoning,
        quant_score=scan_result.quant_score,
        price_at_rejection=scan_result.price_at_scan,
        benchmark_price_at_rejection=benchmark_price,
    )


@shared_task
def check_rejection_outcomes():
    """
    Tâche hebdomadaire : pour chaque RejectionLog ayant atteint 30/60/90
    jours (± quelques jours de tolérance vu la cadence hebdomadaire),
    calcule le rendement depuis le rejet vs le benchmark.
    """
    benchmark_price_now = _get_benchmark_price()
    now = timezone.now()

    for days_target in (30, 60, 90):
        window_start = now - timedelta(days=days_target + 3)
        window_end = now - timedelta(days=days_target - 3)

        logs = RejectionLog.objects.filter(
            date_rejection__gte=window_start,
            date_rejection__lte=window_end,
        ).exclude(outcomes__days_since_rejection=days_target)

        for log in logs:
            _check_single_outcome(log, days_target, benchmark_price_now)


def _check_single_outcome(log: RejectionLog, days_target: int, benchmark_price_now: Decimal):
    try:
        current_price = market_data.Ticker(log.ticker).info.get("regularMarketPrice")
        if not current_price:
            return

        price_then = float(log.price_at_rejection)
        price_now = float(current_price)
        if price_then <= 0:
            logger.warning("price_at_rejection invalide pour rejet %s (%s), skip.", log.ticker, log.id)
            return
        return_pct = (price_now - price_then) / price_then * 100

        # Rendement benchmark réel sur la même fenêtre, à partir du prix
        # capturé au moment du rejet (RejectionLog.benchmark_price_at_rejection).
        # Les rejets antérieurs à l'ajout de ce champ n'ont pas cette valeur
        # (0 ou None) : on retombe alors sur 0.0 plutôt que de planter, mais
        # l'alpha n'est pas fiable dans ce cas — loggé pour transparence.
        benchmark_price_then = float(log.benchmark_price_at_rejection or 0)
        if benchmark_price_then > 0 and benchmark_price_now:
            benchmark_return_pct = (float(benchmark_price_now) - benchmark_price_then) / benchmark_price_then * 100
        else:
            logger.warning(
                "benchmark_price_at_rejection absent pour rejet %s (%s) — alpha calculé sans comparaison benchmark.",
                log.ticker, log.id,
            )
            benchmark_return_pct = 0.0

        alpha = return_pct - benchmark_return_pct

        RejectionOutcome.objects.create(
            rejection_log=log,
            days_since_rejection=days_target,
            price_then=price_then,
            price_now=price_now,
            return_pct=round(return_pct, 2),
            benchmark_return_pct=round(benchmark_return_pct, 2),
            alpha=round(alpha, 2),
            possible_missed_winner=alpha > MISSED_WINNER_ALPHA_THRESHOLD,
        )
    except Exception:
        logger.exception("Erreur calcul outcome pour rejet %s (%s)", log.ticker, log.id)
