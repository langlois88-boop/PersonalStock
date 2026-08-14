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
from portfolio.alpaca_data_lab import submit_lab_market_order, get_lab_order_by_id

from .models import (
    ScreenerPreset, ScanRun, ScanResult, FundamentalLabPosition,
    RejectionLog, RejectionOutcome, CustomWatchlistTicker, LabWeeklySuggestion,
)
from .services.quant_filter import run_quant_filter, fetch_ticker_data
from .services.deepseek_triage import triage_ticker
from .services.claude_verifier import verify_ticker
from .services.broad_scan_filter import evaluate_ticker_strict
from .services.sanity_check import check_sanity

logger = logging.getLogger(__name__)

MISSED_WINNER_ALPHA_THRESHOLD = 15.0  # %, cf. discussion "occasion manquée"

# Partie 3 (2026-08-14) : résumé hebdomadaire automatique de patterns simples
# et objectifs entre un facteur (secteur, confiance Claude, ROE, FCF Yield)
# et la performance réelle des positions FUNDAMENTAL_LAB fermées dans la
# semaine -- voir generate_lab_weekly_suggestions ci-dessous.
#
# Nombre minimum de positions dans un groupe avant de le considérer comme un
# pattern potentiel plutôt qu'une coïncidence sur 1-2 cas.
LAB_SUGGESTION_MIN_OCCURRENCES = 5
# Écart minimum (en points de rendement %) entre la moyenne d'un groupe et
# la moyenne globale de la semaine pour que ce soit signalé comme un pattern.
LAB_SUGGESTION_DEVIATION_THRESHOLD_PCT = 10.0

# univers_source == "sandboxes" : V1, WATCHLIST/AI_BLUECHIP/AI_PENNY
# uniquement. "combined" (2026-08-11) y ajoute les survivants du balayage
# large hebdomadaire (CustomWatchlistTicker, voir run_broad_index_scan) --
# voir docs/TECH_DEBT_NOTES.md pour la condition d'activation (pas avant
# qu'un premier cycle réel ait tourné et produit des résultats sensés).
SANDBOX_UNIVERSE_SOURCES = ["WATCHLIST", "AI_BLUECHIP", "AI_PENNY"]

# Au-delà de ce nombre de tickers dans l'univers combiné, le budget Claude
# mensuel ($4-6/mois validé, cf. claude_verifier.py) risque d'être dépassé
# significativement. Warning informatif ici (ne devrait normalement jamais
# se déclencher : run_broad_index_scan applique un plafond DUR sur ce même
# nombre, voir MAX_ACTIVE_CUSTOM_TICKERS -- ce warning reste un filet de
# sécurité si CustomWatchlistTicker est modifié manuellement en dehors du
# scan hebdomadaire).
COMBINED_UNIVERSE_WARNING_THRESHOLD = 50

# Plafond DUR (pas un simple log) sur le nombre de tickers actifs dans
# CustomWatchlistTicker, appliqué à la fin de run_broad_index_scan. Un
# warning seul risquerait de passer inaperçu pendant des semaines pour
# quelqu'un qui vérifie les logs manuellement (revue du 2026-08-11) --
# au-delà de ce nombre, seuls les N meilleurs par score composite restent
# actifs, les autres sont désactivés (is_active=False, jamais supprimés).
MAX_ACTIVE_CUSTOM_TICKERS = 50

# Couverture minimale (fraction de l'univers pour laquelle les données de
# base ont été obtenues avec succès) en dessous de laquelle un cycle de
# run_broad_index_scan est considéré DÉGRADÉ. Sous ce seuil, la décroissance
# (consecutive_weeks_missed) est sautée entièrement pour ce cycle : un run
# dégradé (panne réseau, yfinance temporairement bloqué, etc.) ne doit
# jamais pénaliser un ticker comme s'il avait légitimement échoué les
# seuils -- bug identifié en revue le 2026-08-11 (rien ne distinguait avant
# "absent car recalé" de "absent car le scan a planté avant de l'évaluer").
MIN_SCAN_COVERAGE_RATIO = 0.7


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
    if universe_source == "combined":
        tickers = set(_get_universe("sandboxes"))
        custom = CustomWatchlistTicker.objects.filter(is_active=True).values_list("ticker", flat=True)
        tickers.update(str(t).strip().upper() for t in custom if t)
        if len(tickers) > COMBINED_UNIVERSE_WARNING_THRESHOLD:
            logger.warning(
                "Univers combiné = %d tickers (> %d) -- risque de dépasser le "
                "budget Claude mensuel validé ($4-6/mois). Pas bloquant, juste "
                "un signal à surveiller.",
                len(tickers), COMBINED_UNIVERSE_WARNING_THRESHOLD,
            )
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
    # verify_ticker() fait un vote majoritaire sur CLAUDE_VOTE_CALLS appels
    # (défaut 3) -- claude_result.verdict_source distingue un verdict
    # confiant ('llm_consensus'/'single_call') d'un désaccord entre appels
    # ('llm_disagreement', voir analysis/models.py::ScanResult.VERDICT_SOURCE_CHOICES).
    claude_result = verify_ticker(ticker, quant_result.details, deepseek_result.reasoning)
    scan_result.claude_verdict = claude_result.verdict
    scan_result.claude_reasoning = claude_result.reasoning
    scan_result.claude_tokens_used = claude_result.tokens_used
    scan_result.claude_confidence = claude_result.confidence
    scan_result.verdict_source = claude_result.verdict_source
    scan_result.save(update_fields=[
        "claude_verdict", "claude_reasoning", "claude_tokens_used", "claude_confidence", "verdict_source",
    ])

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
    correspondant (portfolio.PaperTrade, sandbox='FUNDAMENTAL_LAB').

    Depuis le 2026-08-11 (Partie B), un vrai ordre Alpaca est envoyé sur le
    compte paper DÉDIÉ à FUNDAMENTAL_LAB (portfolio.alpaca_data_lab, séparé
    du compte partagé WATCHLIST/AI_BLUECHIP/AI_PENNY — voir
    docs/TECH_DEBT_NOTES.md item 11) -- SAUF pour les tickers TSX ('.TO'),
    qu'Alpaca (courtier américain) ne peut pas trader du tout : ceux-là
    restent en simulation pure (broker='SIM'), comme avant, sans même
    tenter l'appel API (échec prévisible, pas la peine de le déclencher).

    La position lab est toujours créée en premier et conservée même si la
    création du PaperTrade échoue plus loin — sinon un pick confirmé
    disparaîtrait sans laisser de trace, ce qui est exactement le genre
    d'échec silencieux que ce projet a déjà dû diagnostiquer une fois
    (cf. incident "zéro trade", commit 3c01b62). L'échec (order API ou
    autre) est loggé explicitement et n'interrompt pas le scan des autres
    candidats -- si l'ordre réel échoue, aucun PaperTrade n'est créé
    (jamais de position fabriquée pour un ordre qui n'existe pas), la
    lab_position reste visible sans ordre attaché, à examiner.

    Ne crée jamais de doublon : si une position est déjà ouverte pour ce
    ticker+preset, le pick d'aujourd'hui est une reconfirmation (déjà
    visible via son propre ScanResult), pas une nouvelle position/ordre.
    Bug trouvé le 2026-08-10 : 3 scans le même soir ont créé 3 positions
    distinctes pour BTO.TO (~296$ engagés au lieu de ~99$) faute de ce
    check — voir docs/TECH_DEBT_NOTES.md item 8.
    """
    # Garde-fou de sécurité (2026-08-14, cas HAIN) -- APRÈS le verdict LLM,
    # AVANT toute création de position, peu importe si Claude a dit
    # confirmed/uncertain. Voir analysis/services/sanity_check.py. Le
    # verdict LLM d'origine (scan_result.claude_verdict) n'est PAS écrasé --
    # seul final_verdict passe à 'flagged_anomaly', pour garder la
    # traçabilité complète (ce n'est pas Claude qui a jugé mauvais, c'est un
    # seuil déterministe qui a bloqué l'ouverture malgré le verdict).
    sanity = check_sanity(scan_result.quant_details)
    if not sanity.passed:
        scan_result.final_verdict = "flagged_anomaly"
        scan_result.anomaly_reason = sanity.reason_text
        scan_result.save(update_fields=["final_verdict", "anomaly_reason"])
        logger.warning(
            "%s bloqué par le garde-fou de sécurité (verdict LLM d'origine=%s) -- "
            "aucune position créée. Raison(s) : %s",
            scan_result.ticker, scan_result.claude_verdict or "n/a", sanity.reason_text,
        )
        return

    already_open = FundamentalLabPosition.objects.filter(
        ticker=scan_result.ticker, preset=preset, is_open=True,
    ).exists()
    if already_open:
        logger.info(
            "%s a déjà une position ouverte pour le preset %s -- scan_result=%s "
            "reconfirme le pick, aucune nouvelle position/ordre créé.",
            scan_result.ticker, preset.slug, scan_result.id,
        )
        return

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
        is_tsx = scan_result.ticker.strip().upper().endswith(".TO")

        if is_tsx:
            paper_trade = PaperTrade.objects.create(
                ticker=scan_result.ticker,
                sandbox="FUNDAMENTAL_LAB",
                entry_price=round(entry_price, 4),
                quantity=quantity,
                stop_loss=stop_loss,
                entry_signal=scan_result.quant_score,
                model_name="FUNDAMENTAL_LAB",
                broker="SIM",
                notes=(
                    f"Analyse fondamentale — verdict {scan_result.final_verdict}, preset {preset.slug}. "
                    "Simulation pure : ticker TSX, hors périmètre Alpaca (courtier américain)."
                ),
            )
            lab_position.paper_trade = paper_trade
            lab_position.save(update_fields=["paper_trade"])
            return

        order = submit_lab_market_order(scan_result.ticker, quantity, "buy")
        if order is None:
            logger.error(
                "Échec de l'ordre Alpaca réel (compte FUNDAMENTAL_LAB) pour %s qty=%d — "
                "aucun PaperTrade créé, lab_position=%s conservée sans ordre, à examiner.",
                scan_result.ticker, quantity, lab_position.id,
            )
            return

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
            broker="ALPACA_LAB",
            broker_order_id=str(getattr(order, "id", "") or ""),
            broker_status=str(getattr(order, "status", "") or ""),
            broker_side="BUY",
            broker_updated_at=timezone.now(),
            notes=f"Analyse fondamentale — verdict {scan_result.final_verdict}, preset {preset.slug}. Ordre Alpaca réel (compte FUNDAMENTAL_LAB dédié).",
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


def _custom_ticker_composite_score(piotroski, altman) -> float:
    """
    Score composite simple pour CLASSER des tickers qui ont déjà TOUS passé
    les mêmes seuils stricts obligatoires (sert uniquement au plafond dur,
    pas une mesure de qualité absolue). Piotroski (0-9) et Altman Z (marge
    de sécurité au-dessus du seuil de faillite 3.0) sont les 2 champs déjà
    stockés sur CustomWatchlistTicker -- utilisables aussi bien pour un
    survivant frais de la semaine que pour un holdover non réévalué.
    """
    return (piotroski or 0) + (altman or 0)


@shared_task(bind=True, max_retries=1)
def run_broad_index_scan(self) -> dict:
    """
    Balayage hebdomadaire large : S&P 500 + TSX Composite, garde-fous
    stricts (voir services/broad_scan_filter.py). Alimente
    CustomWatchlistTicker -- ne touche PAS directement à l'univers quotidien
    (_get_universe('combined') lit cette table séparément, activé seulement
    après confirmation manuelle qu'un cycle a produit des résultats sensés,
    voir docs/TECH_DEBT_NOTES.md).

    Durée attendue LONGUE (potentiellement plusieurs heures pour ~700+
    tickers, chaque évaluation stricte nécessitant plusieurs appels
    yfinance -- balance_sheet/financials/cashflow n'ont pas d'équivalent
    batch comme yf.download() pour les prix). Volontairement pas optimisé
    plus loin pour cette itération (planifié hors marché, sur la queue
    dédiée analysis_queue -- voir TECH_DEBT_NOTES.md item 5 -- donc ne
    bloque ni le pipeline de trading ni le scan quotidien même si ça prend
    longtemps).

    Deux garde-fous ajoutés en revue le 2026-08-11 (voir docs/TECH_DEBT_NOTES.md) :
    - la décroissance (consecutive_weeks_missed) est sautée entièrement si le
      cycle est DÉGRADÉ (couverture de données < MIN_SCAN_COVERAGE_RATIO) --
      sinon une panne réseau mi-scan pénaliserait des tickers qui n'ont
      simplement jamais été évalués cette semaine, pas légitimement échoué.
    - plafond DUR (pas un simple warning) sur le nombre de tickers actifs,
      voir MAX_ACTIVE_CUSTOM_TICKERS.
    """
    from portfolio.tasks import _fetch_sp500_symbols, _fetch_tsx_symbols

    sp500 = _fetch_sp500_symbols(limit=500)
    tsx = _fetch_tsx_symbols(limit=300)
    universe = sorted(set(sp500) | set(tsx))

    if not universe:
        logger.error(
            "run_broad_index_scan: univers vide (échec des 2 sources SP500/TSX), scan annulé -- "
            "aucune modification apportée à CustomWatchlistTicker."
        )
        return {"status": "empty_universe", "sp500_count": len(sp500), "tsx_count": len(tsx)}

    logger.info("run_broad_index_scan: univers de %d tickers (%d SP500 + %d TSX).", len(universe), len(sp500), len(tsx))

    # Première passe : données de base pour tout le monde, sert à la fois de
    # pré-filtre implicite (fetch_ticker_data échoue proprement -> ticker
    # exclu) et à calculer la moyenne P/E par secteur pour la 2e passe.
    basic_data: dict[str, dict] = {}
    sector_pes: dict[str, list] = {}
    for ticker in universe:
        try:
            data = fetch_ticker_data(ticker)
        except Exception:
            logger.exception("run_broad_index_scan: échec fetch_ticker_data pour %s, ticker exclu.", ticker)
            data = None
        if data is None:
            continue
        basic_data[ticker] = data
        pe = data.get("pe_ratio")
        sector = data.get("sector") or "DEFAULT"
        if pe and pe > 0:
            sector_pes.setdefault(sector, []).append(pe)

    sector_avg_pe = {s: round(sum(v) / len(v), 2) for s, v in sector_pes.items() if v}
    coverage_ratio = len(basic_data) / len(universe)
    scan_degraded = coverage_ratio < MIN_SCAN_COVERAGE_RATIO
    logger.info(
        "run_broad_index_scan: données de base obtenues pour %d/%d tickers (couverture %.0f%%)%s.",
        len(basic_data), len(universe), coverage_ratio * 100,
        " -- CYCLE DÉGRADÉ, décroissance sautée" if scan_degraded else "",
    )
    if scan_degraded:
        logger.error(
            "run_broad_index_scan: couverture %.0f%% < seuil %.0f%% -- run considéré DÉGRADÉ "
            "(panne réseau/yfinance probable sur une bonne partie de l'univers). Les survivants "
            "trouvés seront quand même enregistrés (information positive individuellement "
            "fiable), mais AUCUNE décroissance (consecutive_weeks_missed) n'est appliquée ce "
            "cycle -- un ticker absent d'un run dégradé n'a pas nécessairement échoué les seuils.",
            coverage_ratio * 100, MIN_SCAN_COVERAGE_RATIO * 100,
        )

    survivors = []
    for ticker, data in basic_data.items():
        sector = data.get("sector") or "DEFAULT"
        try:
            result = evaluate_ticker_strict(ticker, sector_avg_pe=sector_avg_pe.get(sector), data=data)
        except Exception:
            logger.exception("run_broad_index_scan: échec evaluate_ticker_strict pour %s, ticker exclu.", ticker)
            continue
        if result.passed:
            survivors.append(result)

    survivor_tickers = set()
    for r in survivors:
        source_index = "TSX" if r.ticker.endswith(".TO") else "SP500"
        obj, created = CustomWatchlistTicker.objects.get_or_create(
            ticker=r.ticker, defaults={"source_index": source_index},
        )
        obj.piotroski_score = r.piotroski_score
        obj.altman_z_score = r.altman_z_score
        obj.consecutive_weeks_missed = 0
        obj.is_active = True
        obj.save(update_fields=["piotroski_score", "altman_z_score", "consecutive_weeks_missed", "is_active", "last_confirmed"])
        survivor_tickers.add(r.ticker)

    # Tickers déjà suivis mais absents des survivants cette semaine --
    # sortie progressive sur 3 semaines manquées, pas un retrait immédiat
    # (cf. docstring du modèle). SAUTÉ entièrement si le cycle est dégradé :
    # voir docstring de la tâche.
    newly_deactivated_by_miss = 0
    if scan_degraded:
        skipped_count = CustomWatchlistTicker.objects.filter(is_active=True).exclude(ticker__in=survivor_tickers).count()
        logger.warning(
            "run_broad_index_scan: décroissance sautée (cycle dégradé) -- %d tickers actifs "
            "non re-confirmés cette semaine laissés tels quels (consecutive_weeks_missed inchangé).",
            skipped_count,
        )
    else:
        missed_qs = CustomWatchlistTicker.objects.filter(is_active=True).exclude(ticker__in=survivor_tickers)
        for m in missed_qs:
            m.consecutive_weeks_missed += 1
            if m.consecutive_weeks_missed >= 3:
                m.is_active = False
                newly_deactivated_by_miss += 1
            m.save(update_fields=["consecutive_weeks_missed", "is_active"])

    # Plafond DUR : classe tous les tickers actuellement actifs par score
    # composite (Piotroski + Altman Z, déjà stockés -- fonctionne aussi
    # pour un holdover non réévalué cette semaine) et désactive la queue
    # au-delà de MAX_ACTIVE_CUSTOM_TICKERS. Jamais de suppression (cf.
    # principe "no silent disappearance" du projet).
    newly_deactivated_by_cap = 0
    active_rows = list(CustomWatchlistTicker.objects.filter(is_active=True))
    if len(active_rows) > MAX_ACTIVE_CUSTOM_TICKERS:
        ranked = sorted(
            active_rows,
            key=lambda t: _custom_ticker_composite_score(t.piotroski_score, t.altman_z_score),
            reverse=True,
        )
        capped_out = ranked[MAX_ACTIVE_CUSTOM_TICKERS:]
        capped_tickers = [t.ticker for t in capped_out]
        CustomWatchlistTicker.objects.filter(ticker__in=capped_tickers).update(is_active=False)
        newly_deactivated_by_cap = len(capped_out)
        logger.warning(
            "run_broad_index_scan: plafond dur atteint (%d actifs > %d) -- %d tickers désactivés "
            "(score composite le plus faible, jamais supprimés) : %s",
            len(active_rows), MAX_ACTIVE_CUSTOM_TICKERS, newly_deactivated_by_cap, capped_tickers,
        )

    total_active = CustomWatchlistTicker.objects.filter(is_active=True).count()

    result = {
        "status": "degraded" if scan_degraded else "ok",
        "universe_scanned": len(universe),
        "basic_data_ok": len(basic_data),
        "coverage_ratio": round(coverage_ratio, 3),
        "survivors_this_week": len(survivors),
        "total_active_now": total_active,
        "newly_deactivated_by_miss": newly_deactivated_by_miss,
        "newly_deactivated_by_cap": newly_deactivated_by_cap,
    }
    logger.info("run_broad_index_scan terminé: %s", result)
    return result


@shared_task(bind=True, max_retries=1)
def monitor_lab_positions(self) -> dict:
    """
    Stop-loss minimal pour les positions FUNDAMENTAL_LAB à ordre RÉEL
    (broker='ALPACA_LAB' uniquement -- les positions SIM (tickers TSX,
    échecs d'ordre) ne sont pas concernées, aucun capital réel à protéger
    pour elles). Ajouté le 2026-08-11 (Partie B) suite à une découverte en
    revue : aucun mécanisme de sortie n'existait pour FUNDAMENTAL_LAB avant
    ce soir, ni automatique ni manuel (LabPositionMarkManualView ne fait que
    poser un flag "j'y crois", jamais fermer quoi que ce soit). Acceptable
    en pure simulation DB (aucune conséquence), plus une fois du vrai
    capital paper engagé -- voir docs/TECH_DEBT_NOTES.md item 11.

    Volontairement minimal : vend au marché si le prix casse le stop_loss
    déjà calculé à l'entrée (PaperTrade.stop_loss, stocké mais jamais
    utilisé avant ce soir) -- pas de take-profit, pas de trailing stop.
    Objectif : fermer la faille de capital avant d'aller plus loin, pas
    répliquer toute la sophistication des sandboxes ML existants.

    Garde-fou ajouté le 2026-08-11 (revue avant l'ouverture du 2026-08-12,
    suite au dry-run ADBE placé hors marché, order_id 7dbe7a3d-...) : un
    ordre 'ACCEPTED'/'NEW' n'a PAS encore de position réelle derrière --
    comparer son prix au stop-loss et tenter une vente dessus serait une
    action sur du vide (échec silencieux ou bruyant selon le comportement
    d'Alpaca pour une vente à découvert non voulue, jamais testé). On
    vérifie maintenant le statut RÉEL de l'ordre via get_lab_order_by_id
    (compte dédié) avant toute comparaison de prix -- seul un ordre
    effectivement 'filled' est éligible au stop-loss. Sert aussi de
    synchro légère au passage (broker_status/entry_price recalés sur le
    remplissage réel dès qu'il est détecté) -- pas une tâche de synchro
    complète, juste ce qui est nécessaire pour que ce garde-fou soit fiable.
    """
    trades = PaperTrade.objects.filter(sandbox="FUNDAMENTAL_LAB", broker="ALPACA_LAB", status="OPEN")
    checked = 0
    closed = 0
    skipped_unfilled = 0
    for trade in trades:
        checked += 1
        if not trade.stop_loss:
            continue
        if not trade.broker_order_id:
            logger.warning(
                "monitor_lab_positions: %s (PaperTrade %s) sans broker_order_id, stop-loss non vérifiable ce cycle.",
                trade.ticker, trade.id,
            )
            continue

        order_status = get_lab_order_by_id(trade.broker_order_id)
        if order_status is None:
            logger.warning(
                "monitor_lab_positions: échec de la vérification du statut réel de l'ordre pour %s "
                "(order_id=%s) -- API Alpaca indisponible ou ordre introuvable, stop-loss non vérifié ce cycle.",
                trade.ticker, trade.broker_order_id,
            )
            continue

        raw_status = getattr(order_status, "status", None)
        status_value = str(getattr(raw_status, "value", raw_status) or "").lower()
        try:
            filled_qty = float(getattr(order_status, "filled_qty", None) or 0.0)
        except Exception:
            filled_qty = 0.0

        if status_value != "filled" or filled_qty <= 0:
            skipped_unfilled += 1
            logger.info(
                "monitor_lab_positions: %s (order_id=%s) statut réel='%s' filled_qty=%s -- pas encore rempli, "
                "stop-loss non applicable ce cycle (aucune position réelle à protéger).",
                trade.ticker, trade.broker_order_id, status_value, filled_qty,
            )
            if status_value and status_value != (trade.broker_status or "").lower():
                trade.broker_status = status_value
                trade.broker_updated_at = timezone.now()
                trade.save(update_fields=["broker_status", "broker_updated_at"])
            continue

        if (trade.broker_status or "").lower() != "filled":
            update_fields = ["broker_status", "broker_updated_at", "broker_filled_qty"]
            trade.broker_status = "filled"
            trade.broker_updated_at = timezone.now()
            trade.broker_filled_qty = filled_qty
            filled_avg_price = getattr(order_status, "filled_avg_price", None)
            try:
                if filled_avg_price is not None:
                    trade.entry_price = round(float(filled_avg_price), 4)
                    trade.broker_avg_price = float(filled_avg_price)
                    update_fields += ["entry_price", "broker_avg_price"]
            except Exception:
                pass
            trade.save(update_fields=update_fields)
            logger.info(
                "monitor_lab_positions: %s (order_id=%s) confirmé REMPLI -- entry_price recalé sur %.4f.",
                trade.ticker, trade.broker_order_id, float(trade.entry_price),
            )

        try:
            data = fetch_ticker_data(trade.ticker)
        except Exception:
            logger.exception("monitor_lab_positions: échec fetch_ticker_data pour %s, stop-loss non vérifié ce cycle.", trade.ticker)
            continue
        price = data.get("price") if data else None
        if price is None or price <= 0:
            logger.warning("monitor_lab_positions: prix indisponible pour %s, stop-loss non vérifié ce cycle.", trade.ticker)
            continue
        if float(price) > float(trade.stop_loss):
            continue

        order = submit_lab_market_order(trade.ticker, int(trade.quantity), "sell")
        if order is None:
            logger.error(
                "monitor_lab_positions: stop-loss cassé pour %s (prix=%.4f <= stop=%.4f) mais l'ordre de "
                "vente réel a ÉCHOUÉ -- position reste OPEN, capital réel toujours exposé, à vérifier "
                "manuellement sur le dashboard Alpaca (compte FUNDAMENTAL_LAB).",
                trade.ticker, float(price), float(trade.stop_loss),
            )
            continue

        trade.status = "CLOSED"
        trade.exit_price = round(float(price), 4)
        trade.exit_date = timezone.now()
        trade.pnl = round((float(trade.exit_price) - float(trade.entry_price)) * float(trade.quantity), 2)
        trade.outcome = "WIN" if float(trade.pnl) > 0 else "LOSS"
        trade.broker_status = str(getattr(order, "status", "") or "")
        trade.broker_side = "SELL"
        trade.broker_updated_at = timezone.now()
        trade.notes = (
            f"{trade.notes} | Stop-loss déclenché ({float(price):.4f} <= {float(trade.stop_loss):.4f}), "
            f"ordre de vente réel {getattr(order, 'id', '')}."
        )
        trade.save(update_fields=[
            "status", "exit_price", "exit_date", "pnl", "outcome",
            "broker_status", "broker_side", "broker_updated_at", "notes",
        ])

        lab_position = FundamentalLabPosition.objects.filter(paper_trade=trade).first()
        if lab_position:
            lab_position.is_open = False
            lab_position.exit_price = trade.exit_price
            lab_position.exit_date = trade.exit_date
            lab_position.close_reason = "stop_loss"
            try:
                entry = float(lab_position.entry_price)
                lab_position.realized_return_pct = (
                    round((float(trade.exit_price) - entry) / entry * 100, 2) if entry else None
                )
            except Exception:
                lab_position.realized_return_pct = None
            lab_position.save(update_fields=[
                "is_open", "exit_price", "exit_date", "close_reason", "realized_return_pct",
            ])
        else:
            logger.warning(
                "monitor_lab_positions: PaperTrade %s fermé (stop-loss) mais aucune FundamentalLabPosition "
                "liée trouvée -- vérifier la cohérence des données.",
                trade.id,
            )
        closed += 1

    result = {"checked": checked, "closed": closed, "skipped_unfilled": skipped_unfilled}
    logger.info("monitor_lab_positions terminé: %s", result)
    return result


def _lab_suggestion_factor_buckets(positions: list) -> dict:
    """
    Regroupe les positions fermées par facteur simple (secteur, confiance
    Claude, verdict Claude, ROE négatif, FCF Yield élevé) -- une position
    peut apparaître dans plusieurs groupes (ex. son secteur ET sa confiance)
    puisqu'on cherche des corrélations indépendantes, pas une partition.
    Aucun appel réseau/LLM ici -- des comptages/moyennes simples sur des
    données déjà en DB.
    """
    buckets: dict[str, list] = {}

    def _add(label: str, position) -> None:
        buckets.setdefault(label, []).append(position)

    for position in positions:
        scan_result = getattr(position, "scan_result", None)
        if scan_result is None:
            continue
        details = scan_result.quant_details or {}

        if scan_result.sector:
            _add(f"Secteur={scan_result.sector}", position)
        if scan_result.claude_confidence:
            _add(f"Confiance Claude={scan_result.claude_confidence}", position)
        if scan_result.claude_verdict:
            _add(f"Verdict Claude={scan_result.claude_verdict}", position)

        roe = details.get("roe")
        if roe is not None and roe < 0:
            _add("ROE négatif (< 0%)", position)

        # > 15% reste sous le seuil d'anomalie (50%, sanity_check.py) qui
        # bloque déjà l'ouverture -- ce bucket ne peut donc voir que des
        # positions "élevées mais pas assez pour être bloquées d'office".
        fcf_yield = details.get("fcf_yield")
        if fcf_yield is not None and fcf_yield > 15:
            _add("FCF Yield élevé (> 15%)", position)

    return buckets


@shared_task
def generate_lab_weekly_suggestions() -> dict:
    """
    Partie 3 (2026-08-14) : cherche des patterns simples et objectifs entre
    un facteur et la performance réelle des positions FUNDAMENTAL_LAB
    fermées dans les 7 derniers jours -- pour ne pas dépendre d'Eric qui
    fouille manuellement LabPerformanceView/RejectionAuditView chaque
    semaine. Volontairement PAS d'IA/LLM ici : une corrélation simple
    (moyenne de rendement par groupe vs moyenne globale de la semaine), avec
    un seuil minimum d'occurrences (LAB_SUGGESTION_MIN_OCCURRENCES) pour ne
    jamais signaler un pattern sur 1-2 cas isolés.

    NE MODIFIE JAMAIS un seuil ou un comportement automatiquement -- stocke
    uniquement une suggestion texte courte dans LabWeeklySuggestion, à
    valider explicitement par Eric (affichée dans un encart sur la page
    Analytics & ML Lab, voir analysis/views.py::LabWeeklySuggestionsView).

    Doit rester silencieux (pas de crash) sur un volume de données faible ou
    nul -- confirmé par test avec zéro position fermée (le cas le plus
    probable dans les premières semaines de ce garde-fou).
    """
    week_start = (timezone.now() - timedelta(days=7)).date()

    closed_positions = list(
        FundamentalLabPosition.objects.filter(
            is_open=False, exit_date__isnull=False, exit_date__date__gte=week_start,
        ).select_related("scan_result")
    )
    positions_analyzed = len(closed_positions)

    if positions_analyzed == 0:
        summary = "Aucune position FUNDAMENTAL_LAB fermée cette semaine -- rien à analyser."
        LabWeeklySuggestion.objects.create(
            week_start=week_start, positions_closed_analyzed=0, has_pattern=False, summary=summary,
        )
        logger.info("generate_lab_weekly_suggestions: aucune position fermée cette semaine, rien à analyser.")
        return {"status": "no_data", "positions_analyzed": 0}

    returns = [p.realized_return_pct for p in closed_positions if p.realized_return_pct is not None]
    if not returns:
        summary = (
            f"{positions_analyzed} position(s) fermée(s) cette semaine, mais aucune n'a de "
            "realized_return_pct renseigné -- rien à analyser."
        )
        LabWeeklySuggestion.objects.create(
            week_start=week_start, positions_closed_analyzed=positions_analyzed, has_pattern=False, summary=summary,
        )
        logger.warning("generate_lab_weekly_suggestions: %d positions fermées sans realized_return_pct.", positions_analyzed)
        return {"status": "no_returns", "positions_analyzed": positions_analyzed}

    overall_avg = sum(returns) / len(returns)

    patterns = []
    for label, group in _lab_suggestion_factor_buckets(closed_positions).items():
        group_returns = [p.realized_return_pct for p in group if p.realized_return_pct is not None]
        if len(group_returns) < LAB_SUGGESTION_MIN_OCCURRENCES:
            continue
        group_avg = sum(group_returns) / len(group_returns)
        deviation = group_avg - overall_avg
        if abs(deviation) < LAB_SUGGESTION_DEVIATION_THRESHOLD_PCT:
            continue
        direction = "sous-performe" if deviation < 0 else "surperforme"
        patterns.append(
            f"{label} : {len(group_returns)} positions, rendement moyen {group_avg:.1f}% "
            f"({direction} la moyenne globale de {overall_avg:.1f}% de {abs(deviation):.1f} points) "
            "-- garde-fou additionnel à envisager, à confirmer manuellement avant tout changement."
        )

    has_pattern = len(patterns) > 0
    if has_pattern:
        summary = " | ".join(patterns)
    else:
        summary = (
            f"{positions_analyzed} position(s) fermée(s) analysée(s), rendement moyen {overall_avg:.1f}% -- "
            f"aucun pattern net détecté cette semaine (seuil : >= {LAB_SUGGESTION_MIN_OCCURRENCES} occurrences, "
            f"écart >= {LAB_SUGGESTION_DEVIATION_THRESHOLD_PCT:.0f} points vs la moyenne)."
        )

    LabWeeklySuggestion.objects.create(
        week_start=week_start, positions_closed_analyzed=positions_analyzed,
        has_pattern=has_pattern, summary=summary,
    )
    logger.info(
        "generate_lab_weekly_suggestions terminé: %d positions analysées, pattern=%s.",
        positions_analyzed, has_pattern,
    )
    return {"status": "ok", "positions_analyzed": positions_analyzed, "has_pattern": has_pattern}
