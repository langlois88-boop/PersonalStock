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
from portfolio.alpaca_data_lab import (
    submit_lab_market_order, get_lab_order_by_id, submit_lab_stop_order, cancel_lab_order,
)
# 2026-08-26 : réutilise le helper existant plutôt que de dupliquer un
# SystemLog.objects.create() -- voir _create_lab_position, root cause du
# jour (RY/PNR) : un ordre Alpaca réel qui réussit puis un PaperTrade qui ne
# se crée pas ne laissait de trace QUE dans les logs du conteneur (perdus
# au prochain redéploiement, invisible au dashboard) -- jamais en base.
from portfolio.tasks import _system_log

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
    # Was hardcoded to "sandboxes" (~23 tickers from the ML paper-trading
    # watchlists) -- switched to "combined" (2026-08-16, Eric's explicit
    # decision) now that the activation condition documented in
    # docs/TECH_DEBT_NOTES.md item 11 is met: a real run_broad_index_scan
    # cycle completed successfully on 2026-08-11 (720 tickers -> 4
    # survivors). "sandboxes" alone was never a meaningful universe for a
    # value screener -- it just re-scanned tickers already being traded by
    # unrelated ML strategies, not an actual pool of undervalued candidates.
    scan_run = ScanRun.objects.create(preset=preset, universe_source="combined")

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
        # 2026-08-23 : ajouté avec la prise de profit dans monitor_lab_
        # positions -- 20% par défaut, dans la fourchette 20-25% déjà citée
        # dans quant_filter.py (méthodologie CAN SLIM/O'Neil) sans être
        # spécifique à ce seul preset (le Lab mélange Buffett/Minervini/
        # CAN SLIM/etc. sous un même stop/take-profit uniforme, plus simple
        # que différencier par preset avant d'avoir des données pour savoir
        # si ça vaut la peine).
        take_profit_pct = float(os.getenv("FUNDAMENTAL_LAB_TAKE_PROFIT_PCT", "0.20"))
        entry_price = float(scan_result.price_at_scan)

        if entry_price <= 0:
            logger.error(
                "Impossible de créer le PaperTrade FUNDAMENTAL_LAB pour %s : prix d'entrée invalide (%s).",
                scan_result.ticker, entry_price,
            )
            _system_log(
                "PAPER_TRADE", "ERROR",
                f"FUNDAMENTAL_LAB : prix d'entrée invalide ({entry_price}) pour {scan_result.ticker} "
                f"(preset {preset.slug}) -- aucun PaperTrade créé, lab_position={lab_position.id} à examiner.",
                symbol=scan_result.ticker,
            )
            return

        quantity = max(1, int(position_size / entry_price))
        stop_loss = round(entry_price * (1 - stop_pct), 2)
        take_profit = round(entry_price * (1 + take_profit_pct), 2)
        is_tsx = scan_result.ticker.strip().upper().endswith(".TO")

        if is_tsx:
            paper_trade = PaperTrade.objects.create(
                ticker=scan_result.ticker,
                sandbox="FUNDAMENTAL_LAB",
                entry_price=round(entry_price, 4),
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
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
            _system_log(
                "PAPER_TRADE", "ERROR",
                f"FUNDAMENTAL_LAB : échec de l'ordre Alpaca réel pour {scan_result.ticker} qty={quantity} "
                f"(preset {preset.slug}) -- aucun PaperTrade créé, lab_position={lab_position.id} à examiner.",
                symbol=scan_result.ticker,
            )
            return

        paper_trade = PaperTrade.objects.create(
            ticker=scan_result.ticker,
            sandbox="FUNDAMENTAL_LAB",
            entry_price=round(entry_price, 4),
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
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
    except Exception as exc:
        logger.exception(
            "Échec de la création du PaperTrade FUNDAMENTAL_LAB pour %s (lab_position=%s) — "
            "position lab conservée sans ordre, à examiner.",
            scan_result.ticker, lab_position.id,
        )
        # 2026-08-26 : root cause du jour (RY/PNR) -- un ordre Alpaca réel
        # placé avec succès puis un PaperTrade.objects.create() qui échoue
        # ENSUITE laisse une VRAIE position chez Alpaca sans AUCUNE trace en
        # base (ni PaperTrade, ni log persistant) -- seul logger.exception
        # ci-dessus existait avant, perdu au prochain redéploiement (voir
        # incident du jour). SystemLog persiste en base, visible du
        # dashboard, survit à un redéploiement.
        _system_log(
            "PAPER_TRADE", "ERROR",
            f"FUNDAMENTAL_LAB : exception lors de la création du PaperTrade pour {scan_result.ticker} "
            f"(preset {preset.slug}, lab_position={lab_position.id}) -- {exc!r}. Si l'ordre Alpaca réel a "
            "quand même réussi (vérifier le dashboard Alpaca, compte FUNDAMENTAL_LAB), cette position est "
            "SANS PaperTrade et donc SANS stop-loss/take-profit -- à corriger manuellement.",
            symbol=scan_result.ticker,
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


def _fundamental_lab_max_hold_days() -> int:
    return int(os.getenv("FUNDAMENTAL_LAB_MAX_HOLD_DAYS", "60"))


def _close_fundamental_lab_trade(trade: PaperTrade, price: float, close_reason: str, extra_note: str = "") -> None:
    """
    Écriture de fermeture commune à PaperTrade + sa FundamentalLabPosition
    liée -- factorisé (2026-08-23) pour que stop_loss/take_profit/
    max_hold_time, et les chemins ALPACA_LAB/SIM, écrivent tous exactement
    les mêmes champs de la même façon (avant, seul stop_loss avait ce
    code, dupliqué aurait été un risque de désynchronisation évident).
    """
    trade.status = "CLOSED"
    trade.exit_price = round(float(price), 4)
    trade.exit_date = timezone.now()
    trade.pnl = round((float(trade.exit_price) - float(trade.entry_price)) * float(trade.quantity), 2)
    trade.outcome = "WIN" if float(trade.pnl) > 0 else "LOSS"
    if extra_note:
        trade.notes = f"{trade.notes} | {extra_note}"
    trade.save(update_fields=["status", "exit_price", "exit_date", "pnl", "outcome", "notes"])

    lab_position = FundamentalLabPosition.objects.filter(paper_trade=trade).first()
    if lab_position:
        lab_position.is_open = False
        lab_position.exit_price = trade.exit_price
        lab_position.exit_date = trade.exit_date
        lab_position.close_reason = close_reason
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
            "monitor_lab_positions: PaperTrade %s fermé (%s) mais aucune FundamentalLabPosition "
            "liée trouvée -- vérifier la cohérence des données.",
            trade.id, close_reason,
        )


def _apply_split_adjustment_if_needed(trade: PaperTrade) -> bool:
    """
    Détecte un split d'actions survenu depuis l'ouverture d'une position et
    ajuste entry_price/stop_loss/take_profit/quantity en conséquence.

    Trouvé en direct sur BRCC (reverse split 1-pour-10, 2026-08-24) :
    comparer un entry_price d'AVANT le split au prix courant d'APRÈS le
    split donnait un faux mouvement de +1000%, qui a déclenché une vraie
    prise de profit (ordre de vente réel exécuté sur le compte paper
    Alpaca dédié) -- confirmé par le relevé de transactions Alpaca lui-même
    (achats ~0,84$ le 18 août, vente ~9,18$ le 24 août, exactement le
    ratio 0.1 du split). Le compte paper Alpaca n'ajuste pas le nombre
    d'actions détenues pendant un split (contrairement à un vrai
    courtier) -- ce correctif ne porte que sur NOS champs, pour que la
    comparaison de sortie reste juste ; il ne tente pas de resynchroniser
    la position réelle côté Alpaca (hors de portée ici).

    Pour ALPACA_LAB (2026-08-25) : appelée à CHAQUE cycle désormais, pas
    seulement quand un déclenchement de prix semble imminent -- depuis que
    stop_loss est protégé par un vrai ordre stop chez Alpaca (voir
    monitor_lab_positions), attendre un signal de prix pour détecter un
    split arriverait TROP TARD, l'ordre réel ayant pu se déclencher tout
    seul sur un prix post-split faussé avant qu'on ne regarde. Un appel
    réseau de plus par position par cycle est acceptable (peu de positions
    ouvertes à la fois, splits rares mais coût de la vérification faible).
    Pour SIM (pas d'ordre réel à garder synchronisé) : toujours appelée
    seulement quand take_profit semble sur le point de se déclencher,
    l'ancien comportement reste suffisant là.
    """
    try:
        splits = market_data.Ticker(trade.ticker).splits
    except Exception:
        logger.warning("monitor_lab_positions: échec fetch splits pour %s.", trade.ticker)
        return False
    if splits is None or splits.empty or trade.entry_date is None:
        return False

    try:
        relevant = splits[splits.index >= trade.entry_date]
    except Exception:
        return False
    if relevant.empty:
        return False

    ratio = 1.0
    for value in relevant:
        try:
            ratio *= float(value)
        except (TypeError, ValueError):
            continue
    if ratio == 1.0 or ratio <= 0:
        return False

    old_entry, old_stop, old_take, old_qty = trade.entry_price, trade.stop_loss, trade.take_profit, trade.quantity
    trade.entry_price = round(float(old_entry) / ratio, 4)
    if trade.stop_loss:
        trade.stop_loss = round(float(trade.stop_loss) / ratio, 2)
    if trade.take_profit:
        trade.take_profit = round(float(trade.take_profit) / ratio, 2)
    trade.quantity = round(float(old_qty) * ratio, 4)
    trade.notes = (
        f"{trade.notes} | Split détecté (ratio {ratio:g}) le {timezone.now().date().isoformat()} -- "
        f"entry_price/stop_loss/take_profit/quantity ajustés pour rester comparables au prix courant."
    )
    trade.save(update_fields=["entry_price", "stop_loss", "take_profit", "quantity", "notes"])

    lab_position = FundamentalLabPosition.objects.filter(paper_trade=trade).first()
    if lab_position:
        lab_position.entry_price = trade.entry_price
        lab_position.save(update_fields=["entry_price"])

    logger.warning(
        "monitor_lab_positions: split détecté pour %s (ratio=%s) -- entry %.4f->%.4f, "
        "stop %s->%s, take_profit %s->%s, qty %.4f->%.4f.",
        trade.ticker, ratio, float(old_entry), float(trade.entry_price),
        old_stop, trade.stop_loss, old_take, trade.take_profit, float(old_qty), float(trade.quantity),
    )
    return True


def _evaluate_lab_exit(
    trade: PaperTrade, price: float, hold_cutoff, max_hold_days: int, check_stop_loss: bool, simulation: bool = False,
) -> tuple[str | None, str]:
    suffix = ", simulation" if simulation else ""
    if check_stop_loss and trade.stop_loss and float(price) <= float(trade.stop_loss):
        return "stop_loss", f"Stop-loss déclenché ({float(price):.4f} <= {float(trade.stop_loss):.4f}{suffix})"
    if trade.take_profit and float(price) >= float(trade.take_profit):
        return "take_profit", f"Prise de profit déclenchée ({float(price):.4f} >= {float(trade.take_profit):.4f}{suffix})"
    if trade.entry_date and trade.entry_date <= hold_cutoff:
        return "max_hold_time", f"Durée maximale de détention atteinte ({max_hold_days}j{suffix})"
    return None, ""


@shared_task(bind=True, max_retries=1)
def monitor_lab_positions(self) -> dict:
    """
    Sortie automatique pour les positions FUNDAMENTAL_LAB, sur trois
    conditions : stop_loss, take_profit, max_hold_time (2026-08-23 --
    voir docstring de FundamentalLabPosition.CLOSE_REASON_CHOICES pour le
    pourquoi : un stop-loss seul ne fermait jamais les gagnants, rendant
    tout jugement de performance structurellement biaisé vers les pertes).

    Deux populations séparées :
    - broker='ALPACA_LAB' (ordre réel sur le compte paper dédié) : stop_loss
      ET take_profit ET max_hold_time, mais seulement après vérification du
      statut RÉEL de l'ordre via get_lab_order_by_id (garde-fou du
      2026-08-11, dry-run ADBE hors marché -- un ordre 'ACCEPTED'/'NEW' n'a
      pas encore de position réelle derrière, comparer son prix à quoi que
      ce soit serait une action sur du vide).
      2026-08-25 -- stop_loss n'est plus détecté par comparaison de prix
      dans ce cycle : un vrai ordre stop (GTC) est placé chez Alpaca dès
      l'achat confirmé rempli (submit_lab_stop_order), surveillé en continu
      par le courtier au lieu d'attendre le prochain poll de 10 min. Root
      cause du changement : SLQT a perdu -37,57% avant que le stop
      (visé -5%) ne soit détecté, le prix ayant gappé hors du cycle de
      poll. La comparaison de prix (check_stop_loss=True) ne reste qu'en
      secours si le placement de l'ordre stop réel a échoué. take_profit
      et max_hold_time restent détectés par comparaison de prix (pas de
      contrepartie "prise de profit" native chez Alpaca sans passer à un
      ordre bracket, hors scope ici) ; leur fermeture passe par un vrai
      ordre de vente (submit_lab_market_order), après annulation du stop
      order protecteur pour ne jamais avoir deux ordres de vente actifs en
      même temps sur la même position.
    - broker='SIM' (tickers TSX, hors périmètre Alpaca) : take_profit ET
      max_hold_time SEULEMENT -- pas de stop_loss ici par design d'origine
      (aucun capital réel à protéger), mais les exclure aussi de take_profit/
      max_hold_time aurait laissé TOUTE position TSX ouverte pour toujours
      (K.TO, FVI.TO, OGC.TO, BTO.TO...), biaisant l'échantillon fermé d'une
      toute autre façon. Fermeture directe en DB, aucun ordre réel à passer.
    """
    max_hold_days = _fundamental_lab_max_hold_days()
    hold_cutoff = timezone.now() - timedelta(days=max_hold_days)

    checked = 0
    closed = 0
    skipped_unfilled = 0

    # --- ALPACA_LAB : stop_loss + take_profit + max_hold_time ---
    alpaca_trades = PaperTrade.objects.filter(sandbox="FUNDAMENTAL_LAB", broker="ALPACA_LAB", status="OPEN")
    for trade in alpaca_trades:
        checked += 1
        if not trade.broker_order_id:
            logger.warning(
                "monitor_lab_positions: %s (PaperTrade %s) sans broker_order_id, sortie non vérifiable ce cycle.",
                trade.ticker, trade.id,
            )
            continue

        order_status = get_lab_order_by_id(trade.broker_order_id)
        if order_status is None:
            logger.warning(
                "monitor_lab_positions: échec de la vérification du statut réel de l'ordre pour %s "
                "(order_id=%s) -- API Alpaca indisponible ou ordre introuvable, sortie non vérifiée ce cycle.",
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
                "sortie non applicable ce cycle (aucune position réelle à protéger).",
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

        # Split : vérifié à CHAQUE cycle ici (voir docstring de
        # _apply_split_adjustment_if_needed) -- si détecté et qu'un stop
        # order réel est déjà vivant, il faut le replacer au nouveau
        # stop_loss AVANT qu'il ne se déclenche tout seul sur l'ancien prix.
        if _apply_split_adjustment_if_needed(trade) and trade.stop_order_id:
            old_stop_order_id = trade.stop_order_id
            if cancel_lab_order(old_stop_order_id):
                new_stop_order = submit_lab_stop_order(trade.ticker, int(trade.quantity), float(trade.stop_loss))
                trade.stop_order_id = str(getattr(new_stop_order, "id", "") or "") if new_stop_order else ""
                trade.save(update_fields=["stop_order_id"])
                if not trade.stop_order_id:
                    logger.error(
                        "monitor_lab_positions: split détecté pour %s, ancien stop order %s annulé mais le "
                        "nouveau (stop=%.2f) n'a pas pu être placé -- position SANS protection stop réelle, "
                        "à vérifier manuellement sur le dashboard Alpaca (compte FUNDAMENTAL_LAB).",
                        trade.ticker, old_stop_order_id, float(trade.stop_loss),
                    )
            else:
                logger.error(
                    "monitor_lab_positions: split détecté pour %s mais l'annulation de l'ancien stop order "
                    "(%s) a échoué -- potentiellement encore actif à un prix pré-split faux, à vérifier "
                    "manuellement sur le dashboard Alpaca.",
                    trade.ticker, old_stop_order_id,
                )

        # Place le stop order réel s'il n'en existe pas encore (nouvelle
        # position juste confirmée remplie, ou tentative précédente échouée
        # -- retentée à chaque cycle tant qu'aucun id n'est enregistré).
        if not trade.stop_order_id:
            stop_order = submit_lab_stop_order(trade.ticker, int(trade.quantity), float(trade.stop_loss))
            if stop_order is not None:
                trade.stop_order_id = str(getattr(stop_order, "id", "") or "")
                trade.save(update_fields=["stop_order_id"])
            else:
                logger.error(
                    "monitor_lab_positions: échec du placement du stop order réel pour %s (stop=%.2f) -- "
                    "comparaison de prix utilisée en secours ce cycle.",
                    trade.ticker, float(trade.stop_loss),
                )

        # Le stop order réel est la sortie stop_loss primaire désormais --
        # vérifié AVANT toute comparaison de prix, il a pu se déclencher
        # n'importe quand depuis le dernier cycle (c'est tout le point).
        if trade.stop_order_id:
            stop_status = get_lab_order_by_id(trade.stop_order_id)
            stop_status_value = ""
            if stop_status is not None:
                stop_raw_status = getattr(stop_status, "status", None)
                stop_status_value = str(getattr(stop_raw_status, "value", stop_raw_status) or "").lower()
            if stop_status_value == "filled":
                try:
                    stop_filled_price = float(getattr(stop_status, "filled_avg_price", None) or 0.0)
                except Exception:
                    stop_filled_price = 0.0
                if stop_filled_price > 0:
                    trade.broker_status = "filled"
                    trade.broker_side = "SELL"
                    trade.broker_updated_at = timezone.now()
                    trade.save(update_fields=["broker_status", "broker_side", "broker_updated_at"])
                    _close_fundamental_lab_trade(
                        trade, stop_filled_price, "stop_loss",
                        f"Stop-loss réel déclenché chez Alpaca (stop order {trade.stop_order_id}, "
                        f"prix rempli {stop_filled_price:.4f}).",
                    )
                    closed += 1
                    continue

        try:
            data = fetch_ticker_data(trade.ticker)
        except Exception:
            logger.exception("monitor_lab_positions: échec fetch_ticker_data pour %s, sortie non vérifiée ce cycle.", trade.ticker)
            continue
        price = data.get("price") if data else None
        if price is None or price <= 0:
            logger.warning("monitor_lab_positions: prix indisponible pour %s, sortie non vérifiée ce cycle.", trade.ticker)
            continue

        # check_stop_loss=False dès qu'un stop order réel est vivant (il est
        # la source de vérité pour stop_loss ci-dessus) -- ne repasse à la
        # comparaison de prix que si aucun stop order réel n'a pu être
        # établi (secours, voir plus haut).
        close_reason, note = _evaluate_lab_exit(
            trade, price, hold_cutoff, max_hold_days, check_stop_loss=not bool(trade.stop_order_id),
        )
        if close_reason is None:
            continue

        # take_profit / max_hold_time (ou stop_loss de secours) : annule le
        # stop order protecteur AVANT de vendre manuellement pour ne jamais
        # laisser deux ordres de vente actifs en même temps sur la même
        # position (double-vente ou rejet Alpaca pour qty insuffisante).
        if trade.stop_order_id and not cancel_lab_order(trade.stop_order_id):
            logger.error(
                "monitor_lab_positions: sortie '%s' déclenchée pour %s mais l'annulation du stop order "
                "protecteur (%s) a échoué -- vente manuelle ANNULÉE ce cycle pour éviter un double ordre de "
                "vente, à vérifier manuellement sur le dashboard Alpaca.",
                close_reason, trade.ticker, trade.stop_order_id,
            )
            continue

        order = submit_lab_market_order(trade.ticker, int(trade.quantity), "sell")
        if order is None:
            logger.error(
                "monitor_lab_positions: sortie '%s' déclenchée pour %s (prix=%.4f) mais l'ordre de "
                "vente réel a ÉCHOUÉ -- position reste OPEN, capital réel toujours exposé, à vérifier "
                "manuellement sur le dashboard Alpaca (compte FUNDAMENTAL_LAB).",
                close_reason, trade.ticker, float(price),
            )
            continue

        trade.broker_status = str(getattr(order, "status", "") or "")
        trade.broker_side = "SELL"
        trade.broker_updated_at = timezone.now()
        trade.stop_order_id = ""
        trade.save(update_fields=["broker_status", "broker_side", "broker_updated_at", "stop_order_id"])
        _close_fundamental_lab_trade(trade, price, close_reason, f"{note}, ordre de vente réel {getattr(order, 'id', '')}.")
        closed += 1

    # --- SIM (TSX) : take_profit + max_hold_time seulement, fermeture directe ---
    sim_trades = PaperTrade.objects.filter(sandbox="FUNDAMENTAL_LAB", broker="SIM", status="OPEN")
    for trade in sim_trades:
        checked += 1
        try:
            data = fetch_ticker_data(trade.ticker)
        except Exception:
            logger.exception("monitor_lab_positions: échec fetch_ticker_data pour %s (SIM), sortie non vérifiée ce cycle.", trade.ticker)
            continue
        price = data.get("price") if data else None
        if price is None or price <= 0:
            logger.warning("monitor_lab_positions: prix indisponible pour %s (SIM), sortie non vérifiée ce cycle.", trade.ticker)
            continue

        close_reason, note = _evaluate_lab_exit(
            trade, price, hold_cutoff, max_hold_days, check_stop_loss=False, simulation=True,
        )
        if close_reason == "take_profit" and _apply_split_adjustment_if_needed(trade):
            close_reason, note = _evaluate_lab_exit(
                trade, price, hold_cutoff, max_hold_days, check_stop_loss=False, simulation=True,
            )
        if close_reason is None:
            continue

        _close_fundamental_lab_trade(trade, price, close_reason, note)
        closed += 1

    result = {"checked": checked, "closed": closed, "skipped_unfilled": skipped_unfilled}
    logger.info("monitor_lab_positions terminé: %s", result)
    return result


@shared_task
def sync_watchlist_from_fundamental_lab(sandbox: str = "WATCHLIST") -> dict:
    """
    Alimente portfolio.SandboxWatchlist(sandbox) avec les tickers
    actuellement CONFIRMÉS et ouverts dans FUNDAMENTAL_LAB (2026-08-23).

    Remplace, pour WATCHLIST, l'ancien mécanisme d'ajout (fusion des
    trouvailles de portfolio.tasks.market_scanner_task, un scanner penny
    stock 0,50$-10$) -- trouvé en discutant pourquoi WATCHLIST ne
    contenait que des titres sous 10$ alors qu'il partage son modèle
    (FUSION) avec AI_BLUECHIP, un vrai décalage prix/style. Les picks
    "confirmed" de FUNDAMENTAL_LAB (Claude n'a trouvé AUCUNE raison
    fondamentale négative -- voir claude_verifier.py) sont un signal plus
    solide qu'un simple scan technique : vraie vérification fondamentale
    + raisonnement LLM, pas juste RSI/RVOL.

    N'AJOUTE jamais que dans l'espace restant (jusqu'à
    WATCHLIST_FROM_LAB_LIMIT) -- ne retire JAMAIS un symbole ici, ce n'est
    pas son rôle (voir portfolio.tasks.prune_watchlist_weekly pour le
    retrait, source-agnostique, peu importe comment un titre est arrivé
    dans la liste).
    """
    limit = int(os.getenv("WATCHLIST_FROM_LAB_LIMIT", "25"))
    # dict.fromkeys(...) plutôt que .distinct() sur le values_list : combiné
    # à .order_by("-entry_date") sur un champ différent du values_list,
    # .distinct() ne déduplique pas réellement (piège Django documenté --
    # order_by() ajoute la colonne au SELECT, ce qui casse DISTINCT sur un
    # sous-ensemble de colonnes) -- trouvé par un test avec deux positions
    # confirmées sur le même ticker (dates d'entrée différentes).
    ordered_tickers = (
        FundamentalLabPosition.objects.filter(scan_result__final_verdict="confirmed", is_open=True)
        .order_by("-entry_date")
        .values_list("ticker", flat=True)
    )
    confirmed_tickers = list(dict.fromkeys(ordered_tickers))
    if not confirmed_tickers:
        return {"status": "no_data", "sandbox": sandbox}

    watch, _ = SandboxWatchlist.objects.get_or_create(sandbox=sandbox, defaults={"symbols": []})
    existing = list(watch.symbols or [])
    new_unique = [t for t in confirmed_tickers if t not in existing]
    room = max(0, limit - len(existing))
    added = new_unique[:room]

    if not added:
        logger.info(
            "sync_watchlist_from_fundamental_lab: rien à ajouter à %s (existants=%d, room=%d).",
            sandbox, len(existing), room,
        )
        return {"status": "ok", "sandbox": sandbox, "added": [], "existing_count": len(existing)}

    watch.symbols = existing + added
    watch.source = "fundamental_lab_confirmed"
    watch.save(update_fields=["symbols", "source"])
    logger.info(
        "sync_watchlist_from_fundamental_lab: %d titre(s) ajouté(s) à %s -- %s",
        len(added), sandbox, ", ".join(added),
    )
    return {"status": "ok", "sandbox": sandbox, "added": added, "existing_count": len(existing) + len(added)}


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
