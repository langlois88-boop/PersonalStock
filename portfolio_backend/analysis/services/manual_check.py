"""
Vérification manuelle à la demande (2026-08-13) -- réutilise EXACTEMENT
les mêmes fonctions que le pipeline automatique quotidien
(evaluate_ticker, triage_ticker, verify_ticker, voir
analysis/tasks.py::_process_candidate pour l'orchestration d'origine dont
celle-ci est un miroir en lecture seule), sur N'IMPORTE QUEL ticker tapé
par l'utilisateur, pas seulement l'univers habituel (SandboxWatchlist).

N'écrit RIEN dans ScanResult / FundamentalLabPosition /
CustomWatchlistTicker / PaperTrade -- aucune position, aucun ajout à un
sandbox. Le résultat est retourné directement à l'appelant (voir
analysis/views.py::ManualTickerCheckView, qui se charge lui-même de la
trace optionnelle dans ManualTickerCheck).
"""
from __future__ import annotations

from typing import Any

from .quant_filter import evaluate_ticker
from .deepseek_triage import triage_ticker
from .claude_verifier import verify_ticker


def run_manual_check(ticker: str) -> dict[str, Any]:
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return {"error": "Ticker manquant."}

    # Pas de preset associé à une vérification ad hoc -- seuils par
    # secteur par défaut uniquement (SECTOR_THRESHOLDS via merge_thresholds),
    # cohérent avec le fait que ce bouton n'est lié à aucun preset précis.
    quant_result = evaluate_ticker(ticker, preset_thresholds={})
    if quant_result is None:
        return {
            "ticker": ticker,
            "error": (
                "Données insuffisantes pour ce ticker -- introuvable ou "
                "données financières manquantes (yfinance)."
            ),
        }

    result: dict[str, Any] = {
        "ticker": ticker,
        "sector": quant_result.sector,
        "quant_score": quant_result.score,
        "quant_details": quant_result.details,
        "price": quant_result.price,
        "quant_passed": quant_result.passed,
        "quant_reasons_failed": quant_result.reasons_failed,
        "claude_verdict": "",
        "claude_reasoning": "",
        "claude_tokens_used": 0,
    }

    # --- Étape 2 : DeepSeek (toujours exécutée, même si le filtre quant
    # échoue -- ce bouton est pensé comme un "avis complet à la demande",
    # pas une reproduction stricte du pipeline auto qui, lui, ne reçoit
    # que des candidats déjà passés par le filtre quant). ---
    deepseek_result = triage_ticker(ticker)
    result["deepseek_verdict"] = deepseek_result.verdict
    result["deepseek_reasoning"] = deepseek_result.reasoning

    if deepseek_result.verdict == "negative_reason":
        result["final_verdict"] = "rejected_deepseek"
        return result

    # --- Étape 3 : Claude (seulement si DeepSeek n'a pas rejeté, exactement
    # comme le pipeline automatique). ---
    claude_result = verify_ticker(ticker, quant_result.details, deepseek_result.reasoning)
    result["claude_verdict"] = claude_result.verdict
    result["claude_reasoning"] = claude_result.reasoning
    result["claude_tokens_used"] = claude_result.tokens_used

    if claude_result.verdict == "rejected":
        result["final_verdict"] = "rejected_claude"
    else:
        result["final_verdict"] = claude_result.verdict  # "confirmed" ou "uncertain"

    return result
