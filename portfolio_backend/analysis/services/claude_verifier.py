"""
Étape 3 du screener : verdict final via Claude API (Sonnet — budget
$4-5/mois validé). Seulement appelé sur les tickers qui ont survécu
au tri DeepSeek (verdict 'no_reason' ou 'uncertain').

IMPORTANT : nécessite un compte séparé sur console.anthropic.com avec
billing configuré — un abonnement Claude Pro ne donne PAS accès à l'API.
Voir ANTHROPIC_API_KEY dans les variables d'environnement.
"""

from dataclasses import dataclass
from typing import Literal
import logging
import os

import anthropic

from .llm_json import extract_json_object
from .news import fetch_recent_news, format_news_for_prompt

logger = logging.getLogger(__name__)

CLAUDE_MODEL = os.environ.get("CLAUDE_ANALYSIS_MODEL", "claude-sonnet-5")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

ClaudeVerdict = Literal["confirmed", "uncertain", "rejected"]

SYSTEM_PROMPT = """Tu es un analyste financier qui fait la vérification finale sur un
candidat "sous-évalué" présélectionné par un filtre quantitatif et un premier tri
automatique. Ta tâche : déterminer si la baisse/faible valorisation de ce titre a une
raison légitime spécifique à l'entreprise, ou si le marché semble simplement l'avoir
négligé sans raison fondamentale solide.

Sois rigoureux sur le raisonnement causal — distingue un vrai catalyseur négatif
(fraude, poursuite, guidance coupée, perte de contrat, problème structurel) d'un
bruit macro/sectoriel qui n'a rien à voir avec les fondamentaux propres à l'entreprise.

Réponds UNIQUEMENT en JSON, format exact :
{
  "verdict": "confirmed" | "uncertain" | "rejected",
  "reasoning": "2-4 phrases expliquant ton raisonnement",
  "confidence": "low" | "medium" | "high"
}

- "confirmed" : aucune raison fondamentale trouvée, semble être une vraie inefficience de marché
- "rejected" : raison négative claire et spécifique à l'entreprise identifiée
- "uncertain" : signal mixte, informations insuffisantes, ou situation véritablement ambiguë"""


@dataclass
class ClaudeResult:
    verdict: ClaudeVerdict
    reasoning: str
    confidence: str = "low"
    tokens_used: int = 0


def _get_client() -> "anthropic.Anthropic":
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY manquante. Rappel : un abonnement Claude Pro ne "
            "donne pas de clé API — il faut un compte séparé sur console.anthropic.com."
        )
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def verify_ticker(ticker: str, quant_details: dict, deepseek_reasoning: str = "") -> ClaudeResult:
    """
    Appel Claude pour le verdict final. En cas d'erreur API, retourne
    'uncertain' par défaut plutôt que de faire planter le scan complet —
    un seul ticker en échec ne doit pas bloquer le reste du run.
    """
    news_items = fetch_recent_news(ticker, days_back=60)
    news_text = format_news_for_prompt(news_items)

    ratios_summary = (
        f"P/E: {quant_details.get('pe_ratio')}, PEG: {quant_details.get('peg_ratio')}, "
        f"ROE: {quant_details.get('roe')}%, Dette/EBITDA: {quant_details.get('debt_to_ebitda')}x, "
        f"FCF Yield: {quant_details.get('fcf_yield')}%, Secteur: {quant_details.get('sector')}"
    )

    user_message = (
        f"Ticker: {ticker}\n\n"
        f"Ratios fondamentaux: {ratios_summary}\n\n"
        f"Verdict du premier tri automatique: {deepseek_reasoning or 'N/A'}\n\n"
        f"News récentes (60 derniers jours):\n{news_text}"
    )

    try:
        client = _get_client()
        # NOTE : la technique de préremplissage (message assistant "{" pour
        # forcer une continuation JSON directe) a été essayée le 2026-08-09
        # mais rejetée par l'API : "This model does not support assistant
        # message prefill. The conversation must end with a user message."
        # (confirmé en prod avec claude-sonnet-5, pas juste en théorie —
        # certains modèles à raisonnement étendu désactivent le prefill).
        # On garde donc un seul message utilisateur, et on compte
        # uniquement sur extract_json_object() pour nettoyer un éventuel
        # bloc ```json ... ``` dans la réponse.
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw_text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        parsed = extract_json_object(raw_text)

        verdict = parsed.get("verdict", "uncertain")
        if verdict not in ("confirmed", "uncertain", "rejected"):
            verdict = "uncertain"

        tokens_used = (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)

        return ClaudeResult(
            verdict=verdict,
            reasoning=parsed.get("reasoning", ""),
            confidence=parsed.get("confidence", "low"),
            tokens_used=tokens_used,
        )
    except (anthropic.APIError, ValueError, RuntimeError) as e:
        logger.exception("Erreur vérification Claude pour %s", ticker)
        return ClaudeResult(
            verdict="uncertain",
            reasoning=f"Erreur technique lors de l'appel Claude : {e}",
            confidence="low",
        )
