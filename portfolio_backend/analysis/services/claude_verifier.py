"""
Étape 3 du screener : verdict final via Claude API (Sonnet — budget
$4-5/mois validé). Seulement appelé sur les tickers qui ont survécu
au tri DeepSeek (verdict 'no_reason' ou 'uncertain').

IMPORTANT : nécessite un compte séparé sur console.anthropic.com avec
billing configuré — un abonnement Claude Pro ne donne PAS accès à l'API.
Voir ANTHROPIC_API_KEY dans les variables d'environnement.

LIMITATION CONNUE (2026-08-09) : un seul appel Claude par ticker, pas de
vote majoritaire. temperature=0 n'est pas disponible sur ce modèle
(déprécié par l'API, confirmé contre la doc officielle) et n'aurait de
toute façon jamais garanti un déterminisme parfait même sur les anciens
modèles -- la variance entre appels identiques est donc un fait à gérer,
pas un bug à éliminer par ce biais. Le tool use forcé (voir VERDICT_TOOL
ci-dessous) a déjà réduit la variance observée en pratique (3/3 verdicts
convergents sur un cas test, contre un mélange confirmed/rejected/
uncertain avec l'ancienne approche JSON en texte libre), mais reste un
seul échantillon. Amélioration identifiée pour plus tard, pas ce soir :
faire 2-3 appels sur les tickers qui atteignent cette étape, ne garder
un verdict confiant que si les appels convergent, sinon 'uncertain' avec
un champ distinct (ex: verdict_source='llm_disagreement') pour le
distinguer d'un 'uncertain' qui vient d'un manque d'information. Coût
estimé : x2-x3 sur cette étape seulement, toujours dans le budget
mensuel visé. Nécessiterait une migration (nouveau champ sur ScanResult)
-- à faire avec la même rigueur que les migrations de ce soir, pas en
urgence un soir de déploiement.
"""

from dataclasses import dataclass
from typing import Literal
import logging
import os

import anthropic

from .news import fetch_recent_news, format_news_for_prompt

logger = logging.getLogger(__name__)

CLAUDE_MODEL = os.environ.get("CLAUDE_ANALYSIS_MODEL", "claude-sonnet-5")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

ClaudeVerdict = Literal["confirmed", "uncertain", "rejected"]

# Sortie structurée forcée via tool use plutôt que JSON en texte libre --
# élimine à la racine le bug "JSON enveloppé en ```json" trouvé le
# 2026-08-09 (Claude ne peut littéralement pas répondre autrement que
# dans ce schéma) et n'a pas le problème d'incompatibilité avec le
# préremplissage rencontré avec l'autre approche. response.content
# contiendra un bloc tool_use avec .input déjà parsé en dict par le SDK,
# plus besoin de json.loads()/extract_json_object() ici.
VERDICT_TOOL = {
    "name": "submit_verdict",
    "description": "Soumets le verdict final sur ce ticker.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["confirmed", "uncertain", "rejected"]},
            "reasoning": {"type": "string", "description": "2-4 phrases expliquant le raisonnement"},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["verdict", "reasoning", "confidence"],
    },
}

SYSTEM_PROMPT = """Tu es un analyste financier qui fait la vérification finale sur un
candidat "sous-évalué" présélectionné par un filtre quantitatif et un premier tri
automatique. Ta tâche : déterminer si la baisse/faible valorisation de ce titre a une
raison légitime spécifique à l'entreprise, ou si le marché semble simplement l'avoir
négligé sans raison fondamentale solide.

Sois rigoureux sur le raisonnement causal — distingue un vrai catalyseur négatif
(fraude, poursuite, guidance coupée, perte de contrat, problème structurel) d'un
bruit macro/sectoriel qui n'a rien à voir avec les fondamentaux propres à l'entreprise.

Utilise l'outil submit_verdict pour rendre ton verdict :
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
        # Historique des tentatives précédentes (2026-08-09) :
        # - préremplissage (message assistant "{") : rejeté par l'API,
        #   "This model does not support assistant message prefill".
        # - JSON en texte libre + extract_json_object() : fonctionnait mais
        #   fragile (Claude enveloppe parfois en ```json, ou épuise
        #   max_tokens en raisonnement étendu avant d'écrire le JSON).
        # Le tool use forcé (tool_choice) élimine les deux problèmes à la
        # racine : Claude ne peut littéralement répondre que dans le
        # schéma VERDICT_TOOL, et ça n'entre pas en conflit avec le
        # raisonnement étendu (contrairement au préremplissage) puisque le
        # modèle garde la liberté de "réfléchir" avant d'appeler l'outil.
        # max_tokens=2000 : garde la marge trouvée nécessaire le 2026-08-09
        # pour couvrir un éventuel raisonnement étendu avant l'appel d'outil.
        # temperature=0 a été essayé le 2026-08-09 pour réduire la variance
        # entre appels identiques, mais rejeté par l'API : "`temperature`
        # is deprecated for this model." -- confirmé en direct, pas juste
        # en théorie, sur claude-sonnet-5. Pas de levier de déterminisme
        # disponible ici ; le tool use forcé réduit déjà la variance
        # observée en pratique (3/3 verdicts convergents sur NVDA en test,
        # contre un mélange confirmed/rejected/uncertain avec l'ancienne
        # approche JSON en texte libre).
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            tools=[VERDICT_TOOL],
            tool_choice={"type": "tool", "name": "submit_verdict"},
        )

        tool_use_block = next(
            (block for block in response.content if block.type == "tool_use"), None
        )
        if tool_use_block is None:
            raise ValueError(
                f"Pas de bloc tool_use dans la réponse (stop_reason={response.stop_reason})"
            )
        parsed = tool_use_block.input  # déjà un dict parsé par le SDK

        verdict = parsed.get("verdict", "uncertain")
        if verdict not in ("confirmed", "uncertain", "rejected"):
            verdict = "uncertain"

        input_tokens = response.usage.input_tokens or 0
        output_tokens = response.usage.output_tokens or 0
        tokens_used = input_tokens + output_tokens
        # Pas d'estimation de coût en $ ici -- le tarif exact de
        # claude-sonnet-5 par million de tokens n'est pas quelque chose
        # qu'on veut deviner dans le code. Ces chiffres bruts suffisent à
        # vérifier la consommation réelle contre la grille tarifaire
        # actuelle sur console.anthropic.com.
        logger.info(
            "Claude verify_ticker(%s): %d input + %d output = %d tokens",
            ticker, input_tokens, output_tokens, tokens_used,
        )

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
