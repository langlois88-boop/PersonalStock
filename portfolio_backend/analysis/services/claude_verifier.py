"""
Étape 3 du screener : verdict final via Claude API (Sonnet — budget
$4-5/mois validé). Seulement appelé sur les tickers qui ont survécu
au tri DeepSeek (verdict 'no_reason' ou 'uncertain').

IMPORTANT : nécessite un compte séparé sur console.anthropic.com avec
billing configuré — un abonnement Claude Pro ne donne PAS accès à l'API.
Voir ANTHROPIC_API_KEY dans les variables d'environnement.

VOTE MAJORITAIRE (ajouté le 2026-08-10, après le déploiement des 6 fixes
du 2026-08-09) : temperature=0 n'est pas disponible sur ce modèle
(déprécié par l'API, confirmé contre la doc officielle) et n'aurait de
toute façon jamais garanti un déterminisme parfait même sur les anciens
modèles -- la variance entre appels identiques est donc traitée comme un
fait à gérer, pas un bug à éliminer. CLAUDE_VOTE_CALLS (défaut 3) appels
sont faits par ticker ; si une majorité stricte se dégage, son verdict
est utilisé (verdict_source='llm_consensus') ; sinon le verdict retombe
sur 'uncertain' avec verdict_source='llm_disagreement', distinct d'un
'uncertain' qui viendrait d'un manque d'information. CLAUDE_VOTE_CALLS=1
désactive le vote (verdict_source='single_call', comportement
d'origine) -- utile pour limiter le coût si besoin.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal
import logging
import os

import anthropic

from .news import fetch_recent_news, format_news_for_prompt

logger = logging.getLogger(__name__)

CLAUDE_MODEL = os.environ.get("CLAUDE_ANALYSIS_MODEL", "claude-sonnet-5")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_VOTE_CALLS = max(1, int(os.environ.get("CLAUDE_VOTE_CALLS", "3")))

ClaudeVerdict = Literal["confirmed", "uncertain", "rejected"]
VerdictSource = Literal["single_call", "llm_consensus", "llm_disagreement"]

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
    verdict_source: VerdictSource = "single_call"
    # Verdicts individuels avant vote, pour audit/debug (vide en mode
    # single_call). Pas persisté tel quel en DB -- ScanResult ne garde que
    # le résultat agrégé, mais utile en log/tests.
    raw_votes: list = field(default_factory=list)


def _get_client() -> "anthropic.Anthropic":
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY manquante. Rappel : un abonnement Claude Pro ne "
            "donne pas de clé API — il faut un compte séparé sur console.anthropic.com."
        )
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _call_claude_once(client: "anthropic.Anthropic", ticker: str, user_message: str) -> ClaudeResult:
    """
    Un seul appel Claude pour le verdict. En cas d'erreur API, retourne
    'uncertain' par défaut plutôt que de faire planter le scan complet —
    un ticker/appel en échec ne doit pas bloquer le reste du run ni du vote.
    """
    try:
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
        # disponible ici ; voir verify_ticker() pour la gestion de la
        # variance résiduelle via vote majoritaire.
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
            "Claude _call_claude_once(%s): %d input + %d output = %d tokens -> %s",
            ticker, input_tokens, output_tokens, tokens_used, verdict,
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


def _majority_vote(results: list[ClaudeResult]) -> ClaudeResult:
    """
    Agrège N ClaudeResult (déjà obtenus) en un seul verdict.

    Majorité stricte (> moitié des appels) sur le même verdict -> ce
    verdict, verdict_source='llm_consensus', reasoning/confidence pris du
    premier appel qui matche le verdict majoritaire. Sinon (aucune
    majorité stricte, ex: 1-1-1 sur 3 appels) -> 'uncertain',
    verdict_source='llm_disagreement', reasoning résume le désaccord.
    tokens_used = somme des tokens de tous les appels (coût réel de
    l'étape pour ce ticker).
    """
    total_tokens = sum(r.tokens_used for r in results)
    verdicts = [r.verdict for r in results]
    counts = Counter(verdicts)
    top_verdict, top_count = counts.most_common(1)[0]

    if top_count > len(results) / 2:
        agreeing = next(r for r in results if r.verdict == top_verdict)
        return ClaudeResult(
            verdict=top_verdict,
            reasoning=agreeing.reasoning,
            confidence=agreeing.confidence,
            tokens_used=total_tokens,
            verdict_source="llm_consensus",
            raw_votes=verdicts,
        )

    summary = ", ".join(f"{v}={c}" for v, c in counts.most_common())
    return ClaudeResult(
        verdict="uncertain",
        reasoning=f"Désaccord entre {len(results)} appels Claude ({summary}) -- aucune majorité claire.",
        confidence="low",
        tokens_used=total_tokens,
        verdict_source="llm_disagreement",
        raw_votes=verdicts,
    )


def verify_ticker(ticker: str, quant_details: dict, deepseek_reasoning: str = "") -> ClaudeResult:
    """
    Verdict final via Claude, avec vote majoritaire sur CLAUDE_VOTE_CALLS
    appels (défaut 3, CLAUDE_VOTE_CALLS=1 désactive le vote). Le message
    utilisateur est construit une seule fois -- même contenu envoyé à
    chaque appel, seul le résultat du modèle peut varier.
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
    except RuntimeError as e:
        logger.exception("Erreur vérification Claude pour %s", ticker)
        return ClaudeResult(
            verdict="uncertain",
            reasoning=f"Erreur technique lors de l'appel Claude : {e}",
            confidence="low",
        )

    results = [_call_claude_once(client, ticker, user_message) for _ in range(CLAUDE_VOTE_CALLS)]

    if CLAUDE_VOTE_CALLS == 1:
        result = results[0]
        result.verdict_source = "single_call"
        return result

    return _majority_vote(results)
