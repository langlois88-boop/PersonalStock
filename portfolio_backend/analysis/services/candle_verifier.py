"""
Commentaire Claude sur un pattern chandelier détecté (bouton "Interprète"
du graphique de prix, TickerDetail.js). Fonctionnalité distincte du
verdict fondamental (claude_verifier.py) -- ici pas de vote majoritaire
ni de décision confirmed/rejected, juste une interprétation courte et
factuelle du pattern dans son contexte, déclenchée manuellement par
l'utilisateur (pas d'appel automatique au chargement de la page, pour
garder le coût sous contrôle).

Réutilise les mêmes conventions déjà validées le 2026-08-10 dans
claude_verifier.py : tool_choice forcé (pas de JSON en texte libre),
pas de `temperature` (déprécié sur ce modèle), max_tokens=2000 (le
raisonnement étendu peut consommer le budget avant l'appel d'outil).
"""

import logging
import os

import anthropic

logger = logging.getLogger(__name__)

CLAUDE_MODEL = os.environ.get("CLAUDE_ANALYSIS_MODEL", "claude-sonnet-5")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

PATTERN_TOOL = {
    "name": "submit_pattern_commentary",
    "description": "Soumets ton interprétation du pattern chandelier détecté.",
    "input_schema": {
        "type": "object",
        "properties": {
            "commentary": {
                "type": "string",
                "description": "2-4 phrases expliquant ce que ce pattern signifie dans ce contexte précis (pas une recommandation d'achat/vente).",
            },
            "reliability": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Fiabilité du signal ici, compte tenu du volume relatif et du contexte fourni.",
            },
        },
        "required": ["commentary", "reliability"],
    },
}

SYSTEM_PROMPT = """Tu es un analyste technique qui commente un pattern de chandelier
japonais détecté automatiquement sur un graphique de prix. Ta tâche : expliquer
brièvement ce que ce pattern signifie généralement, et si le contexte fourni
(volume relatif, prix) renforce ou affaiblit sa fiabilité ici.

Sois factuel et pédagogique, pas promotionnel -- un pattern détecté n'est qu'un
signal statistique, pas une garantie. Ne donne jamais de recommandation d'achat/
vente explicite ; explique le pattern et son contexte, laisse la décision à
l'utilisateur. Utilise l'outil submit_pattern_commentary pour répondre."""


def _get_client() -> "anthropic.Anthropic":
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY manquante. Rappel : un abonnement Claude Pro ne "
            "donne pas de clé API — il faut un compte séparé sur console.anthropic.com."
        )
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def verify_pattern(
    ticker: str,
    pattern_text: str,
    price: float | None = None,
    rvol: float | None = None,
) -> dict:
    """
    Un seul appel Claude, pas de vote majoritaire (feature de commentaire,
    pas de décision binaire à fiabiliser comme le verdict fondamental).
    En cas d'échec API, renvoie un commentaire d'erreur clair plutôt que
    de faire planter l'appel -- c'est un bouton optionnel sur la page,
    pas une étape bloquante du pipeline.
    """
    try:
        client = _get_client()
        context_lines = [f"Ticker : {ticker}", f"Pattern détecté : {pattern_text}"]
        if price is not None:
            context_lines.append(f"Prix au moment du pattern : {price}")
        if rvol is not None:
            context_lines.append(f"Volume relatif (RVOL) : {rvol:.2f}x la moyenne")
        else:
            context_lines.append("Volume relatif : non disponible (historique insuffisant à ce point du graphique)")
        user_message = "\n".join(context_lines)

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            tools=[PATTERN_TOOL],
            tool_choice={"type": "tool", "name": "submit_pattern_commentary"},
        )

        tool_use_block = next(
            (block for block in response.content if block.type == "tool_use"), None
        )
        if tool_use_block is None:
            raise ValueError(
                f"Pas de bloc tool_use dans la réponse (stop_reason={response.stop_reason})"
            )
        parsed = tool_use_block.input

        input_tokens = response.usage.input_tokens or 0
        output_tokens = response.usage.output_tokens or 0
        logger.info(
            "Claude verify_pattern(%s, %s): %d input + %d output tokens",
            ticker, pattern_text, input_tokens, output_tokens,
        )

        return {
            "commentary": parsed.get("commentary", ""),
            "reliability": parsed.get("reliability", "low"),
            "tokens_used": input_tokens + output_tokens,
        }
    except Exception as exc:
        logger.exception("Échec verify_pattern pour %s / %s", ticker, pattern_text)
        return {
            "commentary": f"Impossible d'obtenir un commentaire pour l'instant ({exc}).",
            "reliability": "low",
            "tokens_used": 0,
            "error": True,
        }
