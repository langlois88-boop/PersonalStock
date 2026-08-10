"""
Extraction robuste d'un objet JSON depuis une réponse de LLM.

Contexte (bug trouvé le 2026-08-09 lors du premier scan réel en prod) :
malgré une instruction explicite "réponds UNIQUEMENT en JSON", les modèles
(Claude comme DeepSeek-R1 via Ollama) enveloppent fréquemment leur réponse
dans un bloc markdown (```json ... ```) et/ou, pour les modèles à
raisonnement explicite comme DeepSeek-R1, dans un bloc <think>...</think>
avant la réponse finale (voir portfolio/ai_advisor.py::DeepSeekAdvisor qui
gère déjà ce cas pour le chat interactif). Un `json.loads()` naïf sur le
texte brut échoue alors avec `JSONDecodeError: Expecting value: line 1
column 1 (char 0)` — exactement l'erreur observée sur chaque ticket du
scan du 2026-08-09.

Cette fonction est volontairement défensive : elle ne suppose PAS que le
modèle a bien suivi l'instruction, elle nettoie ce qu'il faut avant de
parser.
"""

from __future__ import annotations

import json
import re

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json_object(raw_text: str) -> dict:
    """
    Nettoie et parse un texte de réponse LLM censé contenir un objet JSON.

    Retire, dans l'ordre : les blocs <think>...</think> (raisonnement
    explicite de certains modèles), puis les balises de bloc de code
    markdown (```json ... ``` ou ``` ... ```), puis parse le JSON restant.

    Lève json.JSONDecodeError si, même après nettoyage, le texte n'est
    toujours pas du JSON valide — le comportement de l'appelant sur
    l'erreur (fallback 'uncertain', logging) reste inchangé, seul ce qui
    compte comme "vraiment invalide" change.
    """
    text = (raw_text or "").strip()
    text = _THINK_BLOCK_RE.sub("", text).strip()
    text = _CODE_FENCE_RE.sub("", text).strip()
    return json.loads(text)
