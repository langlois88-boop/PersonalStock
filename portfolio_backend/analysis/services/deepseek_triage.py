"""
Étape 2 du screener : tri local gratuit via Ollama (DeepSeek). Filtre les
cas évidents avant d'envoyer quoi que ce soit à Claude (économise le budget
API).

Intégration : réutilise OLLAMA_BASE_URL / OLLAMA_CHAT_BASE_URL /
OLLAMA_CHAT_MODE / OLLAMA_MODEL / OLLAMA_TIMEOUT — les mêmes variables que
portfolio/ai_advisor.py::DeepSeekAdvisor (même instance Ollama/LocalAI sur
le NAS/Mac, un seul modèle à gérer).

Bug corrigé le 2026-08-09 : la version précédente de ce fichier prétendait
"réutiliser la même convention" que DeepSeekAdvisor mais appelait
toujours OLLAMA_BASE_URL + "/api/generate" (endpoint natif Ollama), sans
jamais vérifier chat_mode/chat_base_url. Sur le NAS, cette instance est
LocalAI avec OLLAMA_CHAT_MODE=1 — /api/generate y répond 404 (confirmé en
prod, chaque ticket du scan du 09/08 est tombé dans le fallback
'uncertain' à cause de ça), seul /v1/chat/completions fonctionne
réellement. On réplique donc ici exactement la branche que
DeepSeekAdvisor.stream_answer utilise déjà (voir ai_advisor.py:233-234).
"""

from dataclasses import dataclass
from typing import Literal
import json
import logging
import os

import requests

from .llm_json import extract_json_object
from .news import fetch_recent_news, format_news_for_prompt

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = (os.environ.get("OLLAMA_BASE_URL") or "").strip().rstrip("/")
OLLAMA_CHAT_BASE_URL = (os.environ.get("OLLAMA_CHAT_BASE_URL") or OLLAMA_BASE_URL).strip().rstrip("/")
OLLAMA_CHAT_MODE = str(os.environ.get("OLLAMA_CHAT_MODE", "")).strip().lower() in {"1", "true", "yes", "y"}
OLLAMA_MODEL = (os.environ.get("OLLAMA_MODEL") or "deepseek-r1:8b").strip()
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "30"))

DeepSeekVerdict = Literal["negative_reason", "no_reason", "uncertain"]

SYSTEM_PROMPT = """Tu es un premier filtre de triage pour un screener d'actions sous-évaluées.
Ta tâche est simple : lire les news récentes d'un ticker et déterminer s'il existe une
RAISON NÉGATIVE ÉVIDENTE ET SPÉCIFIQUE À L'ENTREPRISE qui expliquerait une baisse de prix
(fraude, poursuite majeure, guidance coupée, perte de contrat majeur, scandale de direction,
faillite imminente).

Ne compte PAS comme raison négative : mouvement macro général, rotation sectorielle,
volatilité de marché sans lien avec l'entreprise, absence de news.

Réponds UNIQUEMENT en JSON, rien d'autre, format exact :
{"verdict": "negative_reason" | "no_reason" | "uncertain", "reasoning": "une phrase courte"}

Utilise "uncertain" si les news sont insuffisantes, ambiguës, ou si tu n'es pas sûr."""


@dataclass
class DeepSeekResult:
    verdict: DeepSeekVerdict
    reasoning: str
    raw_response: str = ""


def triage_ticker(ticker: str) -> DeepSeekResult:
    """
    Fetch les news et demande à Ollama/DeepSeek un premier verdict rapide.
    En cas d'erreur (Ollama down/pas configuré, timeout, réponse malformée),
    retourne "uncertain" par défaut — on ne veut jamais bloquer le pipeline
    ni rejeter à tort faute de pouvoir juger.
    """
    news_items = fetch_recent_news(ticker, days_back=60)
    news_text = format_news_for_prompt(news_items)

    if not OLLAMA_BASE_URL and not OLLAMA_CHAT_BASE_URL:
        logger.warning("OLLAMA_BASE_URL non configurée — triage DeepSeek sauté pour %s, fallback 'uncertain'.", ticker)
        return DeepSeekResult(
            verdict="uncertain",
            reasoning="OLLAMA_BASE_URL absente — passe à Claude par précaution.",
        )

    try:
        if OLLAMA_CHAT_MODE or "/v1" in OLLAMA_CHAT_BASE_URL:
            # stream=False ne répond jamais sur cette instance LocalAI
            # (confirmé en prod le 2026-08-09 : HTTP 000 après 15s en curl
            # direct, alors que stream=True répond immédiatement). On suit
            # donc le même chemin que ai_advisor.py::DeepSeekAdvisor.stream_answer
            # (déjà en stream=True), en consommant tout le flux SSE d'un
            # coup plutôt qu'en l'affichant au fur et à mesure — ici on n'a
            # besoin que du texte final, pas d'un affichage temps réel.
            resp = requests.post(
                f"{OLLAMA_CHAT_BASE_URL}/chat/completions",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Ticker: {ticker}\n\nNews récentes:\n{news_text}"},
                    ],
                    "stream": True,
                },
                stream=True,
                timeout=OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            chunks = []
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = (json.loads(data).get("choices") or [{}])[0].get("delta") or {}
                except ValueError:
                    continue
                chunks.append(delta.get("content") or "")
            raw = "".join(chunks)
        else:
            prompt = f"{SYSTEM_PROMPT}\n\nTicker: {ticker}\n\nNews récentes:\n{news_text}"
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
                timeout=OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")

        parsed = extract_json_object(raw)

        verdict = parsed.get("verdict", "uncertain")
        if verdict not in ("negative_reason", "no_reason", "uncertain"):
            verdict = "uncertain"

        return DeepSeekResult(
            verdict=verdict,
            reasoning=parsed.get("reasoning", ""),
            raw_response=raw,
        )
    except (requests.RequestException, ValueError, KeyError):
        logger.exception("Erreur triage DeepSeek pour %s — fallback 'uncertain'", ticker)
        return DeepSeekResult(
            verdict="uncertain",
            reasoning="Erreur technique lors du triage local — passe à Claude par précaution.",
        )
