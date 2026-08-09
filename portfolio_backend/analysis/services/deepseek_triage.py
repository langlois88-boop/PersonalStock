"""
Étape 2 du screener : tri local gratuit via Ollama (DeepSeek). Filtre les
cas évidents avant d'envoyer quoi que ce soit à Claude (économise le budget
API).

Intégration : réutilise OLLAMA_BASE_URL / OLLAMA_MODEL / OLLAMA_TIMEOUT —
les mêmes variables que portfolio/ai_advisor.py::DeepSeekAdvisor (même
instance Ollama locale sur le NAS/Mac, un seul modèle à gérer). Pas de
nouvelle variable OLLAMA_URL : DeepSeekAdvisor construit déjà l'endpoint
via f"{OLLAMA_BASE_URL}/api/generate", même convention ici.
"""

from dataclasses import dataclass
from typing import Literal
import json
import logging
import os

import requests

from .news import fetch_recent_news, format_news_for_prompt

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = (os.environ.get("OLLAMA_BASE_URL") or "").strip().rstrip("/")
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

    prompt = f"{SYSTEM_PROMPT}\n\nTicker: {ticker}\n\nNews récentes:\n{news_text}"

    if not OLLAMA_BASE_URL:
        logger.warning("OLLAMA_BASE_URL non configurée — triage DeepSeek sauté pour %s, fallback 'uncertain'.", ticker)
        return DeepSeekResult(
            verdict="uncertain",
            reasoning="OLLAMA_BASE_URL absente — passe à Claude par précaution.",
        )

    try:
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
        parsed = json.loads(raw)

        verdict = parsed.get("verdict", "uncertain")
        if verdict not in ("negative_reason", "no_reason", "uncertain"):
            verdict = "uncertain"

        return DeepSeekResult(
            verdict=verdict,
            reasoning=parsed.get("reasoning", ""),
            raw_response=raw,
        )
    except (requests.RequestException, json.JSONDecodeError, KeyError):
        logger.exception("Erreur triage DeepSeek pour %s — fallback 'uncertain'", ticker)
        return DeepSeekResult(
            verdict="uncertain",
            reasoning="Erreur technique lors du triage local — passe à Claude par précaution.",
        )
