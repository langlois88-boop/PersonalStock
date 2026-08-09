"""
Fetch des news récentes pour un ticker, via l'API news d'Alpaca.
Partagé entre l'étape 2 (DeepSeek) et l'étape 3 (Claude) pour éviter
de fetcher deux fois les mêmes news.

Intégration : ALPACA_API_KEY / ALPACA_SECRET_KEY sont les mêmes variables
que celles déjà utilisées par portfolio/alpaca_data.py pour les prix —
réutilisées telles quelles, pas de duplication de clés. Le repo n'a pas
encore de wrapper générique pour l'endpoint news Alpaca (le module news
existant, portfolio/management/commands/fetch_news_articles.py, fetch
NewsAPI en batch dans NewsArticle — différent de ce fetch à la demande),
donc ce client reste local à l'app analysis.
"""

from datetime import datetime, timedelta
import logging
import os

import requests

logger = logging.getLogger(__name__)

ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")


def fetch_recent_news(ticker: str, days_back: int = 60, limit: int = 10) -> list:
    """
    Retourne une liste de news récentes pour le ticker :
    [{"headline": str, "summary": str, "created_at": str, "source": str}, ...]
    Retourne une liste vide en cas d'erreur — ne doit jamais faire planter
    le pipeline en amont (mieux vaut un verdict "incertain" qu'un crash).
    """
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        logger.warning("Clés Alpaca absentes — vérifie ALPACA_API_KEY / ALPACA_SECRET_KEY dans l'environnement.")
        return []

    since = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }
    params = {
        "symbols": ticker,
        "start": since,
        "limit": limit,
        "sort": "desc",
    }

    try:
        resp = requests.get(ALPACA_NEWS_URL, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("news", [])
        return [
            {
                "headline": a.get("headline", ""),
                "summary": a.get("summary", ""),
                "created_at": a.get("created_at", ""),
                "source": a.get("source", ""),
            }
            for a in articles
        ]
    except requests.RequestException:
        logger.exception("Erreur fetch news Alpaca pour %s", ticker)
        return []


def format_news_for_prompt(news_items: list) -> str:
    """Formate les news en texte compact pour un prompt LLM."""
    if not news_items:
        return "Aucune news récente trouvée pour ce ticker."
    lines = []
    for item in news_items:
        date = item["created_at"][:10] if item["created_at"] else "?"
        lines.append(f"- [{date}] {item['headline']} — {item['summary'][:200]}")
    return "\n".join(lines)
