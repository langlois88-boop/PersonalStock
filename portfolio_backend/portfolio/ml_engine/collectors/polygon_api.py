"""Collecteur Polygon.io (2026-08-23) -- remplace fmp_api.py (fondamentaux)
et news_rss.py (sentiment) dans le pipeline du modèle "recommender"
(portfolio/ml_engine/recommender.py, recommender_training.py), après
découverte que :
- fmp_api.py tapait l'API FMP v3, morte pour tout le monde depuis le
  31 août 2025 (403 "Legacy Endpoint") -- fetch_fmp_fundamentals
  retournait {'roe': None, 'debt_to_equity': None} pour absolument tout
  ticker depuis cette date, silencieusement (jamais d'exception).
- Le sentiment FMP (fetch_fmp_sentiment) était de toute façon écrasé sans
  condition par le sentiment RSS/FinBERT dans processor.py::merge()
  (les deux calculaient une clé 'news_sentiment', le second toujours
  fusionné en dernier) -- mort une deuxième fois, indépendamment de FMP.

Polygon donne à la fois les fondamentaux (états financiers structurés,
XBRL) ET les actualités avec sentiment déjà calculé par LLM par article
('insights', positif/neutre/négatif + justification) -- une seule source
pour remplacer les deux anciennes, plus riche que le score FinBERT sur
titres RSS seuls (Polygon analyse l'article complet).

Vérifié en direct (2026-08-23) :
- AAPL : roe=119.91%, debt_to_equity=76.54% -- plausible (rachats
  d'actions massifs chez Apple, ROE structurellement très élevé).
- ONCY (biotech penny stock, capitaux propres négatifs) : le calcul
  brut net_income/equity donnait +607% (négatif/négatif) -- un piège
  classique, une entreprise en détresse aurait l'air d'avoir un ROE
  excellent. Gardé explicitement : equity <= 0 -> roe/debt_to_equity
  = None plutôt qu'un ratio trompeur.
- KEGS (micro-cap trop petite) : aucun état financier chez Polygon --
  dégrade vers None, jamais d'exception.
- Couverture actualités inégale sur les penny stocks (ONCY/GMNI ont des
  articles, mais souvent hors de la fenêtre de 7 jours ; SONI/BB/KEGS
  n'en ont aucun) -- attendu, même limite que l'ancien flux RSS pour ces
  titres peu suivis.

Complément yfinance (2026-08-23) -- Polygon est centré sur les marchés
américains : AUCUN état financier pour les titres TSX/CSE (RY.TO, ATD.TO,
PDN.TO, TEC.TO, SONI.CN...), vérifié en direct. yfinance les couvre bien
via `.info` (returnOnEquity, debtToEquity -- mêmes champs et mêmes
conventions d'unité que quant_filter.py::_evaluate_factor_investing dans
l'app analysis : debtToEquity déjà en points de pourcentage, PAS besoin
de multiplier par 100, contrairement à returnOnEquity qui est un ratio
0-1). fetch_fundamentals() ci-dessous essaie Polygon D'ABORD (meilleure
couverture US, cohérent avec le sentiment qui reste Polygon-only), et ne
complète qu'avec yfinance les champs encore manquants -- jamais l'usage
inverse, pour ne pas perdre le garde-fou equity<=0 de Polygon sur les
titres qu'il couvre déjà. Vérifié en direct : RY.TO/ATD.TO/PDN.TO/SONI.CN
ont un vrai ROE yfinance ; TEC.TO/VFV.TO(ETF)/QQQ/SPY(ETF) restent None
des deux côtés -- pas un bug, ces titres n'ont simplement pas de
fondamentaux d'entreprise chez l'un ou l'autre fournisseur.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import requests

from portfolio import market_data

BASE_URL = "https://api.polygon.io"

_SENTIMENT_LABEL_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def _get_json(url: str) -> Any:
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_polygon_fundamentals(symbol: str) -> Dict[str, Optional[float]]:
    """ROE et dette/équité (en points de pourcentage, même convention que
    quant_filter.py::_evaluate_factor_investing) via les états financiers
    TTM de Polygon. None si equity <= 0 -- un ratio net_income/equity avec
    équité négative inverse le signe (une perte devient un 'ROE positif'
    trompeur, vu en direct sur ONCY), donc explicitement écarté plutôt que
    silencieusement faux."""
    default: Dict[str, Optional[float]] = {"roe": None, "debt_to_equity": None}
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        return default

    symbol = (symbol or "").upper().strip()
    if not symbol:
        return default

    url = f"{BASE_URL}/vX/reference/financials?ticker={symbol}&limit=1&timeframe=ttm&apiKey={api_key}"
    data = _get_json(url) or {}
    results = data.get("results") or []
    if not results:
        return default

    financials = results[0].get("financials", {}) or {}
    balance_sheet = financials.get("balance_sheet", {}) or {}
    income_statement = financials.get("income_statement", {}) or {}

    equity = _safe_float((balance_sheet.get("equity") or {}).get("value"))
    if equity is None or equity <= 0:
        return default

    net_income_field = (
        income_statement.get("net_income_loss_attributable_to_parent")
        or income_statement.get("net_income_loss")
        or {}
    )
    net_income = _safe_float(net_income_field.get("value"))
    long_term_debt = _safe_float((balance_sheet.get("long_term_debt") or {}).get("value")) or 0.0

    roe = round(net_income / equity * 100, 2) if net_income is not None else None
    debt_to_equity = round(long_term_debt / equity * 100, 2)

    return {"roe": roe, "debt_to_equity": debt_to_equity}


def fetch_polygon_sentiment(symbol: str, days: int = 7) -> Dict[str, float]:
    """Sentiment moyen des actualités récentes (fenêtre de `days` jours) --
    même forme de retour que l'ancien news_rss.py::fetch_news_sentiment
    ({'news_sentiment': float, 'news_count': int}) pour un remplacement
    direct dans processor.py. Utilise les 'insights' par ticker déjà
    calculés par Polygon (positif/neutre/négatif par article, analysés sur
    le texte complet -- pas seulement le titre comme le RSS+FinBERT
    remplacé)."""
    default = {"news_sentiment": 0.0, "news_count": 0}
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        return default

    symbol = (symbol or "").upper().strip()
    if not symbol:
        return default

    url = (
        f"{BASE_URL}/v2/reference/news?ticker={symbol}&limit=50"
        f"&order=desc&sort=published_utc&apiKey={api_key}"
    )
    data = _get_json(url) or {}
    items = data.get("results") or []
    if not items:
        return default

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    scores: list[float] = []
    count = 0
    for item in items:
        published_utc = item.get("published_utc")
        try:
            published_dt = datetime.fromisoformat(published_utc.replace("Z", "+00:00")) if published_utc else None
        except Exception:
            published_dt = None
        if published_dt is not None and published_dt < cutoff:
            continue
        count += 1
        for insight in (item.get("insights") or []):
            if (insight.get("ticker") or "").upper() != symbol:
                continue
            score = _SENTIMENT_LABEL_SCORE.get((insight.get("sentiment") or "").lower())
            if score is not None:
                scores.append(score)

    if count == 0:
        return default

    avg_sentiment = round(sum(scores) / len(scores), 4) if scores else 0.0
    return {"news_sentiment": avg_sentiment, "news_count": count}


def fetch_yfinance_fundamentals(symbol: str) -> Dict[str, Optional[float]]:
    """Fallback pour les titres hors couverture Polygon (surtout TSX/CSE).
    Mêmes clés/unités que fetch_polygon_fundamentals -- debtToEquity de
    yfinance est déjà en points de pourcentage (vérifié en direct
    2026-08-23, cohérent avec quant_filter.py::fetch_ticker_data dans
    l'app analysis), returnOnEquity est un ratio 0-1 à multiplier par 100.
    Passe par portfolio.market_data.Ticker plutôt que yfinance brut pour
    hériter du timeout déjà en place sur ce wrapper."""
    default: Dict[str, Optional[float]] = {"roe": None, "debt_to_equity": None}
    symbol = (symbol or "").upper().strip()
    if not symbol:
        return default
    try:
        info = market_data.Ticker(symbol).info or {}
    except Exception:
        return default

    roe = _safe_float(info.get("returnOnEquity"))
    debt_to_equity = _safe_float(info.get("debtToEquity"))
    return {
        "roe": round(roe * 100, 2) if roe is not None else None,
        "debt_to_equity": round(debt_to_equity, 2) if debt_to_equity is not None else None,
    }


def fetch_fundamentals(symbol: str) -> Dict[str, Optional[float]]:
    """Point d'entrée combiné utilisé par processor.py et
    recommender_training.py -- Polygon d'abord (meilleure couverture US,
    garde-fou equity<=0 déjà appliqué), yfinance complète seulement les
    champs encore None (couverture TSX/CSE que Polygon n'a pas). Jamais
    l'inverse : un titre déjà résolu par Polygon garde ses valeurs,
    prioritaires sur le fallback."""
    result = fetch_polygon_fundamentals(symbol)
    if result.get("roe") is not None and result.get("debt_to_equity") is not None:
        return result
    fallback = fetch_yfinance_fundamentals(symbol)
    return {
        "roe": result.get("roe") if result.get("roe") is not None else fallback.get("roe"),
        "debt_to_equity": (
            result.get("debt_to_equity") if result.get("debt_to_equity") is not None else fallback.get("debt_to_equity")
        ),
    }
