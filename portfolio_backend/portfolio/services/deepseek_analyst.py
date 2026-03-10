from __future__ import annotations

"""
deepseek_analyst.py
===================
Prompts spécialisés et riches pour DeepSeek/Ollama.

Couvre :
- Analyse dip bluechip « injustifié » (bonne compagnie, bon cashflow, prête à remonter)
- Swing trading penny stock  (entrée, cible, stop-loss)
- Revue de portfolio + suggestions d'ajout
- Analyse complète d'un ticker (entry/exit/stop + conviction)
- Pondération de samples d'entraînement (confidence scoring)
"""

import json
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Iterator

import requests

logger = logging.getLogger(__name__)

_CONVICTION_RANK = {
    "FAIBLE": 1,
    "MODÉRÉE": 2,
    "MODEREE": 2,
    "FORTE": 3,
}


def _conviction_rank(value: str) -> int:
    if not value:
        return 0
    key = value.strip().upper()
    return _CONVICTION_RANK.get(key, 0)


def _min_conviction_from_env(key: str, default: str = "FAIBLE") -> int:
    raw = os.getenv(key, default)
    return _conviction_rank(raw or default)


# ─── Configuration ────────────────────────────────────────────────────────────

def _base_url() -> str:
    url = (
        os.getenv("OLLAMA_CHAT_BASE_URL")
        or os.getenv("OLLAMA_BASE_URL")
        or "http://localhost:11434"
    ).strip().rstrip("/")
    if "/v1" not in url:
        url = f"{url}/v1"
    return url


def _model() -> str:
    return os.getenv("OLLAMA_MODEL", "deepseek-r1:8b").strip()


def _timeout() -> int:
    return int(os.getenv("OLLAMA_TIMEOUT", "120"))


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class SwingLevels:
    entry: float
    target: float
    stop_loss: float
    risk_reward: float
    atr: float
    conviction: str          # "FORTE" | "MODÉRÉE" | "FAIBLE"
    reasoning: str
    horizon_days: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry": round(self.entry, 4),
            "target": round(self.target, 4),
            "stop_loss": round(self.stop_loss, 4),
            "risk_reward": round(self.risk_reward, 2),
            "atr": round(self.atr, 4),
            "conviction": self.conviction,
            "reasoning": self.reasoning,
            "horizon_days": self.horizon_days,
        }


@dataclass
class DipAnalysis:
    symbol: str
    is_unjustified: bool        # True = dip injustifié = opportunité d'achat
    conviction: str             # "FORTE" | "MODÉRÉE" | "FAIBLE"
    dip_reason: str             # Raison du dip selon le marché
    counter_argument: str       # Pourquoi c'est injustifié
    entry_zone: tuple[float, float]
    target_price: float
    stop_loss: float
    risk_reward: float
    reasoning: str
    watchlist_priority: int = 50   # 0-100

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "is_unjustified_dip": self.is_unjustified,
            "conviction": self.conviction,
            "dip_reason": self.dip_reason,
            "counter_argument": self.counter_argument,
            "entry_zone_low": round(self.entry_zone[0], 4),
            "entry_zone_high": round(self.entry_zone[1], 4),
            "target_price": round(self.target_price, 4),
            "stop_loss": round(self.stop_loss, 4),
            "risk_reward": round(self.risk_reward, 2),
            "reasoning": self.reasoning,
            "watchlist_priority": self.watchlist_priority,
        }


@dataclass
class PortfolioRecommendation:
    action: str               # "ADD" | "TRIM" | "HOLD" | "SELL"
    symbol: str
    reason: str
    confidence: float         # 0.0–1.0
    suggested_weight_pct: float | None = None
    universe: str = "BLUECHIP"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "symbol": self.symbol,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "suggested_weight_pct": self.suggested_weight_pct,
            "universe": self.universe,
        }


# ─── Core HTTP helper ─────────────────────────────────────────────────────────

def _chat(
    system: str,
    user: str,
    temperature: float = 0.15,
    max_tokens: int = 900,
    stream: bool = False,
) -> str | Iterator[str]:
    """Appel générique à l'API OpenAI-compatible (Ollama/DeepSeek)."""
    url = f"{_base_url()}/chat/completions"
    payload = {
        "model": _model(),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    try:
        resp = requests.post(url, json=payload, timeout=_timeout(), stream=stream)
        resp.raise_for_status()
        if stream:
            def _gen() -> Iterator[str]:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    text = line.decode("utf-8", errors="replace")
                    if text.startswith("data: "):
                        text = text[6:]
                    if text.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(text)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except Exception:
                        continue
            return _gen()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning("DeepSeek call failed: %s", exc)
        return "" if not stream else iter([])


def _parse_json_block(text: str) -> dict | None:
    """Extrait le premier bloc JSON d'une réponse DeepSeek."""
    import re
    # Remove <think>...</think> blocks from DeepSeek-R1
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Try to find a JSON block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        raw = match.group(1)
    else:
        # Find first { ... } block
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        raw = text[start:end + 1]
    try:
        return json.loads(raw)
    except Exception:
        return None


# ─── System personas ──────────────────────────────────────────────────────────

_SYSTEM_QUANT = (
    "Tu es un quant trader senior avec 15 ans d'expérience en gestion de portefeuille institutionnel "
    "et trading algorithmique. Tu maîtrises l'analyse technique (RSI, MACD, Bollinger, ATR, Fibonacci), "
    "l'analyse fondamentale (FCF, ROE, D/E, Altman-Z), et la gestion du risque. "
    "Tu réponds TOUJOURS en français, de façon concise, factuelle et structurée. "
    "Tes réponses JSON sont strictement valides, sans texte hors du bloc JSON demandé."
)

_SYSTEM_DIP_HUNTER = (
    "Tu es un chasseur de dips institutionnel spécialisé dans l'identification des corrections injustifiées "
    "sur des compagnies de qualité (bon cashflow, ROE élevé, bilan solide). "
    "Ton rôle est de distinguer un dip structurel d'un dip émotionnel/temporaire sur lequel acheter. "
    "Tu réponds en français avec un JSON structuré."
)

_SYSTEM_SWING = (
    "Tu es un swing trader professionnel spécialisé sur les penny stocks et small caps. "
    "Tu utilises l'ATR, le rubber band index, les supports/résistances et le momentum volumique "
    "pour calculer des niveaux d'entrée, de cible et de stop-loss précis. "
    "Chaque trade a un R:R minimum de 2.0. Tu réponds en français avec un JSON structuré."
)

_SYSTEM_PORTFOLIO = (
    "Tu es un gestionnaire de portefeuille senior qui analyse la diversification, l'exposition sectorielle, "
    "la corrélation entre actifs, le momentum et le risque global. "
    "Tu fournis des recommandations concrètes (ADD/TRIM/HOLD/SELL) avec une logique claire. "
    "Tu réponds en français avec un JSON structuré."
)


# ─── 1. Analyse dip bluechip injustifié ───────────────────────────────────────

def analyze_bluechip_dip(
    symbol: str,
    current_price: float,
    price_52w_high: float,
    dip_pct: float,           # e.g. -0.18 = -18%
    rsi: float,
    macd_hist: float,
    volume_z: float,
    fundamentals: dict[str, Any],   # fcf, roe, debt_to_equity, eps_growth, sector
    news_headlines: list[str] | None = None,
    atr: float | None = None,
) -> DipAnalysis | None:
    """
    Analyse si le dip d'un bluechip est injustifié.

    Un dip injustifié = bonne compagnie, bon cashflow, correction émotionnelle/de marché.
    Retourne un DipAnalysis avec entrée, cible, stop-loss.
    """
    fcf = fundamentals.get("fcf_yield", "N/A")
    roe = fundamentals.get("roe", "N/A")
    de = fundamentals.get("debt_to_equity", "N/A")
    sector = fundamentals.get("sector", "N/A")
    eps_growth = fundamentals.get("eps_growth", "N/A")
    altman_z = fundamentals.get("altman_z", "N/A")

    headlines_text = "\n".join(f"- {h}" for h in (news_headlines or [])[:5]) or "Aucune news récente"

    # Calcul technique de base pour l'entrée
    atr_val = atr or (current_price * 0.02)
    support_1 = round(current_price * 0.98, 4)
    support_2 = round(current_price * 0.95, 4)
    target_base = round(current_price * 1.08, 4)
    stop_base = round(current_price - 2.0 * atr_val, 4)
    rr = round((target_base - current_price) / max(current_price - stop_base, 0.001), 2)

    user_prompt = f"""
Analyse ce dip sur le bluechip **{symbol}** :

## Données de prix
- Prix actuel : {current_price:.4f}
- Sommet 52 semaines : {price_52w_high:.4f}
- Correction depuis le sommet : {dip_pct*100:.1f}%
- ATR (14) : {atr_val:.4f}

## Indicateurs techniques
- RSI(14) : {rsi:.1f}
- MACD Histogramme : {macd_hist:.4f}
- Volume Z-score : {volume_z:.2f}

## Fondamentaux
- Secteur : {sector}
- FCF Yield : {fcf}
- ROE : {roe}
- D/E Ratio : {de}
- Croissance EPS : {eps_growth}
- Altman Z-Score : {altman_z}

## News récentes
{headlines_text}

## Niveaux techniques calculés (à valider/ajuster)
- Support 1 : {support_1}
- Support 2 : {support_2}
- Cible initiale : {target_base}
- Stop-loss initial : {stop_base}
- R:R initial : {rr}

Réponds UNIQUEMENT avec ce JSON (pas de texte hors JSON) :
{{
  "is_unjustified_dip": true/false,
  "conviction": "FORTE"|"MODÉRÉE"|"FAIBLE",
  "dip_reason": "raison probable du dip selon le marché",
  "counter_argument": "pourquoi c'est injustifié (ou justifié)",
  "entry_zone_low": <float>,
  "entry_zone_high": <float>,
  "target_price": <float>,
  "stop_loss": <float>,
  "risk_reward": <float>,
  "watchlist_priority": <int 0-100>,
  "reasoning": "analyse complète en 3-4 phrases"
}}
"""

    text = _chat(_SYSTEM_DIP_HUNTER, user_prompt, temperature=0.1, max_tokens=700)
    if not isinstance(text, str):
        return None
    data = _parse_json_block(text)
    if not data:
        logger.warning("DipAnalysis: JSON parse failed for %s", symbol)
        return None

    try:
        result = DipAnalysis(
            symbol=symbol,
            is_unjustified=bool(data.get("is_unjustified_dip", False)),
            conviction=str(data.get("conviction", "FAIBLE")),
            dip_reason=str(data.get("dip_reason", "")),
            counter_argument=str(data.get("counter_argument", "")),
            entry_zone=(
                float(data.get("entry_zone_low", support_2)),
                float(data.get("entry_zone_high", support_1)),
            ),
            target_price=float(data.get("target_price", target_base)),
            stop_loss=float(data.get("stop_loss", stop_base)),
            risk_reward=float(data.get("risk_reward", rr)),
            reasoning=str(data.get("reasoning", "")),
            watchlist_priority=int(data.get("watchlist_priority", 50)),
        )
        min_rank = _min_conviction_from_env("DEEPSEEK_DIP_MIN_CONVICTION", "FAIBLE")
        if _conviction_rank(result.conviction) < min_rank:
            result.is_unjustified = False
            result.watchlist_priority = 0
            result.reasoning = f"{result.reasoning} (Conviction insuffisante)".strip()
        return result
    except Exception as exc:
        logger.warning("DipAnalysis: dataclass build failed: %s", exc)
        return None
    try:
        result = SwingLevels(
# ─── 2. Swing trading penny stock ─────────────────────────────────────────────

def analyze_penny_swing(
    symbol: str,
    current_price: float,
    atr: float,
    rsi: float,
    rubber_band_index: float,   # distance normalisée de la SMA20 (ex: -1.8 = très oversold)
    volume_z: float,
        min_rr = float(os.getenv("DEEPSEEK_SWING_MIN_RR", "2.0"))
        if result.risk_reward < min_rr:
            return None
        return result
    rvol: float,
    macd_hist: float,
    support_levels: list[float],
    resistance_levels: list[float],
    sentiment_score: float = 0.0,
    altman_z: float | None = None,
    news_headlines: list[str] | None = None,
) -> SwingLevels | None:
    """
    Calcule les niveaux de swing trading pour un penny stock.

    Cible : le rubber band est comprimé et prêt à remonter (snapback).
    Retourne entrée, cible, stop-loss avec R:R >= 2.0.
    """
    headlines_text = "\n".join(f"- {h}" for h in (news_headlines or [])[:4]) or "Aucune"

    # Niveaux par défaut basés sur ATR
    atr_mult_stop = float(os.getenv("AI_PENNY_ATR_MULT", "1.5"))
    stop_default = round(current_price - atr_mult_stop * atr, 4)
    target_default = round(current_price + 3.0 * atr_mult_stop * atr, 4)

    # Utiliser le premier support/résistance connus
    near_support = min(support_levels, default=stop_default, key=lambda x: abs(x - current_price) if x < current_price else float("inf"))
    near_resist = min(resistance_levels, default=target_default, key=lambda x: abs(x - current_price) if x > current_price else float("inf"))

    supports_str = ", ".join(f"{s:.4f}" for s in support_levels[:3]) or "N/A"
    resistances_str = ", ".join(f"{r:.4f}" for r in resistance_levels[:3]) or "N/A"

    user_prompt = f"""
Analyse ce penny stock pour un trade swing :

## Ticker : {symbol}
- Prix actuel : {current_price:.4f}
- ATR(14) : {atr:.4f}

## Indicateurs techniques
- RSI(14) : {rsi:.1f}  (< 35 = oversold intéressant)
- Rubber Band Index : {rubber_band_index:.2f}  (-1.0 = SMA - 1 écart-type, -2.0 = extrême oversold)
- Volume Z-score : {volume_z:.2f}
- RVOL : {rvol:.2f}
- MACD Histogramme : {macd_hist:.4f}

## Niveaux de marché
- Supports proches : {supports_str}
- Résistances proches : {resistances_str}
- Support calculé ATR : {near_support:.4f}
- Résistance calculée ATR : {near_resist:.4f}

## Contexte fondamental
- Sentiment news : {sentiment_score:.2f} (-1 à +1)
- Altman Z : {altman_z if altman_z is not None else "N/A"}

## News récentes
{headlines_text}

Calcule les niveaux optimaux pour un swing de 3–7 jours.
Le R:R MINIMUM est 2.0. Si impossible, indique conviction=FAIBLE.

Réponds UNIQUEMENT avec ce JSON :
{{
  "entry": <float>,
  "target": <float>,
  "stop_loss": <float>,
  "risk_reward": <float>,
  "horizon_days": <int>,
  "conviction": "FORTE"|"MODÉRÉE"|"FAIBLE",
  "reasoning": "analyse en 3-4 phrases: pourquoi le rubber band va se détendre, catalyseur, risque"
}}
"""

    text = _chat(_SYSTEM_SWING, user_prompt, temperature=0.1, max_tokens=600)
    if not isinstance(text, str):
        return None
    data = _parse_json_block(text)
    if not data:
        logger.warning("SwingLevels: JSON parse failed for %s", symbol)
        return None

    try:
        entry = float(data.get("entry", current_price))
        target = float(data.get("target", target_default))
        stop = float(data.get("stop_loss", stop_default))
        rr = round((target - entry) / max(entry - stop, 1e-6), 2)
        return SwingLevels(
            entry=entry,
            target=target,
            stop_loss=stop,
            risk_reward=float(data.get("risk_reward", rr)),
            atr=atr,
            conviction=str(data.get("conviction", "FAIBLE")),
            reasoning=str(data.get("reasoning", "")),
            horizon_days=int(data.get("horizon_days", 5)),
        )
    except Exception as exc:
        logger.warning("SwingLevels: dataclass build failed: %s", exc)
        return None


# ─── 3. Analyse complète d'un ticker ─────────────────────────────────────────

def analyze_ticker_full(
    symbol: str,
    price: float,
    universe: str,          # "BLUECHIP" | "PENNY" | "ETF" | "CRYPTO"
    ml_score: float,        # score du modèle sklearn (0.0–1.0)
    rsi: float,
    macd_hist: float,
    volume_z: float,
    rvol: float,
    atr: float,
    ma20: float,
    ma50: float,
    ma200: float | None,
    bollinger_pct: float,   # position dans les bandes de Bollinger (0–1)
    adx: float | None = None,
    stochastic_k: float | None = None,
    sentiment: float = 0.0,
    news_headlines: list[str] | None = None,
    fundamentals: dict[str, Any] | None = None,
    stream: bool = False,
) -> str | Iterator[str]:
    """
    Analyse complète d'un ticker avec recommandation entrée/sortie/stop.
    Peut streamer la réponse (stream=True).
    """
    headlines_text = "\n".join(f"- {h}" for h in (news_headlines or [])[:5]) or "Aucune"
    fund = fundamentals or {}
    adx_str = f"{adx:.1f}" if adx is not None else "N/A"
    stoch_str = f"{stochastic_k:.1f}" if stochastic_k is not None else "N/A"

    # Calcul ATR-based pour avoir une base
    stop_atr = round(price - 1.5 * atr, 4)
    target_atr = round(price + 3.0 * atr, 4)

    user_prompt = f"""
## Analyse complète — {symbol} ({universe})
**Date d'analyse :** Aujourd'hui | **ML Score :** {ml_score:.2%}

### Prix & tendance
| Indicateur | Valeur |
|---|---|
| Prix actuel | {price:.4f} |
| MA20 | {ma20:.4f} | MA50 | {ma50:.4f} | MA200 | {f"{ma200:.4f}" if ma200 else "N/A"} |
| Prix vs MA20 | {"DESSUS ↑" if price > ma20 else "DESSOUS ↓"} |
| Prix vs MA50 | {"DESSUS ↑" if price > ma50 else "DESSOUS ↓"} |

### Momentum & oscillateurs
| Indicateur | Valeur | Signal |
|---|---|---|
| RSI(14) | {rsi:.1f} | {"Oversold 🟢" if rsi < 35 else "Overbought 🔴" if rsi > 70 else "Neutre ⚪"} |
| MACD Hist | {macd_hist:.4f} | {"Haussier 🟢" if macd_hist > 0 else "Baissier 🔴"} |
| Bollinger %B | {bollinger_pct:.2f} | {"Bas de bande 🟢" if bollinger_pct < 0.2 else "Haut de bande 🔴" if bollinger_pct > 0.8 else "Mid ⚪"} |
| ADX | {adx_str} | {"Tendance forte" if adx and adx > 25 else "Pas de tendance"} |
| Stochastic %K | {stoch_str} |
| Volume Z | {volume_z:.2f} | {"Volume élevé 📊" if volume_z > 1.5 else "Normal"} |
| RVOL | {rvol:.2f} |
| ATR(14) | {atr:.4f} |

### Fondamentaux
- FCF Yield : {fund.get('fcf_yield', 'N/A')}
- ROE : {fund.get('roe', 'N/A')}
- D/E : {fund.get('debt_to_equity', 'N/A')}
- Altman Z : {fund.get('altman_z', 'N/A')}
- Secteur : {fund.get('sector', 'N/A')}

### Sentiment & News
- Score de sentiment : {sentiment:.2f}
{headlines_text}

### Niveaux ATR de base
- Stop-loss (1.5x ATR) : {stop_atr}
- Cible (3.0x ATR) : {target_atr}
- R:R estimé : {round((target_atr - price) / max(price - stop_atr, 1e-6), 2)}

---
Fournis une analyse structurée avec :
1. **Signal** (ACHAT FORT / ACHAT / NEUTRE / VENTE / VENTE FORTE)
2. **Prix d'entrée optimal** (avec justification technique)
3. **Prix cible** (1er objectif + 2e objectif si R:R le permet)
4. **Stop-loss** (avec justification ATR/support)
5. **Horizon recommandé** (swing court, swing moyen, position)
6. **Risques principaux**
7. **Conclusion** (2–3 phrases max)

Réponds en français, clair et actionnable.
"""

    return _chat(
        _SYSTEM_QUANT,
        user_prompt,
        temperature=0.15,
        max_tokens=1000,
        stream=stream,
    )


# ─── 4. Recommandations de portfolio ─────────────────────────────────────────

def analyze_portfolio_recommendations(
    holdings: list[dict[str, Any]],         # [{symbol, shares, weight_pct, pnl_pct, sector, ml_score, rsi}]
    sandbox_stats: dict[str, Any] | None,   # stats de performance des sandboxes
    candidate_additions: list[dict[str, Any]] | None = None,  # [{symbol, ml_score, sector}]
    total_value: float | None = None,
    question: str | None = None,
) -> dict[str, Any]:
    """
    Analyse le portfolio actuel et génère des recommandations d'ajout/réduction.
    Retourne un dict structuré avec actions prioritaires.
    """
    holdings_str = json.dumps(holdings, ensure_ascii=False, indent=2)
    candidates_str = json.dumps(candidate_additions or [], ensure_ascii=False, indent=2)
    sandbox_str = json.dumps(sandbox_stats or {}, ensure_ascii=False, indent=2)
    question_str = f"\n**Question spécifique :** {question}" if question else ""

    user_prompt = f"""
## Revue de Portfolio PersonalStock
**Valeur totale :** {f"${total_value:,.2f}" if total_value else "N/A"}

### Positions actuelles
```json
{holdings_str}
```

### Candidats d'ajout potentiels (top ML scores)
```json
{candidates_str}
```

### Performance des sandboxes (paper trading)
```json
{sandbox_str}
```
{question_str}

Analyse le portefeuille et retourne UNIQUEMENT ce JSON :
{{
  "portfolio_score": <int 0-100>,
  "diversification_rating": "BONNE"|"MOYENNE"|"FAIBLE",
  "sector_concentration_risk": "description du risque sectoriel si applicable",
  "recommendations": [
    {{
      "action": "ADD"|"TRIM"|"HOLD"|"SELL",
      "symbol": "<ticker>",
      "reason": "<raison concise>",
      "confidence": <float 0-1>,
      "suggested_weight_pct": <float ou null>,
      "universe": "BLUECHIP"|"PENNY"|"ETF"|"CRYPTO",
      "urgency": "HAUTE"|"MOYENNE"|"BASSE"
    }}
  ],
  "top_additions": [
    {{
      "symbol": "<ticker>",
      "rationale": "<pourquoi ajouter>",
      "universe": "BLUECHIP"|"PENNY",
      "conviction": "FORTE"|"MODÉRÉE"|"FAIBLE"
    }}
  ],
  "risk_alerts": ["<alerte 1>", "<alerte 2>"],
  "summary": "<3-4 phrases de synthèse>"
}}
"""

    text = _chat(_SYSTEM_PORTFOLIO, user_prompt, temperature=0.15, max_tokens=1200)
    if not isinstance(text, str):
        return {"error": "DeepSeek indisponible"}
    data = _parse_json_block(text)
    if not data:
        return {"error": "Parsing JSON échoué", "raw": text[:500]}
    return data


# ─── 5. Confidence scoring amélioré (training weighter) ──────────────────────

def get_training_confidence(
    symbol: str,
    date: str,
    features: dict[str, float],
    raw_label: int,
    universe: str = "BLUECHIP",
) -> float:
    """
    Demande à DeepSeek d'évaluer la confiance qu'un label d'entraînement est correct.
    Retourne un float 0.0–1.0 utilisé comme sample_weight.

    Amélioré par rapport à l'ancienne version :
    - Prompt plus riche (Bollinger, ADX, MACD)
    - Contexte univers (penny vs bluechip)
    - Parse robuste avec fallback
    """
    if not (os.getenv("DEEPSEEK_WEIGHTING_ENABLED") or "").lower() in {"1", "true", "yes", "y"}:
        return 1.0

    rsi = features.get("rsi_14", features.get("RSI14", 50.0))
    sma_r = features.get("sma_ratio_10_50", features.get("sma_ratio_10_20", 1.0))
    vol_z = features.get("volume_zscore_20", features.get("VolumeZ", 0.0))
    ret = features.get("return_20d", features.get("return_5d", features.get("Momentum20", 0.0)))
    sent = features.get("sentiment_score", 0.0)
    macd_h = features.get("MACD_HIST", features.get("macd_hist", 0.0))
    bb_pct = features.get("bollinger_pct_b", 0.5)
    adx = features.get("adx_14", features.get("ADX14", None))
    vol_regime = features.get("vol_regime", 1.0)

    label_str = "HAUSSE (+)" if raw_label == 1 else "BAISSE (-)"
    adx_str = f"{adx:.1f}" if adx is not None else "N/A"

    prompt = f"""
Univers : {universe} | Ticker : {symbol} | Date : {date}
Label brut : {label_str} (1=hausse attendue, 0=pas de hausse)

Indicateurs :
- RSI(14) = {rsi:.1f}
- SMA Ratio (fast/slow) = {sma_r:.3f}
- Volume Z-score = {vol_z:.2f}
- Retour momentum = {ret:.3f}
- MACD Histogramme = {macd_h:.4f}
- Bollinger %B = {bb_pct:.2f}
- ADX(14) = {adx_str}
- Régime de volatilité = {vol_regime:.2f}
- Sentiment news = {sent:.2f}

Question : Ces indicateurs sont-ils COHÉRENTS avec le label "{label_str}" ?
Réponds UNIQUEMENT avec un nombre entre 0.0 et 1.0 (ex: 0.85).
0.0 = label incohérent avec les indicateurs
1.0 = label très cohérent avec les indicateurs
"""

    text = _chat(
        "Tu es un validateur de labels de machine learning pour la finance quantitative. "
        "Tu réponds UNIQUEMENT avec un nombre float entre 0.0 et 1.0, sans aucun texte supplémentaire.",
        prompt,
        temperature=0.0,
        max_tokens=10,
    )
    if not isinstance(text, str):
        return 1.0
    try:
        import re
        nums = re.findall(r"0?\.\d+|\d+\.\d*", text.strip())
        if nums:
            val = float(nums[0])
            return max(0.0, min(1.0, val))
    except Exception:
        pass
    return 1.0


# ─── 6. Consensus multi-signal ────────────────────────────────────────────────

def build_consensus(
    symbol: str,
    ml_signal: float,           # 0.0–1.0 du modèle sklearn
    rsi: float,
    macd_hist: float,
    bollinger_pct: float,
    adx: float | None,
    volume_z: float,
    rubber_band: float | None,  # pour penny
    sentiment: float,
    universe: str = "BLUECHIP",
) -> dict[str, Any]:
    """
    Construit un consensus rapide entre le modèle ML et les indicateurs techniques.
    Retourne un score composite et une recommandation.
    """
    # Score technique normalisé (0.0–1.0)
    tech_score = 0.0
    weight_total = 0.0

    # RSI (oversold = bullish)
    rsi_score = 1.0 - min(max((rsi - 30) / 40.0, 0.0), 1.0)
    tech_score += rsi_score * 0.20
    weight_total += 0.20

    # MACD
    macd_score = 1.0 if macd_hist > 0 else 0.3
    tech_score += macd_score * 0.15
    weight_total += 0.15

    # Bollinger %B (bas = bullish)
    bb_score = 1.0 - min(max(bollinger_pct, 0.0), 1.0)
    tech_score += bb_score * 0.15
    weight_total += 0.15

    # Volume Z (élevé = confirmation)
    vol_score = min(max(volume_z / 3.0 + 0.5, 0.0), 1.0)
    tech_score += vol_score * 0.10
    weight_total += 0.10

    # ADX (tendance forte = plus fiable)
    if adx is not None:
        adx_mult = min(adx / 50.0, 1.0)
        tech_score += adx_mult * 0.10
        weight_total += 0.10

    # Rubber band (penny specific — très négatif = élastique prêt à snapper)
    if rubber_band is not None and universe in {"PENNY", "AI_PENNY"}:
        rbi_score = 1.0 - min(max((rubber_band + 2.0) / 4.0, 0.0), 1.0)
        tech_score += rbi_score * 0.15
        weight_total += 0.15

    # Sentiment
    sent_score = min(max((sentiment + 1.0) / 2.0, 0.0), 1.0)
    tech_score += sent_score * 0.15
    weight_total += 0.15

    # Normalise
    if weight_total > 0:
        tech_score /= weight_total

    # Composite : 60% ML, 40% technique
    composite = ml_signal * 0.60 + tech_score * 0.40
    composite = round(composite, 4)

    if composite >= 0.80:
        signal = "ACHAT FORT"
        signal_color = "🟢"
    elif composite >= 0.65:
        signal = "ACHAT"
        signal_color = "🟡"
    elif composite >= 0.45:
        signal = "NEUTRE"
        signal_color = "⚪"
    elif composite >= 0.30:
        signal = "VENTE"
        signal_color = "🟠"
    else:
        signal = "VENTE FORTE"
        signal_color = "🔴"

    return {
        "symbol": symbol,
        "composite_score": composite,
        "ml_score": round(ml_signal, 4),
        "tech_score": round(tech_score, 4),
        "signal": signal,
        "signal_color": signal_color,
        "universe": universe,
        "components": {
            "rsi_score": round(rsi_score, 3),
            "macd_score": round(macd_score, 3),
            "bollinger_score": round(bb_score, 3),
            "volume_score": round(vol_score, 3),
            "sentiment_score": round(sent_score, 3),
        },
    }
