"""
Étape 1 du screener : filtre quantitatif pur (aucun appel LLM).
Calcule les ratios du guide (P/E, PEG, EV/EBITDA, ROE, dette/EBITDA, FCF Yield),
les facteurs additionnels (inflexion de marge, insider cluster, couverture
analyste) et produit un score composite par ticker.

Intégration : `.info` passe par portfolio.market_data.Ticker plutôt que
yfinance brut — c'est le wrapper déjà utilisé par le pipeline ML (timeout,
prix Alpaca en priorité avec fallback yfinance). `.quarterly_financials`
n'y est pas exposé (le wrapper ne couvre que .info/.history/.financials/
.balance_sheet), donc on retombe sur yfinance direct pour cette seule
lecture — mêmes garde-fous (try/except, jamais de crash amont).
"""

from dataclasses import dataclass, field
from typing import Optional
import logging

import yfinance as yf

from portfolio import market_data

logger = logging.getLogger(__name__)


# Seuils par défaut par secteur — reprend le tableau de l'étape 3 du guide.
# Ces valeurs sont des DÉFAUTS ; le ScreenerPreset.thresholds en DB a priorité
# s'il définit une clé équivalente (voir merge_thresholds).
SECTOR_THRESHOLDS = {
    "Technology": {"debt_ebitda_max": 1.5, "peg_max": 1.5, "roe_min": 15},
    "Financial Services": {"pb_max": 1.5, "roe_min": 12},
    "Utilities": {"debt_ebitda_max": 6, "roe_min": 8},
    "Communication Services": {"debt_ebitda_max": 5, "roe_min": 10},
    "Industrials": {"debt_ebitda_max": 3.5, "roe_min": 12},
    "Consumer Defensive": {"peg_max": 2, "fcf_yield_min": 4},
    "Consumer Cyclical": {"debt_ebitda_max": 3, "roe_min": 15},
    "Healthcare": {"debt_ebitda_max": 3, "roe_min": 15},
    "Energy": {"debt_ebitda_max": 3, "roe_min": 10},
    "Basic Materials": {"debt_ebitda_max": 3, "roe_min": 10},
    "Real Estate": {"debt_ebitda_max": 6, "roe_min": 8},
    "DEFAULT": {"debt_ebitda_max": 3, "roe_min": 15, "peg_max": 1.5, "pb_max": 3, "fcf_yield_min": 5},
}


@dataclass
class QuantResult:
    ticker: str
    sector: str
    passed: bool
    score: float
    price: float
    details: dict = field(default_factory=dict)
    reasons_failed: list = field(default_factory=list)


def _safe(value, default=None):
    """yfinance retourne souvent None ou NaN — normalise ça proprement."""
    if value is None:
        return default
    try:
        if value != value:  # NaN check sans dépendre de math/numpy ici
            return default
    except TypeError:
        pass
    return value


def _fetch_margin_trend(ticker: str) -> list:
    """
    Historique trimestriel pour l'inflexion de marge (4 derniers trimestres).
    portfolio.market_data.Ticker n'expose pas quarterly_financials, donc
    yfinance direct ici — best-effort, ne doit jamais faire planter l'appelant.
    """
    try:
        financials_q = yf.Ticker(ticker).quarterly_financials
    except Exception:
        logger.warning("Erreur fetch quarterly_financials pour %s", ticker)
        return []

    margin_trend = []
    if financials_q is not None and not financials_q.empty:
        try:
            revenue = financials_q.loc["Total Revenue"]
            gross_profit = financials_q.loc["Gross Profit"]
            for col in financials_q.columns[:4]:
                rev = revenue.get(col)
                gp = gross_profit.get(col)
                if rev and gp and rev != 0:
                    # pandas .loc[...] renvoie des scalaires numpy (float64) —
                    # cast en float natif tout de suite, sinon les comparaisons
                    # dans _detect_margin_inflection produisent un numpy.bool_
                    # (pas un bool Python), que JSONField/json.dumps refuse de
                    # sérialiser au moment de sauvegarder ScanResult.quant_details.
                    margin_trend.append(round(float(gp) / float(rev) * 100, 2))
        except KeyError:
            pass  # certains tickers n'ont pas ces lignes exactement nommées
    return list(reversed(margin_trend))  # ordre chronologique


def fetch_ticker_data(ticker: str) -> Optional[dict]:
    """
    Récupère les données brutes nécessaires via portfolio.market_data.Ticker
    (wrapper Alpaca-first + fallback yfinance déjà utilisé par le pipeline ML).
    Retourne None si le ticker est introuvable ou si des données essentielles
    manquent (pas de crash, juste un skip côté appelant).
    """
    try:
        info = market_data.Ticker(ticker).info
        if not info or info.get("regularMarketPrice") is None:
            logger.warning("Pas de données pour %s, skip.", ticker)
            return None

        margin_trend = _fetch_margin_trend(ticker)
        price = _safe(info.get("regularMarketPrice") or info.get("currentPrice"))

        return {
            "ticker": ticker,
            "price": price,
            "sector": _safe(info.get("sector"), "DEFAULT"),
            "pe_ratio": _safe(info.get("trailingPE")),
            "peg_ratio": _safe(info.get("pegRatio")),
            "pb_ratio": _safe(info.get("priceToBook")),
            "ev_ebitda": _safe(info.get("enterpriseToEbitda")),
            "roe": _safe(info.get("returnOnEquity"), 0) * 100 if info.get("returnOnEquity") else None,
            "debt_to_ebitda": _compute_debt_to_ebitda(info),
            "fcf_yield": _compute_fcf_yield(info, price),
            "num_analysts": _safe(info.get("numberOfAnalystOpinions"), 0),
            "gross_margin_trend": margin_trend,
            "market_cap": _safe(info.get("marketCap")),
        }
    except Exception:
        logger.exception("Erreur fetch données pour %s", ticker)
        return None


def _compute_debt_to_ebitda(info: dict) -> Optional[float]:
    total_debt = _safe(info.get("totalDebt"))
    ebitda = _safe(info.get("ebitda"))
    if total_debt and ebitda and ebitda != 0:
        return round(total_debt / ebitda, 2)
    return None


def _compute_fcf_yield(info: dict, price: Optional[float]) -> Optional[float]:
    fcf = _safe(info.get("freeCashflow"))
    shares = _safe(info.get("sharesOutstanding"))
    if fcf and shares and price and price != 0:
        fcf_per_share = fcf / shares
        return round(fcf_per_share / price * 100, 2)
    return None


def _detect_margin_inflection(margin_trend: list) -> bool:
    """
    True si les 2 derniers trimestres montrent une amélioration après
    un creux — cf. preset "Margin Inflection" discuté.
    """
    if len(margin_trend) < 4:
        return False
    # creux dans les 2 premiers trimestres de la fenêtre, amélioration ensuite
    trough_idx = margin_trend.index(min(margin_trend[:3]))
    recent = margin_trend[trough_idx:]
    return len(recent) >= 3 and recent[-1] > recent[-2] > recent[0]


def merge_thresholds(sector: str, preset_thresholds: dict) -> dict:
    base = dict(SECTOR_THRESHOLDS.get(sector, SECTOR_THRESHOLDS["DEFAULT"]))
    base.update({k: v for k, v in (preset_thresholds or {}).items() if v is not None})
    return base


def evaluate_ticker(ticker: str, preset_thresholds: dict) -> Optional[QuantResult]:
    """
    Point d'entrée principal de l'étape 1. Retourne un QuantResult
    (passed=True/False) ou None si les données étaient insuffisantes.
    """
    data = fetch_ticker_data(ticker)
    if data is None:
        return None

    sector = data["sector"]
    thresholds = merge_thresholds(sector, preset_thresholds)
    reasons_failed = []
    score = 0.0

    # --- Filtres obligatoires (garde-fous qualité, étape 1 du guide) ---
    roe_min = thresholds.get("roe_min")
    if roe_min and data["roe"] is not None:
        if data["roe"] < roe_min:
            reasons_failed.append(f"ROE {data['roe']:.1f}% < seuil {roe_min}%")
        else:
            score += 1

    debt_max = thresholds.get("debt_ebitda_max")
    if debt_max and data["debt_to_ebitda"] is not None:
        if data["debt_to_ebitda"] > debt_max:
            reasons_failed.append(f"Dette/EBITDA {data['debt_to_ebitda']}x > seuil {debt_max}x")
        else:
            score += 1

    # --- Facteurs de scoring additionnels (pondération, pas obligatoires) ---
    peg_max = thresholds.get("peg_max")
    if peg_max and data["peg_ratio"] and data["peg_ratio"] > 0:
        if data["peg_ratio"] < peg_max:
            score += 1

    fcf_min = thresholds.get("fcf_yield_min")
    if fcf_min and data["fcf_yield"] is not None and data["fcf_yield"] >= fcf_min:
        score += 1

    margin_inflection = _detect_margin_inflection(data["gross_margin_trend"])
    if margin_inflection:
        score += 1.5  # facteur "coup de circuit", poids plus élevé

    if data["num_analysts"] is not None and 0 < data["num_analysts"] < 5:
        score += 1  # faible couverture analyste

    passed = len(reasons_failed) == 0

    return QuantResult(
        ticker=ticker,
        sector=sector,
        passed=passed,
        score=round(score, 2),
        price=data["price"],
        details={
            **data,
            "margin_inflection_detected": margin_inflection,
            "thresholds_used": thresholds,
        },
        reasons_failed=reasons_failed,
    )


def run_quant_filter(tickers: list, preset_thresholds: dict) -> list:
    """Lance l'étape 1 sur une liste de tickers, retourne uniquement ceux qui passent."""
    results = []
    for ticker in tickers:
        result = evaluate_ticker(ticker, preset_thresholds)
        if result and result.passed:
            results.append(result)
        elif result:
            logger.info("%s rejeté à l'étape 1: %s", ticker, result.reasons_failed)
    return results
