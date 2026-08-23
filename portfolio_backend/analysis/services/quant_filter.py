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
from datetime import datetime, timedelta
from typing import Optional
import logging
import os

import requests
import yfinance as yf
from django.core.cache import cache
from django.utils import timezone

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
                # `if rev and gp` seul ne filtre PAS un NaN -- `bool(float('nan'))`
                # vaut True en Python (seul 0.0 est falsy), donc un Gross Profit
                # manquant (NaN, courant sur les tickers du balayage large
                # S&P500+TSX -- petites capitalisations, dépôts trimestriels
                # incomplets -- confirmé en direct le 2026-08-17 sur OGC.TO,
                # qui a fait planter 2 scans réels le jour même de l'activation
                # de universe_source='combined') passait ce garde-fou tel quel,
                # produisait un NaN dans margin_trend, puis faisait planter
                # ScanResult.objects.create() (Postgres jsonb refuse le token
                # NaN, invalide en JSON strict, contrairement à json.dumps
                # Python qui l'accepte silencieusement). _safe() applique déjà
                # le test d'auto-inégalité NaN utilisé ailleurs dans ce module.
                rev = _safe(revenue.get(col))
                gp = _safe(gross_profit.get(col))
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


def _fetch_insider_buying(ticker: str) -> dict:
    """
    Achats d'initiés récents (90 jours), via Financial Modeling Prep --
    même clé de cache (`insider_summary:{ticker}`) que
    portfolio/views.py::_insider_summary, pour partager les entrées de
    cache entre les deux usages plutôt que doubler les appels API pour le
    même ticker. Volontairement une fonction indépendante ici (pas un
    import cross-app de cette méthode de View) -- même logique répliquée
    en ~25 lignes plutôt que de refactoriser un chemin de code existant
    et déjà en prod, pour ce lot ajouté au screener "Value + Catalyst"
    (2026-08-16).

    Ne fait jamais planter l'appelant : retourne {'insiders_buying': False,
    'insider_buy_total': 0.0} sur toute erreur/absence de clé API, comme
    un simple "pas de signal détecté" plutôt qu'une exception.
    """
    default = {"insiders_buying": False, "insider_buy_total": 0.0}
    cache_key = f"insider_summary:{ticker}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        return default

    url = f"https://financialmodelingprep.com/api/v4/insider-trading?symbol={ticker}&limit=50&apikey={api_key}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return default
        items = resp.json() or []
    except Exception:
        logger.warning("Erreur fetch insider trading pour %s", ticker)
        return default

    if not isinstance(items, list) or not items:
        cache.set(cache_key, default, 60 * 60)
        return default

    cutoff = timezone.now() - timedelta(days=90)
    total_buy = 0.0
    for item in items:
        date_str = item.get("transactionDate") or item.get("date")
        if date_str:
            try:
                if datetime.fromisoformat(date_str).date() < cutoff.date():
                    continue
            except Exception:
                pass
        trans_type = (item.get("transactionType") or "").lower()
        if "purchase" not in trans_type and "buy" not in trans_type:
            continue
        amount = item.get("transactionValue")
        if amount is None:
            price = float(item.get("price") or 0)
            shares = float(item.get("securitiesTransacted") or 0)
            amount = price * shares
        try:
            total_buy += float(amount or 0)
        except Exception:
            continue

    result = {
        "insiders_buying": total_buy >= 50000,
        "insider_buy_total": round(total_buy, 2),
    }
    cache.set(cache_key, result, 60 * 60 * 6)
    return result


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
            # CAN SLIM "C"/"A" (2026-08-23) -- vérifiés en direct sur
            # NVDA (210%/85%) et AAPL (27%/16%), champs yfinance fiables.
            "eps_growth_qoq": _safe(info.get("earningsQuarterlyGrowth")),
            "revenue_growth_qoq": _safe(info.get("revenueGrowth")),
            # Factor Investing (2026-08-23) -- déjà calculés par yfinance,
            # pas besoin de régression maison pour le bêta. ATTENTION :
            # debtToEquity est déjà en POINTS DE POURCENTAGE (vérifié en
            # direct : NVDA=6.56 -> 6,56% de dette, PAS 656% -- comparer à
            # un seuil du même ordre, ex. 50, pas 0.5).
            "debt_to_equity": _safe(info.get("debtToEquity")),
            "beta": _safe(info.get("beta")),
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


def _fetch_earnings_revision(ticker: str) -> dict:
    """
    Signal de révision des bénéfices (méthode Thorp, "Comment profiter des
    erreurs des analystes") -- via yfinance earnings_estimate/eps_trend/
    eps_revisions. Vérifié en direct le 2026-08-23 : disponible et fiable
    pour les titres à bonne couverture (AAPL/MSFT/NVDA : 20-51 analystes)
    mais PAS uniforme même chez les blue chips (JNJ : seulement 4).

    Règle de l'article ADAPTÉE, pas appliquée littéralement : "aucune
    révision à la baisse" est presque impossible à satisfaire dès qu'un
    titre a une bonne couverture -- vérifié en direct : MSFT, clairement
    en tendance haussière sur ses prévisions (BPA année en cours et
    suivante tous deux révisés à la hausse sur 30 jours), a quand même eu
    9 révisions à la baisse sur 30 jours -- normal avec 26-34 analystes,
    il y en a toujours qui vont à contre-courant. On utilise un RATIO
    (hausses / baisses) au lieu d'exiger zéro baisse.
    """
    cache_key = f"earnings_revision:{ticker}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    default = {
        "num_analysts": None, "revised_up_current_year": None,
        "revised_up_next_year": None, "revision_ratio_30d": None,
        "revision_magnitude_pct": None, "thorp_signal": False,
    }
    try:
        tk = yf.Ticker(ticker)
        trend = tk.eps_trend
        est = tk.earnings_estimate
        revisions = tk.eps_revisions
        if trend is None or trend.empty or est is None or est.empty:
            cache.set(cache_key, default, 60 * 60 * 6)
            return default

        num_analysts = None
        if "0y" in est.index:
            raw = _safe(est.loc["0y", "numberOfAnalysts"])
            num_analysts = int(raw) if raw is not None else None

        def _revised_up(period):
            if period not in trend.index:
                return None, None
            current = _safe(trend.loc[period, "current"])
            ago = _safe(trend.loc[period, "30daysAgo"])
            if current is None or ago is None or ago == 0:
                return None, None
            current, ago = float(current), float(ago)
            return current > ago, (current - ago) / abs(ago) * 100

        up_0y, mag_0y = _revised_up("0y")
        up_1y, mag_1y = _revised_up("+1y")
        magnitudes = [m for m in (mag_0y, mag_1y) if m is not None]

        ratio = None
        if revisions is not None and not revisions.empty:
            up_total, down_total = 0, 0
            for period in ("0y", "+1y"):
                if period in revisions.index:
                    up_total += int(_safe(revisions.loc[period, "upLast30days"], 0) or 0)
                    down_total += int(_safe(revisions.loc[period, "downLast30days"], 0) or 0)
            if down_total > 0:
                ratio = round(up_total / down_total, 2)
            elif up_total > 0:
                ratio = 999.0  # pas de baisse du tout -- "Infinity" n'est pas JSON-valide

        thorp_signal = bool(
            num_analysts and num_analysts >= 10
            and up_0y and up_1y
            and ratio is not None and ratio >= 2.0
        )

        result = {
            "num_analysts": num_analysts,
            "revised_up_current_year": up_0y,
            "revised_up_next_year": up_1y,
            "revision_ratio_30d": ratio,
            "revision_magnitude_pct": round(max(magnitudes), 2) if magnitudes else None,
            "thorp_signal": thorp_signal,
        }
        cache.set(cache_key, result, 60 * 60 * 12)  # 12h -- ces données ne bougent pas d'heure en heure
        return result
    except Exception:
        logger.warning("Erreur fetch earnings revision pour %s", ticker)
        return default


def _fetch_market_distribution_days(index_symbol: str = "^GSPC", window_days: int = 25) -> dict:
    """
    "Jours de distribution" (William O'Neil, CAN SLIM) -- signal de SANTÉ
    DU MARCHÉ GLOBAL, pas un critère par titre : compte, sur les ~5
    dernières semaines de séances, les jours où l'indice clôture en
    baisse d'au moins 0,2% sur un volume supérieur à la veille, OU
    stagne (variation quasi nulle) sur un volume nettement supérieur
    ("stalling" -- les institutionnels vendent pendant que les
    particuliers achètent). 4 jours ou plus sur cette fenêtre = signal
    classique d'O'Neil pour arrêter les nouveaux achats.

    Mis en cache par INDICE, pas par ticker -- même résultat pour tous
    les tickers scannés dans le même cycle, pas la peine de le
    recalculer à chaque appel de _evaluate_can_slim.
    """
    cache_key = f"market_distribution_days:{index_symbol}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    default = {"available": False, "distribution_days": 0, "under_distribution": False}
    try:
        hist = yf.Ticker(index_symbol).history(period="3mo")
        if hist is None or hist.empty or len(hist) < window_days + 1:
            cache.set(cache_key, default, 60 * 60)
            return default

        close, volume = hist["Close"], hist["Volume"]
        count = 0
        for pos in range(len(hist) - window_days, len(hist)):
            if pos <= 0:
                continue
            today_close, prev_close = float(close.iloc[pos]), float(close.iloc[pos - 1])
            today_vol, prev_vol = float(volume.iloc[pos]), float(volume.iloc[pos - 1])
            if prev_close == 0 or today_vol <= prev_vol:
                continue
            pct_change = (today_close - prev_close) / prev_close * 100
            is_down = pct_change <= -0.2
            is_stalling = abs(pct_change) < 0.1 and today_vol > prev_vol * 1.1
            if is_down or is_stalling:
                count += 1

        result = {"available": True, "distribution_days": count, "window_days": window_days, "under_distribution": count >= 4}
        cache.set(cache_key, result, 60 * 60)  # 1h -- pertinent au jour le jour, pas la peine de recalculer plus souvent
        return result
    except Exception:
        logger.warning("Erreur fetch distribution days pour %s", index_symbol)
        return default


def _fetch_technical_trend(ticker: str) -> dict:
    """
    Base technique partagée par les presets Minervini (Trend Template) et
    CAN SLIM (partie technique du "L" et du "N") -- moyennes mobiles,
    sommet/creux 52 semaines, force relative vs marché. Une seule requête
    d'historique de prix pour les deux, plutôt que dupliquer l'appel réseau.

    Hors scope volontairement (voir docs/TECH_DEBT_NOTES.md pour le suivi) :
    - Détection de la figure "Tasse avec Anse" -- reconnaissance de forme
      sur série temporelle, pas un simple seuil ; effort séparé, à
      valider contre de vrais exemples avant de la brancher sur du
      capital réel même simulé (voir _fetch_market_distribution_days
      juste au-dessus pour "jours de distribution", qui LUI est implémenté).
    - Règles de sortie (stop-loss 7-8%, prise de profit 20-25%,
      pyramidage) -- ce sont des règles de GESTION DE POSITION après
      achat, pas des critères de screening. Ça se brancherait sur
      monitor_lab_positions, pas ici.
    """
    cache_key = f"technical_trend:{ticker}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    default = {"available": False}
    try:
        hist = yf.Ticker(ticker).history(period="14mo")
        if hist is None or hist.empty or len(hist) < 210:
            cache.set(cache_key, default, 60 * 60 * 6)
            return default

        close = hist["Close"]
        price = float(close.iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        ma150 = float(close.rolling(150).mean().iloc[-1])
        ma200_series = close.rolling(200).mean()
        ma200 = float(ma200_series.iloc[-1])
        ma200_1m_ago = float(ma200_series.iloc[-22]) if len(ma200_series) > 22 else None

        window = close[-252:] if len(close) >= 252 else close
        week52_high, week52_low = float(window.max()), float(window.min())

        avg_volume = float(hist["Volume"][-50:].mean()) if "Volume" in hist else None

        rel_strength_vs_market = None
        try:
            bench = yf.Ticker("^GSPC").history(period="7mo")["Close"]
            if len(bench) > 126 and len(close) > 126:
                ticker_return = price / float(close.iloc[-126]) - 1
                bench_return = float(bench.iloc[-1]) / float(bench.iloc[-126]) - 1
                rel_strength_vs_market = ticker_return > bench_return
        except Exception:
            pass

        # Momentum Factor Investing (2026-08-23) : rendement 12 mois moins
        # le dernier mois -- Ret_12-1 = P[t-1m]/P[t-12m] - 1. Exclut le
        # dernier mois volontairement (effet de retournement à court terme
        # bien documenté -- la formule standard de la littérature, pas une
        # approximation).
        momentum_12_1 = None
        if len(close) > 252:
            price_12m_ago = float(close.iloc[-252])
            price_1m_ago = float(close.iloc[-21])
            if price_12m_ago > 0:
                momentum_12_1 = round((price_1m_ago / price_12m_ago - 1) * 100, 2)

        result = {
            "available": True,
            "price": round(price, 2), "ma50": round(ma50, 2), "ma150": round(ma150, 2),
            "ma200": round(ma200, 2), "ma200_trending_up_1m": ma200_1m_ago is not None and ma200 > ma200_1m_ago,
            "week_52_high": round(week52_high, 2), "week_52_low": round(week52_low, 2),
            "avg_volume_50d": round(avg_volume) if avg_volume is not None else None,
            "outperforms_market_6m": rel_strength_vs_market,
            "momentum_12_1_pct": momentum_12_1,
        }
        cache.set(cache_key, result, 60 * 60 * 12)
        return result
    except Exception:
        logger.warning("Erreur fetch technical trend pour %s", ticker)
        return default


def _evaluate_minervini_template(trend: dict) -> dict:
    """8 critères du "Trend Template" de Mark Minervini (SEPA) -- un
    filtre binaire chez lui : les 8 doivent être respectés, pas une
    majorité."""
    if not trend.get("available"):
        return {"criteria_met": 0, "criteria_total": 8, "passes_template": False}

    price, ma50, ma150, ma200 = trend["price"], trend["ma50"], trend["ma150"], trend["ma200"]
    criteria = {
        "price_above_150_200": price > ma150 and price > ma200,
        "ma150_above_ma200": ma150 > ma200,
        "ma200_trending_up_1m": trend["ma200_trending_up_1m"],
        "ma50_above_150_200": ma50 > ma150 and ma50 > ma200,
        "price_above_ma50": price > ma50,
        "price_25pct_above_52w_low": price >= trend["week_52_low"] * 1.25,
        "price_within_25pct_of_52w_high": price >= trend["week_52_high"] * 0.75,
        "outperforms_market_6m": bool(trend.get("outperforms_market_6m")),
    }
    criteria_met = sum(1 for v in criteria.values() if v)
    return {
        **criteria, "criteria_met": criteria_met, "criteria_total": len(criteria),
        "passes_template": criteria_met == len(criteria),
    }


def _evaluate_factor_investing(data: dict, trend: dict) -> dict:
    """
    Factor Investing multi-facteurs -- Momentum, Qualité, Faible
    volatilité. Version SIMPLIFIÉE d'un vrai système de classement (le
    texte original veut un Z-score composite sur tout un univers,
    rebalancé mensuellement, achat du top 10%) : ici, un filtre binaire
    PAR TICKER (3 des 4 critères requis, pas un classement relatif) --
    plus simple à intégrer dans le pipeline existant (par ticker, pas
    par univers classé ensemble). À revisiter si un vrai classement
    relatif est voulu plus tard.

    Qualité : ROE > 15% ET dette/capitaux propres < 50 (déjà en points
    de pourcentage chez yfinance -- vérifié en direct : NVDA=6,56 =
    6,56% de dette, PAS 656%).
    Faible volatilité : bêta < 0,8 (déjà calculé par yfinance, pas de
    régression maison).
    Momentum : rendement 12 mois moins le dernier mois positif (voir
    _fetch_technical_trend).

    Note : low-vol et momentum sont en tension naturelle (un titre en
    fort momentum a souvent un bêta plus élevé) -- exiger les 4 critères
    serait presque toujours impossible. 3/4 requis.
    """
    if not trend.get("available"):
        return {"available": False, "criteria_met": 0, "criteria_total": 4, "passes_screen": False}

    roe, debt_to_equity, beta = data.get("roe"), data.get("debt_to_equity"), data.get("beta")
    momentum = trend.get("momentum_12_1_pct")

    criteria = {
        "quality_roe_over_15pct": roe is not None and roe > 15,
        "quality_debt_to_equity_under_50": debt_to_equity is not None and debt_to_equity < 50,
        "low_vol_beta_under_0_8": beta is not None and beta < 0.8,
        "momentum_12_1_positive": momentum is not None and momentum > 0,
    }
    criteria_met = sum(1 for v in criteria.values() if v)
    return {
        "available": True, **criteria,
        "roe_pct": roe, "debt_to_equity_pct": debt_to_equity, "beta": beta, "momentum_12_1_pct": momentum,
        "criteria_met": criteria_met, "criteria_total": len(criteria),
        "passes_screen": criteria_met >= 3,
    }


def _evaluate_can_slim(data: dict, trend: dict) -> dict:
    """
    CAN SLIM (William O'Neil) -- partie QUANTIFIABLE seulement (Tasse
    avec Anse toujours hors scope, voir docstring de
    _fetch_technical_trend -- reconnaissance de forme, effort séparé,
    à valider contre de vrais exemples avant de la brancher sur du
    capital réel même simulé).

    C/A (bénéfices/ventes en forte accélération) : earningsQuarterlyGrowth
    et revenueGrowth de yfinance -- vérifiés en direct (NVDA 210%/85%,
    AAPL 27%/16%). La croissance BPA sur 5 ans (aussi dans la méthode
    originale) est omise : demande un historique annuel fiable sur 5 ans,
    peu robuste sur un balayage large (dépôts trimestriels incomplets,
    même mise en garde que _fetch_margin_trend plus haut dans ce module).
    S (volume) : moyenne 50 jours > 200k actions.
    L/N (leader technique, nouveaux sommets) : réutilise _fetch_technical_trend.
    M (direction du marché, "jours de distribution") : voir
    _fetch_market_distribution_days -- signal de marché global, pas
    propre au ticker, mais évalué ici parce que c'est une règle CAN SLIM
    explicite ("ne jamais acheter si le marché est sous distribution").
    """
    if not trend.get("available"):
        return {"available": False, "criteria_met": 0, "criteria_total": 7, "passes_screen": False}

    eps_growth_qoq = data.get("eps_growth_qoq")
    revenue_growth_qoq = data.get("revenue_growth_qoq")
    avg_volume = trend.get("avg_volume_50d")
    market_health = _fetch_market_distribution_days()

    criteria = {
        "eps_growth_over_20pct": eps_growth_qoq is not None and eps_growth_qoq > 0.20,
        "revenue_growth_over_20pct": revenue_growth_qoq is not None and revenue_growth_qoq > 0.20,
        "volume_over_200k": avg_volume is not None and avg_volume > 200_000,
        "price_above_ma50": trend["price"] > trend["ma50"],
        "ma50_above_ma200": trend["ma50"] > trend["ma200"],
        "within_25pct_of_52w_high": trend["price"] >= trend["week_52_high"] * 0.75,
        "market_not_under_distribution": not market_health.get("under_distribution", False),
    }
    criteria_met = sum(1 for v in criteria.values() if v)
    return {
        "available": True, **criteria,
        "eps_growth_qoq_pct": round(eps_growth_qoq * 100, 1) if eps_growth_qoq is not None else None,
        "revenue_growth_qoq_pct": round(revenue_growth_qoq * 100, 1) if revenue_growth_qoq is not None else None,
        "market_distribution_days": market_health.get("distribution_days"),
        "criteria_met": criteria_met, "criteria_total": len(criteria),
        # Filtre "screen", pas un template binaire comme Minervini -- au moins
        # 6/7 (toutes sauf une, ex. volume tout juste sous 200k reste acceptable).
        "passes_screen": criteria_met >= 6,
    }


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

    # Preset "Value + Catalyst" (2026-08-16) : contrairement à "Undervalued
    # Without Reason" (qui ne demande QUE l'absence de mauvaise raison),
    # ce preset exige positivement un catalyseur visible -- inflexion de
    # marge (déjà calculée ci-dessus) OU achat d'initiés significatif
    # (>= 50k$ sur 90 jours). Contrôlé par thresholds['require_catalyst']
    # sur le ScreenerPreset -- absent/False pour "Undervalued Without
    # Reason" (thresholds={}), donc aucun changement de comportement pour
    # ce preset existant. L'appel réseau (_fetch_insider_buying) n'est fait
    # que si ce flag est actif, pour ne pas ajouter un appel API inutile
    # aux presets qui n'en ont pas besoin.
    insiders_buying = False
    if thresholds.get("require_catalyst"):
        insider_data = _fetch_insider_buying(ticker)
        data["insiders_buying"] = insider_data["insiders_buying"]
        data["insider_buy_total"] = insider_data["insider_buy_total"]
        insiders_buying = insider_data["insiders_buying"]
        if insiders_buying:
            score += 1.5  # même poids que l'inflexion de marge -- deux voies équivalentes vers "catalyseur"
        has_catalyst = margin_inflection or insiders_buying
        if not has_catalyst:
            reasons_failed.append(
                "Aucun catalyseur détecté (ni inflexion de marge, ni achat d'initiés >= 50k$ sur 90 jours)"
            )

    # Preset "Insider Trading" (2026-08-23) -- contrairement à
    # require_catalyst ci-dessus (OU avec l'inflexion de marge), ce preset
    # exige l'achat d'initiés SEUL, sans porte de sortie. Appel réseau
    # séparé de celui ci-dessus : si un preset active les deux flags un
    # jour, on évite un double appel en réutilisant insider_data s'il a
    # déjà été fetché.
    if thresholds.get("require_insider_buying"):
        if not thresholds.get("require_catalyst"):
            insider_data = _fetch_insider_buying(ticker)
            data["insiders_buying"] = insider_data["insiders_buying"]
            data["insider_buy_total"] = insider_data["insider_buy_total"]
            insiders_buying = insider_data["insiders_buying"]
        if insiders_buying:
            score += 2.0  # seul signal du preset -- poids élevé, comme Minervini/CAN SLIM
        else:
            reasons_failed.append(
                "Aucun achat d'initiés significatif détecté (>= 50k$ sur 90 jours)"
            )

    # Méthode Thorp (2026-08-23, "Comment profiter des erreurs des
    # analystes") -- révisions de bénéfices, voir docstring de
    # _fetch_earnings_revision pour la règle exacte (ratio, pas "zéro
    # baisse" littéral). Même garde-fou que require_catalyst : appel réseau
    # seulement si le preset le demande.
    earnings_revision = {}
    if thresholds.get("require_earnings_revision"):
        earnings_revision = _fetch_earnings_revision(ticker)
        data["earnings_revision"] = earnings_revision
        if earnings_revision.get("thorp_signal"):
            score += 1.5
        else:
            reasons_failed.append(
                "Pas de signal de révision de bénéfices (Thorp) : "
                f"analystes={earnings_revision.get('num_analysts')}, "
                f"ratio hausse/baisse={earnings_revision.get('revision_ratio_30d')}"
            )

    # Modèles "style d'investisseur" (2026-08-23) -- Minervini (Trend
    # Template, filtre binaire 8/8) et CAN SLIM (William O'Neil, partie
    # quantifiable seulement -- voir _fetch_technical_trend pour ce qui
    # est hors scope). Une seule requête d'historique de prix partagée
    # entre les deux si le preset active l'un OU l'autre.
    technical_trend = {}
    if (
        thresholds.get("require_minervini_trend")
        or thresholds.get("require_can_slim")
        or thresholds.get("require_factor_investing")
    ):
        technical_trend = _fetch_technical_trend(ticker)
        data["technical_trend"] = technical_trend

    if thresholds.get("require_minervini_trend"):
        minervini = _evaluate_minervini_template(technical_trend)
        data["minervini"] = minervini
        if minervini["passes_template"]:
            score += 2.0  # filtre binaire chez Minervini -- poids élevé, tout ou rien
        else:
            reasons_failed.append(
                f"Trend Template Minervini non respecté ({minervini['criteria_met']}/{minervini['criteria_total']} critères)"
            )

    if thresholds.get("require_can_slim"):
        can_slim = _evaluate_can_slim(data, technical_trend)
        data["can_slim"] = can_slim
        if can_slim["passes_screen"]:
            score += 2.0
        else:
            reasons_failed.append(
                f"Filtre CAN SLIM non respecté ({can_slim['criteria_met']}/{can_slim['criteria_total']} critères)"
            )

    # Factor Investing (2026-08-23) -- version simplifiée par titre (voir
    # docstring de _evaluate_factor_investing) : Qualité (ROE + D/E) + Faible
    # volatilité (bêta) + Momentum 12-1. Exige 3/4, pas 4/4 (faible-vol et
    # momentum sont naturellement en tension).
    if thresholds.get("require_factor_investing"):
        factor = _evaluate_factor_investing(data, technical_trend)
        data["factor_investing"] = factor
        if factor["passes_screen"]:
            score += 2.0
        else:
            reasons_failed.append(
                f"Filtre Factor Investing non respecté ({factor['criteria_met']}/{factor['criteria_total']} critères)"
            )

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
