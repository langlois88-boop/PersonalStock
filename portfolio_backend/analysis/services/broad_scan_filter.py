"""
Filtre quantitatif STRICT pour le balayage large hebdomadaire (S&P 500 +
TSX Composite, voir analysis/tasks.py::run_broad_index_scan). Fonction
séparée de quant_filter.evaluate_ticker() -- celle-ci reste inchangée pour
le pipeline quotidien sur univers restreint (garde-fous plus permissifs,
tourne tous les jours). Ici : garde-fous BEAUCOUP plus stricts (tous
obligatoires, ET pas OU), tourne une fois par semaine seulement sur un
univers large -- le but est de trouver une poignée de titres de très haute
qualité, pas un tri large.

Formules Piotroski F-Score et Altman Z-Score standard (pas improvisées) :
- Piotroski (2000), "Value Investing: The Use of Historical Financial
  Statement Information to Separate Winners from Losers" -- 9 tests
  binaires (profitabilité, effet de levier/liquidité, efficacité
  opérationnelle).
- Altman (1968) Z-Score original (entreprises publiques, hors services
  financiers) : Z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E.

Toutes les données via yfinance direct (portfolio.market_data.Ticker
n'expose pas balance_sheet/financials/cashflow annuels détaillés -- même
limitation déjà documentée dans quant_filter.py::_fetch_margin_trend).
"""

from dataclasses import dataclass, field
from typing import Optional
import logging

import yfinance as yf

from .quant_filter import SECTOR_THRESHOLDS, merge_thresholds, _safe

logger = logging.getLogger(__name__)

# Seuils stricts fixes (pas de variation par preset ici, contrairement au
# pipeline quotidien -- ce sont les garde-fous du balayage large lui-même).
STRICT_ROE_MIN = 15.0
STRICT_ROIC_MIN = 12.0
STRICT_PEG_MAX = 1.2
STRICT_FCF_YIELD_MIN = 6.0
STRICT_PIOTROSKI_MIN = 7
STRICT_ALTMAN_Z_MIN = 3.0
DEFAULT_TAX_RATE = 0.25  # utilisé seulement si 'Tax Rate For Calcs' absent


@dataclass
class StrictResult:
    ticker: str
    passed: bool
    reasons_failed: list = field(default_factory=list)
    piotroski_score: Optional[int] = None
    altman_z_score: Optional[float] = None
    roic: Optional[float] = None
    pe_ratio: Optional[float] = None
    sector: str = "DEFAULT"
    details: dict = field(default_factory=dict)


def _row(df, label):
    """Renvoie la ligne `label` d'un DataFrame yfinance (balance_sheet/
    financials/cashflow), ou None si absente -- ces DataFrames n'ont pas
    toujours toutes les lignes selon le secteur/l'émetteur (ex. les
    financières n'ont pas 'Gross Profit')."""
    if df is None or df.empty or label not in df.index:
        return None
    return df.loc[label]


def _val(row, col):
    """Valeur d'une ligne à une colonne (année) donnée, normalisée."""
    if row is None or col not in row.index:
        return None
    return _safe(row.get(col))


def fetch_annual_statements(ticker: str) -> Optional[dict]:
    """
    Fetch balance_sheet/financials(income_stmt)/cashflow/info en un bloc.
    Renvoie None si le ticker est introuvable ou si les 3 états financiers
    sont tous vides -- jamais de crash, l'appelant traite ça comme un échec
    du filtre strict (donnée insuffisante = ne passe pas, cohérent avec
    "obligatoires").
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        bs = t.balance_sheet
        inc = t.financials
        cf = t.cashflow
        if (bs is None or bs.empty) and (inc is None or inc.empty):
            return None
        years = list(bs.columns) if bs is not None and not bs.empty else list(inc.columns)
        if len(years) < 2:
            return None  # Piotroski a besoin d'au moins 2 ans pour les comparaisons YoY
        return {"info": info, "bs": bs, "inc": inc, "cf": cf, "years": years}
    except Exception:
        logger.warning("Échec fetch_annual_statements pour %s", ticker, exc_info=True)
        return None


def compute_piotroski_score(stmts: dict) -> Optional[int]:
    """9 tests binaires, Piotroski (2000). None si données insuffisantes
    pour au moins 2 années consécutives (impossible de faire les
    comparaisons YoY qui composent 5 des 9 tests)."""
    bs, inc, cf, years = stmts["bs"], stmts["inc"], stmts["cf"], stmts["years"]
    if len(years) < 2:
        return None
    y0, y1 = years[0], years[1]  # année la plus récente, puis précédente

    net_income = _row(inc, "Net Income")
    op_cf = _row(cf, "Operating Cash Flow")
    total_assets = _row(bs, "Total Assets")
    long_term_debt = _row(bs, "Long Term Debt")
    current_assets = _row(bs, "Current Assets")
    current_liabilities = _row(bs, "Current Liabilities")
    shares = _row(bs, "Ordinary Shares Number")
    gross_profit = _row(inc, "Gross Profit")
    revenue = _row(inc, "Total Revenue")

    ni0, ni1 = _val(net_income, y0), _val(net_income, y1)
    ocf0 = _val(op_cf, y0)
    ta0, ta1 = _val(total_assets, y0), _val(total_assets, y1)
    ltd0, ltd1 = _val(long_term_debt, y0), _val(long_term_debt, y1)
    ca0, ca1 = _val(current_assets, y0), _val(current_assets, y1)
    cl0, cl1 = _val(current_liabilities, y0), _val(current_liabilities, y1)
    sh0, sh1 = _val(shares, y0), _val(shares, y1)
    gp0, gp1 = _val(gross_profit, y0), _val(gross_profit, y1)
    rev0, rev1 = _val(revenue, y0), _val(revenue, y1)

    # Il faut au minimum le résultat net et le total des actifs sur les 2
    # ans pour que le score ait un sens -- sinon on ne peut affirmer aucun
    # des 9 tests avec confiance.
    if None in (ni0, ni1, ta0, ta1) or ta0 == 0 or ta1 == 0:
        return None

    score = 0
    roa0 = ni0 / ta0
    roa1 = ni1 / ta1

    # --- Profitabilité (4 points) ---
    if ni0 > 0:
        score += 1
    if ocf0 is not None and ocf0 > 0:
        score += 1
    if roa0 > roa1:
        score += 1
    if ocf0 is not None and ocf0 > ni0:
        score += 1

    # --- Effet de levier / liquidité (3 points) ---
    if ltd0 is not None and ltd1 is not None:
        if (ltd0 / ta0) < (ltd1 / ta1):
            score += 1
    if ca0 is not None and ca1 is not None and cl0 and cl1:
        if (ca0 / cl0) > (ca1 / cl1):
            score += 1
    if sh0 is not None and sh1 is not None and sh0 <= sh1:
        score += 1

    # --- Efficacité opérationnelle (2 points) ---
    if gp0 is not None and gp1 is not None and rev0 and rev1:
        if (gp0 / rev0) > (gp1 / rev1):
            score += 1
    if rev0 is not None and rev1 is not None:
        if (rev0 / ta0) > (rev1 / ta1):
            score += 1

    return score


def compute_altman_z_score(stmts: dict, market_cap: Optional[float]) -> Optional[float]:
    """Z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E, Altman (1968) original.
    None si le total des actifs ou le total des passifs manque (dénominateurs
    de 4 des 5 composantes)."""
    bs, inc, years = stmts["bs"], stmts["inc"], stmts["years"]
    y0 = years[0]

    total_assets = _val(_row(bs, "Total Assets"), y0)
    if not total_assets:
        return None
    total_liabilities = _val(_row(bs, "Total Liabilities Net Minority Interest"), y0)
    working_capital = _val(_row(bs, "Working Capital"), y0)
    retained_earnings = _val(_row(bs, "Retained Earnings"), y0)
    ebit = _val(_row(inc, "EBIT"), y0)
    revenue = _val(_row(inc, "Total Revenue"), y0)

    if working_capital is None:
        # Pas toujours une ligne directe -- calcul manuel de secours.
        ca = _val(_row(bs, "Current Assets"), y0)
        cl = _val(_row(bs, "Current Liabilities"), y0)
        working_capital = (ca - cl) if (ca is not None and cl is not None) else None

    a = (working_capital / total_assets) if working_capital is not None else 0.0
    b = (retained_earnings / total_assets) if retained_earnings is not None else 0.0
    c = (ebit / total_assets) if ebit is not None else 0.0
    d = (market_cap / total_liabilities) if (market_cap and total_liabilities) else 0.0
    e = (revenue / total_assets) if revenue is not None else 0.0

    if retained_earnings is None and ebit is None and revenue is None:
        return None  # aucune des composantes principales n'est disponible

    return round(1.2 * a + 1.4 * b + 3.3 * c + 0.6 * d + 1.0 * e, 2)


def compute_roic(stmts: dict) -> Optional[float]:
    """ROIC = NOPAT / Invested Capital, NOPAT = EBIT * (1 - taux d'impôt).
    'Invested Capital' est une ligne directe yfinance (pas besoin de la
    recalculer depuis dette+capitaux propres-cash). Taux d'impôt réel
    ('Tax Rate For Calcs') si disponible, sinon défaut 25%."""
    bs, inc, years = stmts["bs"], stmts["inc"], stmts["years"]
    y0 = years[0]
    ebit = _val(_row(inc, "EBIT"), y0)
    invested_capital = _val(_row(bs, "Invested Capital"), y0)
    if ebit is None or not invested_capital:
        return None
    tax_rate = _val(_row(inc, "Tax Rate For Calcs"), y0)
    if tax_rate is None or not (0 <= tax_rate <= 1):
        tax_rate = DEFAULT_TAX_RATE
    nopat = ebit * (1 - tax_rate)
    return round(nopat / invested_capital * 100, 2)


def compute_margin_trend_3y(stmts: dict) -> Optional[list]:
    """Marge brute (%) sur jusqu'à 3 années annuelles disponibles, la plus
    ancienne en premier -- pour vérifier "stable ou en croissance"."""
    inc, years = stmts["inc"], stmts["years"]
    gross_profit = _row(inc, "Gross Profit")
    revenue = _row(inc, "Total Revenue")
    if gross_profit is None or revenue is None:
        return None
    trend = []
    for y in years[:3]:
        gp, rev = _val(gross_profit, y), _val(revenue, y)
        if gp is None or not rev:
            continue
        trend.append(round(gp / rev * 100, 2))
    return list(reversed(trend)) if len(trend) >= 2 else None


def _margin_stable_or_growing(trend: Optional[list]) -> bool:
    if not trend or len(trend) < 2:
        return False
    # Tolère une baisse marginale (bruit de mesure) mais pas une vraie
    # dégradation -- chaque pas ne doit pas reculer de plus de 0.5pt.
    return all(trend[i] >= trend[i - 1] - 0.5 for i in range(1, len(trend)))


def evaluate_ticker_strict(ticker: str, sector_avg_pe: Optional[float] = None, data: Optional[dict] = None) -> StrictResult:
    """
    Applique TOUS les garde-fous stricts (ET, pas OU) -- voir le docstring
    du module. `sector_avg_pe` est calculé par l'appelant (run_broad_index_
    scan) en agrégeant sur le lot en cours de scan, pas recalculé par
    ticker (éviterait un balayage en O(n²) sur les appels API).

    `data` : résultat déjà obtenu de quant_filter.fetch_ticker_data(ticker),
    pour éviter un second appel réseau identique si l'appelant l'a déjà
    récupéré pour sa première passe (calcul de sector_avg_pe notamment).
    Fetché ici seulement si non fourni (garde la fonction utilisable seule,
    ex. pour la validation manuelle faite avant l'écriture de la tâche).
    """
    from .quant_filter import fetch_ticker_data  # import tardif, évite un cycle avec quant_filter

    if data is None:
        data = fetch_ticker_data(ticker)
    if data is None:
        return StrictResult(ticker=ticker, passed=False, reasons_failed=["Données de base indisponibles."])

    stmts = fetch_annual_statements(ticker)
    if stmts is None:
        return StrictResult(
            ticker=ticker, passed=False, sector=data.get("sector", "DEFAULT"),
            reasons_failed=["États financiers annuels indisponibles (< 2 ans de données)."],
        )

    sector = data.get("sector") or "DEFAULT"
    thresholds = merge_thresholds(sector, {})
    reasons: list[str] = []

    roe = data.get("roe")
    if roe is None or roe <= STRICT_ROE_MIN:
        reasons.append(f"ROE {roe if roe is not None else 'n/a'} <= {STRICT_ROE_MIN}%")

    roic = compute_roic(stmts)
    if roic is None or roic <= STRICT_ROIC_MIN:
        reasons.append(f"ROIC {roic if roic is not None else 'n/a'} <= {STRICT_ROIC_MIN}%")

    debt_max = thresholds.get("debt_ebitda_max", SECTOR_THRESHOLDS["DEFAULT"]["debt_ebitda_max"])
    debt_ebitda = data.get("debt_to_ebitda")
    if debt_ebitda is None or debt_ebitda >= debt_max:
        reasons.append(f"Dette/EBITDA {debt_ebitda if debt_ebitda is not None else 'n/a'} >= seuil secteur {debt_max}x")

    margin_trend = compute_margin_trend_3y(stmts)
    if not _margin_stable_or_growing(margin_trend):
        reasons.append(f"Marge brute pas stable/croissante sur 3 ans (trend={margin_trend}).")

    pe = data.get("pe_ratio")
    if pe is None or pe <= 0:
        reasons.append("P/E indisponible ou négatif.")
    else:
        if sector_avg_pe is not None and pe >= sector_avg_pe:
            reasons.append(f"P/E {pe} >= moyenne secteur {sector_avg_pe}")
        elif sector_avg_pe is None:
            reasons.append("Moyenne P/E du secteur indisponible pour cette comparaison.")
        # Moyenne historique 5 ans du titre lui-même : non calculée ici
        # (nécessiterait un historique de prix + EPS par ticker, coût API
        # trop élevé sur un lot de 700+ titres hebdomadaire) -- voir
        # docs/TECH_DEBT_NOTES.md pour la limitation documentée. Le critère
        # P/E < moyenne secteur seul reste appliqué en attendant.

    peg = data.get("peg_ratio")
    if peg is None or peg <= 0 or peg >= STRICT_PEG_MAX:
        reasons.append(f"PEG {peg if peg is not None else 'n/a'} >= {STRICT_PEG_MAX}")

    fcf_yield = data.get("fcf_yield")
    if fcf_yield is None or fcf_yield <= STRICT_FCF_YIELD_MIN:
        reasons.append(f"FCF Yield {fcf_yield if fcf_yield is not None else 'n/a'} <= {STRICT_FCF_YIELD_MIN}%")

    piotroski = compute_piotroski_score(stmts)
    if piotroski is None or piotroski < STRICT_PIOTROSKI_MIN:
        reasons.append(f"Piotroski F-Score {piotroski if piotroski is not None else 'n/a'} < {STRICT_PIOTROSKI_MIN}")

    altman_z = compute_altman_z_score(stmts, data.get("market_cap"))
    if altman_z is None or altman_z <= STRICT_ALTMAN_Z_MIN:
        reasons.append(f"Altman Z-Score {altman_z if altman_z is not None else 'n/a'} <= {STRICT_ALTMAN_Z_MIN}")

    return StrictResult(
        ticker=ticker,
        passed=len(reasons) == 0,
        reasons_failed=reasons,
        piotroski_score=piotroski,
        altman_z_score=altman_z,
        roic=roic,
        pe_ratio=pe,
        sector=sector,
        details={**data, "margin_trend_3y": margin_trend},
    )
