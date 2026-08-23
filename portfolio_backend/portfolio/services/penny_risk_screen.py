"""Filtre de risque/catalyseur spécifique aux pennystocks (2026-08-20),
inspiré d'une méthodologie de due diligence micro-cap discutée avec
l'utilisateur (10-Q/10-K, Form 4, 8-K -- SEC EDGAR, gratuit et public).

Deux familles de fonctions, deux sources de données différentes :
- Cash runway / dilution ATM : yfinance (états financiers trimestriels)
  -- fonctionne pour TOUT ticker déjà supporté ailleurs dans ce projet,
  Canada (.TO/.CN) compris.
- Going concern / dette convertible toxique / 8-K récent : SEC EDGAR
  (full text search + flux de dépôts) -- couvre uniquement les émetteurs
  américains déposant auprès de la SEC. Retourne toujours des valeurs
  neutres (jamais d'exception) pour un ticker hors SEC (ex. SONI.CN),
  même discipline que _fetch_insider_buying (analysis/services/
  quant_filter.py) : un signal absent n'est jamais traité comme une
  erreur qui casse l'appelant.

Aucun appel ici ne télécharge un document de filing entier (parfois
plusieurs Mo) -- la recherche plein texte de SEC EDGAR (efts.sec.gov)
renvoie directement les correspondances, sans jamais rapatrier le HTML
complet. Résultats mis en cache (Django cache, TTL longs -- ces données
ne changent qu'au rythme trimestriel des dépôts) pour ne jamais refaire
ces appels réseau à chaque cycle de trading (15-30 min) -- voir
refresh_penny_red_flags_weekly (tasks.py) pour le rafraîchissement
périodique en arrière-plan.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
import yfinance as yf
from django.core.cache import cache

logger = logging.getLogger(__name__)

_SEC_TICKER_MAP_CACHE_KEY = "sec_ticker_cik_map"
_SEC_TICKER_MAP_TTL = 60 * 60 * 24  # 24h -- la liste des tickers SEC ne change pas d'heure en heure
_SEC_FILING_CACHE_TTL = 60 * 60 * 24  # 24h -- un nouveau dépôt trimestriel n'arrive pas plus vite que ça
_RED_FLAGS_CACHE_TTL = 60 * 60 * 24 * 7  # 7 jours -- rafraîchi par la tâche hebdomadaire, pas à chaque lookup


def _sec_headers() -> dict[str, str]:
    # La SEC exige un User-Agent identifiable (politique d'usage équitable,
    # pas juste une convention) -- un contact réel, configurable, avec un
    # défaut raisonnable si jamais rien n'est fourni.
    contact = os.getenv("SEC_EDGAR_CONTACT", "PersonalStock research contact@personalstock.local")
    return {"User-Agent": contact}


def _ticker_to_cik(ticker: str) -> str | None:
    """Table de correspondance ticker -> CIK (identifiant SEC), mise en
    cache 24h -- un seul fichier (~1 Mo) pour tous les tickers américains,
    pas la peine de le retélécharger à chaque appel."""
    mapping = cache.get(_SEC_TICKER_MAP_CACHE_KEY)
    if mapping is None:
        try:
            resp = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers=_sec_headers(), timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            mapping = {
                str(v.get("ticker", "")).upper(): str(v.get("cik_str", "")).zfill(10)
                for v in data.values() if v.get("ticker")
            }
            cache.set(_SEC_TICKER_MAP_CACHE_KEY, mapping, timeout=_SEC_TICKER_MAP_TTL)
        except Exception:
            return None
    return mapping.get(ticker.strip().upper())


def _sum_form4_purchases(xml_text: str) -> float:
    """Somme des achats en bourse ouverte (ou privés) dans un seul document
    Form 4 -- ne compte que les transactions NON dérivées (actions
    ordinaires, pas options/RSU) avec transactionCode='P' (achat) ET
    transactionAcquiredDisposedCode='A' (acquis, pas cédé). Exclut
    volontairement 'M' (levée d'option), 'F' (retenue fiscale), 'G' (don),
    etc. -- même filtre sémantique que l'ancienne logique FMP ('purchase'
    seulement, jamais 'exercise'), vérifié en direct sur un vrai Form 4
    MSFT (2026-08-23) contenant une transaction P/A de 5000 actions à
    397,35$."""
    total = 0.0
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return 0.0
    for txn in root.iter("nonDerivativeTransaction"):
        code_el = txn.find("./transactionCoding/transactionCode")
        acquired_el = txn.find("./transactionAmounts/transactionAcquiredDisposedCode/value")
        if code_el is None or (code_el.text or "").strip() != "P":
            continue
        if acquired_el is None or (acquired_el.text or "").strip() != "A":
            continue
        shares_el = txn.find("./transactionAmounts/transactionShares/value")
        price_el = txn.find("./transactionAmounts/transactionPricePerShare/value")
        try:
            shares = float(shares_el.text) if shares_el is not None else 0.0
            price = float(price_el.text) if price_el is not None else 0.0
        except (TypeError, ValueError):
            continue
        total += shares * price
    return total


def fetch_insider_buying(ticker: str) -> dict[str, Any]:
    """
    Achats d'initiés récents (90 jours), via SEC EDGAR (Form 4) --
    remplace l'ancienne implémentation Financial Modeling Prep utilisée à
    la fois par analysis/services/quant_filter.py (screener "Value +
    Catalyst"/"Insider Trading") et portfolio/views.py::_insider_summary
    (fiche ticker), après découverte en direct (2026-08-23) que l'endpoint
    insider-trading de FMP est mort pour tout le monde (v3/v4 legacy ->
    403 depuis le 31 août 2025, l'équivalent 'stable' -> 402, hors du plan
    Basic gratuit). Gratuit, direct à la source légale (un Form 4 doit
    être déposé dans les 2 jours ouvrables suivant la transaction) --
    fonction unique ici plutôt que dupliquée dans les deux appelants (les
    deux anciennes copies avaient d'ailleurs des clés de résultat
    différentes -- 'total_buy' côté views.py, 'insider_buy_total' côté
    quant_filter.py -- alors qu'elles partageaient la MÊME clé de cache :
    un vrai bug de longue date, silencieux tant que les deux endpoints
    FMP répondaient).

    Ne couvre que les émetteurs américains déposant auprès de la SEC --
    retourne le défaut 'aucun signal' pour un ticker sans CIK (ex. titres
    canadiens comme FVI.TO, SONI.CN), jamais une exception. Plafonne à 10
    Form 4 examinés par ticker sur la fenêtre de 90 jours -- suffisant
    pour capter un vrai signal d'achat cluster sans scanner des dizaines
    de dépôts pour un gros titre à forte rotation d'options (donc surtout
    des codes 'M'/'F', pas 'P').
    """
    default = {"insiders_buying": False, "insider_buy_total": 0.0}
    cache_key = f"insider_summary:{ticker}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    cik = _ticker_to_cik(ticker)
    if not cik:
        cache.set(cache_key, default, 60 * 60 * 6)
        return default

    try:
        resp = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=_sec_headers(), timeout=15,
        )
        resp.raise_for_status()
        recent = (resp.json().get("filings", {}) or {}).get("recent", {}) or {}
    except Exception:
        logger.warning("Erreur fetch SEC submissions pour %s", ticker)
        return default

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).date()

    total_buy = 0.0
    checked = 0
    for form, date_str, accession, doc in zip(forms, dates, accessions, docs):
        if form not in ("4", "4/A"):
            continue
        try:
            filing_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            continue
        if filing_date < cutoff:
            continue
        if checked >= 10:
            break
        checked += 1
        accession_nodash = accession.replace("-", "")
        cik_nolead = str(int(cik))
        basename = doc.split("/")[-1]  # le primaryDocument pointe parfois vers un sous-dossier
        # xslF345XXX/ (vue rendue HTML) -- le vrai XML est à la racine de l'accession
        xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik_nolead}/{accession_nodash}/{basename}"
        try:
            xml_resp = requests.get(xml_url, headers=_sec_headers(), timeout=15)
            if xml_resp.status_code != 200:
                continue
            total_buy += _sum_form4_purchases(xml_resp.text)
        except Exception:
            continue

    result = {
        "insiders_buying": total_buy >= 50000,
        "insider_buy_total": round(total_buy, 2),
    }
    cache.set(cache_key, result, 60 * 60 * 6)
    return result


def _recent_filing(cik: str, form_type: str) -> dict[str, Any] | None:
    """Métadonnées (numéro d'accession, date) du dépôt le plus récent d'un
    type donné (ex. '10-Q', '10-K', '8-K'). None si le CIK n'a jamais
    déposé ce type de formulaire, ou en cas d'échec réseau."""
    cache_key = f"sec_recent_filing:{cik}:{form_type}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached or None
    try:
        resp = requests.get(
            "https://www.sec.gov/cgi-bin/browse-edgar",
            params={
                "action": "getcompany", "CIK": cik, "type": form_type,
                "dateb": "", "owner": "include", "count": "1", "output": "atom",
            },
            headers=_sec_headers(), timeout=15,
        )
        resp.raise_for_status()
        acc_match = re.search(r"<accession-number>([^<]+)</accession-number>", resp.text)
        date_match = re.search(r"<filing-date>([^<]+)</filing-date>", resp.text)
        if not acc_match:
            cache.set(cache_key, False, timeout=_SEC_FILING_CACHE_TTL)  # False = "vérifié, aucun dépôt"
            return None
        result = {"accession": acc_match.group(1), "filing_date": date_match.group(1) if date_match else None}
        cache.set(cache_key, result, timeout=_SEC_FILING_CACHE_TTL)
        return result
    except Exception:
        return None


def _filing_contains_phrase(cik: str, accession: str, phrase: str, forms: str) -> bool | None:
    """Vrai si `phrase` apparaît dans le dépôt identifié par `accession` --
    via la recherche plein texte SEC EDGAR (renvoie des correspondances,
    jamais le document complet). None si la recherche elle-même échoue
    (pas la même chose qu'un "non trouvé" confirmé)."""
    try:
        resp = requests.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={"q": f'"{phrase}"', "forms": forms, "ciks": cik},
            headers=_sec_headers(), timeout=15,
        )
        resp.raise_for_status()
        hits = (resp.json().get("hits", {}) or {}).get("hits", []) or []
        for hit in hits:
            hit_adsh = ((hit.get("_source", {}) or {}).get("adsh", "") or "")
            if hit_adsh == accession:
                return True
        return False
    except Exception:
        return None


def check_going_concern(ticker: str) -> dict[str, Any]:
    """Le dernier 10-Q (ou 10-K si plus récent) contient-il un
    avertissement de continuité de l'exploitation ("going concern") ?
    Le signal d'alarme comptable le plus grave selon la méthodologie
    -- mais reste une INFORMATION, pas un blocage automatique ici (voir
    tasks.py::_execute_paper_trades_for_sandbox pour comment c'est
    utilisé côté AI_PENNY)."""
    default = {"going_concern_risk": None, "filing_date": None, "source": None}
    cik = _ticker_to_cik(ticker)
    if not cik:
        return default
    filing = _recent_filing(cik, "10-Q") or _recent_filing(cik, "10-K")
    if not filing:
        return default
    found = _filing_contains_phrase(cik, filing["accession"], "going concern", "10-Q,10-K")
    return {
        "going_concern_risk": found,
        "filing_date": filing.get("filing_date"),
        "source": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}",
    }


def check_toxic_convertible_debt(ticker: str) -> dict[str, Any]:
    """Le dernier 10-Q/10-K mentionne-t-il une clause de conversion à prix
    variable/flottant ("variable conversion price") -- la signature d'une
    dette convertible toxique ("death spiral debt")."""
    default = {"toxic_convertible_debt": None, "filing_date": None, "source": None}
    cik = _ticker_to_cik(ticker)
    if not cik:
        return default
    filing = _recent_filing(cik, "10-Q") or _recent_filing(cik, "10-K")
    if not filing:
        return default
    found = _filing_contains_phrase(cik, filing["accession"], "variable conversion price", "10-Q,10-K")
    return {
        "toxic_convertible_debt": found,
        "filing_date": filing.get("filing_date"),
        "source": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}",
    }


def check_recent_8k_catalyst(ticker: str, days: int = 14) -> dict[str, Any]:
    """Un formulaire 8-K (événement matériel important -- partenariat,
    résultat clinique, financement...) a-t-il été déposé dans les
    `days` derniers jours ? Un 8-K récent est un signal factuel qu'un
    vrai événement vient de se produire, sans préjuger s'il est positif
    ou négatif -- à vérifier manuellement (voir 'Analyse manuelle
    complète' dans l'app)."""
    default = {"recent_8k": False, "filing_date": None, "days_ago": None, "source": None}
    cik = _ticker_to_cik(ticker)
    if not cik:
        return default
    filing = _recent_filing(cik, "8-K")
    if not filing or not filing.get("filing_date"):
        return default
    try:
        filed = datetime.strptime(filing["filing_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - filed).days
    except Exception:
        return default
    return {
        "recent_8k": age_days <= days,
        "filing_date": filing["filing_date"],
        "days_ago": age_days,
        "source": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K",
    }


def compute_cash_runway_quarters(ticker: str) -> dict[str, Any]:
    """Trésorerie disponible / combustion de cash trimestrielle -- voir
    docstring du module pour la formule. yfinance (pas SEC) : fonctionne
    pour n'importe quel ticker déjà supporté ailleurs dans ce projet,
    Canada (.TO/.CN) compris."""
    default = {"cash_runway_quarters": None, "cash_available": None, "quarterly_burn": None}
    try:
        t = yf.Ticker(ticker)
        bs = t.quarterly_balance_sheet
        cf = t.quarterly_cashflow
        if bs is None or bs.empty or cf is None or cf.empty:
            return default
        cash_row = None
        for label in ("Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"):
            if label in bs.index:
                cash_row = bs.loc[label]
                break
        if cash_row is None:
            return default
        cash_available = next((float(v) for v in cash_row if v == v and v is not None), None)  # v==v exclut NaN
        if cash_available is None:
            return default

        op_cf_row = cf.loc["Operating Cash Flow"] if "Operating Cash Flow" in cf.index else None
        if op_cf_row is None:
            return default
        quarterly_ocf = next((float(v) for v in op_cf_row if v == v and v is not None), None)
        if quarterly_ocf is None:
            return default

        capex = 0.0
        if "Capital Expenditure" in cf.index:
            capex_val = next((float(v) for v in cf.loc["Capital Expenditure"] if v == v and v is not None), 0.0)
            capex = capex_val or 0.0

        quarterly_burn = -(quarterly_ocf + capex)  # capex est déjà négatif dans yfinance
        if quarterly_burn <= 0:
            # L'entreprise génère du cash (ou est à l'équilibre) ce
            # trimestre -- pas de "runway" à calculer, ce n'est pas le
            # profil de risque que cette fonction cherche à mesurer.
            return {"cash_runway_quarters": None, "cash_available": round(cash_available, 2), "quarterly_burn": round(quarterly_burn, 2)}

        return {
            "cash_runway_quarters": round(cash_available / quarterly_burn, 2),
            "cash_available": round(cash_available, 2),
            "quarterly_burn": round(quarterly_burn, 2),
        }
    except Exception:
        return default


def detect_atm_dilution(ticker: str) -> dict[str, Any]:
    """Croissance du nombre d'actions en circulation sur 4 trimestres --
    un proxy quantitatif de la dilution ATM (financement continu par
    émission d'actions au marché), sans avoir besoin de parser le texte
    des dépôts pour trouver le mot "ATM" explicitement. Un taux élevé
    (>10-15% sur 4 trimestres) est le signe d'une dilution active."""
    default = {"shares_growth_4q_pct": None, "shares_current": None, "shares_4q_ago": None}
    try:
        t = yf.Ticker(ticker)
        bs = t.quarterly_balance_sheet
        if bs is None or bs.empty:
            return default
        row = None
        for label in ("Ordinary Shares Number", "Share Issued"):
            if label in bs.index:
                row = bs.loc[label]
                break
        if row is None:
            return default
        values = [float(v) for v in row if v == v and v is not None]
        if len(values) < 2:
            return default
        current = values[0]
        oldest = values[min(3, len(values) - 1)]  # jusqu'à 4 trimestres en arrière, ou le plus ancien dispo
        if not oldest or oldest <= 0:
            return default
        growth_pct = round(((current / oldest) - 1.0) * 100, 2)
        return {"shares_growth_4q_pct": growth_pct, "shares_current": current, "shares_4q_ago": oldest}
    except Exception:
        return default


def penny_red_flags_summary(ticker: str) -> dict[str, Any]:
    """Point d'entrée unique -- combine les 5 vérifications. Utilisé par
    refresh_penny_red_flags_weekly (tasks.py) pour peupler le cache que
    la boucle de décision AI_PENNY lit ensuite (jamais d'appel réseau SEC
    depuis le chemin de trading rapide lui-même)."""
    return {
        "ticker": ticker.strip().upper(),
        "going_concern": check_going_concern(ticker),
        "toxic_convertible_debt": check_toxic_convertible_debt(ticker),
        "recent_8k_catalyst": check_recent_8k_catalyst(ticker),
        "cash_runway": compute_cash_runway_quarters(ticker),
        "atm_dilution": detect_atm_dilution(ticker),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def get_cached_red_flags(ticker: str) -> dict[str, Any] | None:
    """Lecture seule, rapide -- utilisée par la boucle de décision AI_PENNY.
    None si jamais rafraîchi pour ce ticker (pas encore passé par
    refresh_penny_red_flags_weekly), pas une erreur."""
    return cache.get(f"penny_red_flags:{ticker.strip().upper()}")


def refresh_and_cache_red_flags(ticker: str) -> dict[str, Any]:
    """Calcule et met en cache (7 jours) -- appelé par la tâche
    hebdomadaire, jamais par la boucle de trading elle-même."""
    result = penny_red_flags_summary(ticker)
    cache.set(f"penny_red_flags:{ticker.strip().upper()}", result, timeout=_RED_FLAGS_CACHE_TTL)
    return result
