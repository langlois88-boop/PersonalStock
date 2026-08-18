"""Tests pour le filtre strict du balayage large hebdomadaire
(services/broad_scan_filter.py). Première suite de tests pour ce module --
evaluate_ticker_strict n'avait aucune couverture avant.

Contexte du test principal (2026-08-18) : STRICT_FCF_YIELD_MIN abaissé de
6.0 à 5.0 après vérification manuelle en direct sur un échantillon spread de
35/720 tickers de l'univers réel (voir commentaire dans broad_scan_filter.py)
-- INTU (FCF Yield 5.46%, seule raison d'échec à 6%) aurait été repêché,
sans affecter les autres tickers proches de la frontière qui échouaient déjà
pour d'autres critères.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
from django.test import TestCase

from analysis.services import broad_scan_filter as bsf


def _build_passing_stmts() -> dict:
    """Construit des états financiers synthétiques sur 2 ans où les 7
    critères stricts AUTRES que le FCF Yield passent tous confortablement
    (Piotroski=9/9, Altman Z~4.8, ROIC~14%, marge en croissance) -- pour
    isoler l'effet du seuil FCF Yield dans les tests ci-dessous.
    """
    bs = pd.DataFrame(
        {
            "y0": [1050.0, 250.0, 500.0, 200.0, 100.0, 500.0, 400.0, 800.0],
            "y1": [1000.0, 300.0, 400.0, 200.0, 100.0, 450.0, 350.0, 750.0],
        },
        index=[
            "Total Assets", "Long Term Debt", "Current Assets", "Current Liabilities",
            "Ordinary Shares Number", "Total Liabilities Net Minority Interest",
            "Retained Earnings", "Invested Capital",
        ],
    )
    inc = pd.DataFrame(
        {
            "y0": [100.0, 400.0, 1100.0, 150.0],
            "y1": [80.0, 350.0, 1000.0, 120.0],
        },
        index=["Net Income", "Gross Profit", "Total Revenue", "EBIT"],
    )
    cf = pd.DataFrame(
        {"y0": [120.0], "y1": [90.0]},
        index=["Operating Cash Flow"],
    )
    return {"info": {}, "bs": bs, "inc": inc, "cf": cf, "years": ["y0", "y1"]}


def _passing_data(fcf_yield: float) -> dict:
    return {
        "sector": "DEFAULT",
        "roe": 20.0,           # > STRICT_ROE_MIN (15.0)
        "debt_to_ebitda": 1.0,  # < seuil secteur DEFAULT (3x)
        "pe_ratio": 10.0,       # < sector_avg_pe passé en argument (15.0)
        "peg_ratio": 1.0,       # < STRICT_PEG_MAX (1.2)
        "fcf_yield": fcf_yield,
        "market_cap": 2000.0,
    }


class StrictFcfYieldThresholdTests(TestCase):
    def test_threshold_constant_is_5_percent(self):
        # Verrouille la valeur pour éviter une régression silencieuse si
        # quelqu'un retouche ce fichier plus tard sans lire le commentaire.
        self.assertEqual(bsf.STRICT_FCF_YIELD_MIN, 5.0)

    def test_ticker_with_fcf_yield_between_5_and_6_percent_now_passes(self):
        """C'est exactement le cas INTU qui a motivé le changement de seuil :
        FCF Yield 5.46% -- passait à aucun seuil avant, doit passer maintenant."""
        stmts = _build_passing_stmts()
        data = _passing_data(fcf_yield=5.46)
        with patch.object(bsf, "fetch_annual_statements", return_value=stmts):
            result = bsf.evaluate_ticker_strict("INTU", sector_avg_pe=15.0, data=data)

        self.assertTrue(result.passed, f"reasons_failed={result.reasons_failed}")
        self.assertEqual(result.reasons_failed, [])
        self.assertEqual(result.piotroski_score, 9)

    def test_ticker_with_fcf_yield_below_5_percent_still_fails(self):
        """Le seuil abaissé n'ouvre pas la porte en grand -- un FCF Yield
        franchement sous la nouvelle barre continue de bloquer le ticker,
        même si tous les 7 autres critères stricts passent."""
        stmts = _build_passing_stmts()
        data = _passing_data(fcf_yield=4.5)
        with patch.object(bsf, "fetch_annual_statements", return_value=stmts):
            result = bsf.evaluate_ticker_strict("VRSK", sector_avg_pe=15.0, data=data)

        self.assertFalse(result.passed)
        self.assertTrue(any("FCF Yield" in r for r in result.reasons_failed))

    def test_ticker_with_fcf_yield_at_old_6_percent_threshold_still_passes(self):
        """Non-régression : un ticker qui passait déjà à l'ancien seuil (6%)
        continue de passer -- le changement n'est qu'additif."""
        stmts = _build_passing_stmts()
        data = _passing_data(fcf_yield=8.25)  # ex. FVI.TO
        with patch.object(bsf, "fetch_annual_statements", return_value=stmts):
            result = bsf.evaluate_ticker_strict("FVI.TO", sector_avg_pe=15.0, data=data)

        self.assertTrue(result.passed, f"reasons_failed={result.reasons_failed}")

    def test_other_strict_criteria_still_enforced_independently_of_fcf_yield(self):
        """Un FCF Yield confortable ne suffit pas seul -- les 7 autres
        garde-fous restent des ET obligatoires (pas de régression accidentelle
        vers un filtre plus laxiste dans son ensemble)."""
        stmts = _build_passing_stmts()
        data = _passing_data(fcf_yield=8.0)
        data["peg_ratio"] = 3.0  # dépasse largement STRICT_PEG_MAX (1.2)
        with patch.object(bsf, "fetch_annual_statements", return_value=stmts):
            result = bsf.evaluate_ticker_strict("XYZ", sector_avg_pe=15.0, data=data)

        self.assertFalse(result.passed)
        self.assertTrue(any("PEG" in r for r in result.reasons_failed))
