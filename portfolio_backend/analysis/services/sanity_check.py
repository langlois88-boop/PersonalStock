"""
Garde-fou de sécurité contre les cas extrêmes (2026-08-14, déclenché par le
cas HAIN) : un verdict LLM confirmed/uncertain peut ouvrir une position même
quand les ratios sous-jacents sont statistiquement suspects. Sur HAIN, Claude
avait lui-même noté dans son raisonnement que ROE -113%, FCF Yield 234% et
Dette/EBITDA 7.12x étaient anormaux — sans que ça change son verdict, parce
que l'API Claude n'a AUCUNE mémoire d'un appel à l'autre : rien n'empêche le
même jugement "ambigu mais je penche pour confirmed/uncertain" de se
reproduire indéfiniment sur des cas similaires à l'avenir. Un LLM ne peut pas
"apprendre" de ce genre d'incident tout seul — un garde-fou par règles,
déterministe, est nécessaire en complément.

Appliqué APRÈS le verdict Claude (LLM) mais AVANT toute création de position
réelle (_create_lab_position) — indépendant du verdict : même un "confirmed"
peut être bloqué ici. Voir analysis/tasks.py::_create_lab_position et
docs/TECH_DEBT_NOTES.md.

Les seuils ci-dessous sont des constantes nommées et documentées, pas des
nombres magiques dispersés dans le code — à ajuster ici si l'expérience
montre qu'ils sont trop ou pas assez stricts, sans avoir à chercher où ils
sont utilisés.
"""

from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# ROE < -30% : une perte massive et structurelle, pas juste un trimestre
# faible (un ROE négatif ponctuel à -5/-10% n'est pas en soi anormal pour
# une entreprise en investissement lourd ou en restructuration passagère).
ROE_ANOMALY_THRESHOLD = -30.0  # %

# FCF Yield > 50% : quasiment toujours un artefact de calcul (FCF gonflé
# ponctuellement par un désinvestissement, une variation de BFR, une
# capitalisation boursière écrasée par une baisse récente, etc.) plutôt
# qu'un vrai signal de sous-évaluation — cas HAIN : 234%, largement au-delà
# de ce qu'un free cash flow yield légitime peut atteindre.
FCF_YIELD_ANOMALY_THRESHOLD = 50.0  # %

# Dette/EBITDA : levier jugé extrême au-delà de 2x le seuil sectoriel
# "normal" déjà calculé par quant_filter.merge_thresholds (stocké dans
# quant_details['thresholds_used']['debt_ebitda_max']) — volontairement pas
# un seuil fixe unique, parce que le seuil normal varie fortement par
# secteur (Utilities tolère jusqu'à 6x, Technology seulement 1.5x) : un
# multiplicateur du seuil sectoriel reste cohérent avec cette réalité.
DEBT_EBITDA_ANOMALY_MULTIPLIER = 2.0


@dataclass
class SanityCheckResult:
    passed: bool
    reasons: list = field(default_factory=list)

    @property
    def reason_text(self) -> str:
        return "; ".join(self.reasons)


def check_sanity(quant_details: dict) -> SanityCheckResult:
    """
    Applique les seuils d'anomalie sur les ratios bruts déjà calculés par
    quant_filter (quant_details, tel que stocké sur ScanResult.quant_details).
    Ne fait aucun appel réseau, aucune dépendance au verdict LLM — une
    fonction pure sur les données déjà en main.

    Retourne passed=False dès qu'AU MOINS un seuil est déclenché, avec le
    détail de chaque déclenchement dans `reasons` (plusieurs peuvent être
    vrais en même temps, ex. HAIN déclenchait à la fois ROE et FCF Yield).
    """
    quant_details = quant_details or {}
    reasons: list = []

    roe = quant_details.get("roe")
    fcf_yield = quant_details.get("fcf_yield")
    pe_ratio = quant_details.get("pe_ratio")
    debt_to_ebitda = quant_details.get("debt_to_ebitda")
    thresholds_used = quant_details.get("thresholds_used") or {}
    debt_ebitda_max = thresholds_used.get("debt_ebitda_max")

    if roe is not None and roe < ROE_ANOMALY_THRESHOLD:
        reasons.append(
            f"ROE {roe:.1f}% < seuil d'anomalie {ROE_ANOMALY_THRESHOLD:.0f}% (perte massive)"
        )

    if fcf_yield is not None and fcf_yield > FCF_YIELD_ANOMALY_THRESHOLD:
        reasons.append(
            f"FCF Yield {fcf_yield:.1f}% > seuil d'anomalie {FCF_YIELD_ANOMALY_THRESHOLD:.0f}% "
            "(statistiquement suspect, probable artefact de calcul)"
        )

    if debt_to_ebitda is not None and debt_ebitda_max:
        anomaly_debt_max = debt_ebitda_max * DEBT_EBITDA_ANOMALY_MULTIPLIER
        if debt_to_ebitda > anomaly_debt_max:
            reasons.append(
                f"Dette/EBITDA {debt_to_ebitda}x > {DEBT_EBITDA_ANOMALY_MULTIPLIER:.0f}x le seuil "
                f"sectoriel normal ({anomaly_debt_max:.1f}x, levier extrême)"
            )

    # Absence simultanée de P/E ET de FCF Yield, avec un ROE négatif : signal
    # qu'il n'y a probablement pas de données fondamentales fiables du tout
    # pour ce ticker, plutôt qu'une vraie inefficience de marché.
    if pe_ratio is None and fcf_yield is None and roe is not None and roe < 0:
        reasons.append(
            "P/E absent, FCF Yield absent et ROE négatif simultanément -- "
            "données fondamentales probablement non fiables pour ce ticker"
        )

    return SanityCheckResult(passed=len(reasons) == 0, reasons=reasons)
