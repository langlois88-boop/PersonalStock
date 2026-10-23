# Volet A3 (2026-10) : backfill ponctuel et réversible des 20
# FundamentalLabPosition dont benchmark_price_at_entry est resté à
# Decimal("0") (échec transitoire de la cotation du jour le 2026-09-01,
# PAS un marché fermé -- ce jour-là est un mardi, jour de bourse normal,
# yfinance a une clôture ^GSPTSE disponible directement). Voir
# docs/TECH_DEBT_NOTES.md pour le détail du dry-run (20 valeurs vérifiées
# avant écriture, toutes égales à la même clôture ^GSPTSE du 2026-09-01 :
# 35825.6992 -- days_back=0 pour les 20).
#
# Réversible : reverse_code remet ces mêmes 20 lignes à Decimal("0").
# Portée volontairement restreinte aux 20 IDs vérifiés un par un dans le
# dry-run -- ne touche aucune autre ligne même si une future ligne
# retombe à 0 pour une autre raison.

from decimal import Decimal

from django.db import migrations

BACKFILL_IDS = [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84]
PROPOSED_VALUE = Decimal("35825.6992")


def backfill_benchmark(apps, schema_editor):
    FundamentalLabPosition = apps.get_model('analysis', 'FundamentalLabPosition')
    qs = FundamentalLabPosition.objects.filter(id__in=BACKFILL_IDS, benchmark_price_at_entry=0)
    updated = qs.update(benchmark_price_at_entry=PROPOSED_VALUE)
    # 0 -- rien à faire ici (base de test vide, ou déjà appliqué ailleurs) :
    # pas une erreur. Un nombre PARTIEL (ni 0 ni les 20) l'est : au moins
    # une des lignes vérifiées dans le dry-run n'est plus à 0 -- arrêt pour
    # investigation plutôt que d'écraser une valeur apparue depuis.
    if updated not in (0, len(BACKFILL_IDS)):
        raise RuntimeError(
            f"Backfill A3 : {updated}/{len(BACKFILL_IDS)} lignes mises à jour -- "
            "au moins une des 20 n'était plus à 0 (déjà modifiée depuis le dry-run?). "
            "Arrêt pour investigation manuelle plutôt que d'écraser une valeur inattendue."
        )


def revert_benchmark(apps, schema_editor):
    FundamentalLabPosition = apps.get_model('analysis', 'FundamentalLabPosition')
    FundamentalLabPosition.objects.filter(id__in=BACKFILL_IDS).update(benchmark_price_at_entry=Decimal("0"))


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0008_alter_fundamentallabposition_close_reason'),
    ]

    operations = [
        migrations.RunPython(backfill_benchmark, revert_benchmark),
    ]
