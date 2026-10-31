# 2026-10-30 : correction réversible des deux PaperTrade (AXG id=866,
# ATYR id=868, sandbox AI_PENNY) corrompus par des reverse splits le
# 2026-10-29 (AXG 1-pour-200, ATYR 1-pour-40) jamais gérés par
# _execute_paper_trades_for_sandbox (voir la nouvelle fonction
# _apply_sandbox_split_adjustment_if_needed, portfolio/tasks.py, pour le
# correctif préventif sur les positions encore ouvertes -- ces deux-là
# étaient déjà fermées au moment de la découverte).
#
# Méthode : entry_price/stop_loss/initial_stop_loss/quantity/
# initial_quantity divisés (ou multipliés pour quantity) par le ratio de
# split -- exactement la même formule que analysis/tasks.py::
# _apply_split_adjustment_if_needed (cas BRCC, 2026-08-24). exit_price
# n'est PAS touché : c'est un prix réel post-split, déjà correct. pnl et
# realized_partial_pnl sont RECALCULÉS à partir de ces valeurs corrigées
# et du exit_price réel -- ce n'est pas une invention de ce que le bot
# "aurait dû" décider, seulement la correction d'une erreur d'unité qui
# faussait la comparaison entry/exit ; les prix de vente réellement
# enregistrés (exit_price, et le prix implicite de la vente partielle)
# ne changent pas.
#
# Vérifié avant écriture : pour chacune, la vente partielle et la
# clôture finale ont eu lieu quasiment au même prix (reconstruit depuis
# l'ancien realized_partial_pnl -- 4.1502$ pour AXG, 5.0400$ pour ATYR --
# cohérent avec exit_price à la précision d'affichage), donc le pnl total
# corrigé = 2x le pnl d'une moitié (même quantité des deux côtés).
#
# Réversible : reverse_code remet les valeurs d'origine (corrompues),
# conservées ci-dessous.

from decimal import Decimal

from django.db import migrations

ORIGINAL = {
    866: {
        'entry_price': Decimal('0.029900'), 'stop_loss': Decimal('0.028320'),
        'initial_stop_loss': Decimal('0.009918'), 'quantity': 16722.0,
        'initial_quantity': 33444.0, 'pnl': Decimal('137792.62'),
        'realized_partial_pnl': Decimal('68896.31'),
    },
    868: {
        'entry_price': Decimal('0.126900'), 'stop_loss': Decimal('0.121824'),
        'initial_stop_loss': Decimal('0.099675'), 'quantity': 3940.0,
        'initial_quantity': 7880.0, 'pnl': Decimal('38715.23'),
        'realized_partial_pnl': Decimal('19357.61'),
    },
}

CORRECTED = {
    866: {  # AXG, ratio 0.005 (reverse split 1-pour-200, 2026-10-29)
        'entry_price': Decimal('5.980000'), 'stop_loss': Decimal('5.664000'),
        'initial_stop_loss': Decimal('1.983600'), 'quantity': 83.61,
        'initial_quantity': 167.22, 'pnl': Decimal('-306.01'),
        'realized_partial_pnl': Decimal('-153.01'),
    },
    868: {  # ATYR, ratio 0.025 (reverse split 1-pour-40, 2026-10-29)
        'entry_price': Decimal('5.076000'), 'stop_loss': Decimal('4.872960'),
        'initial_stop_loss': Decimal('3.987000'), 'quantity': 98.5,
        'initial_quantity': 197.0, 'pnl': Decimal('-7.09'),
        'realized_partial_pnl': Decimal('-3.55'),
    },
}

SPLIT_NOTE_SUFFIX = (
    " | Split détecté rétroactivement (AXG 1-pour-200 / ATYR 1-pour-40, "
    "2026-10-29) -- entry_price/stop_loss/quantity/pnl corrigés le "
    "2026-10-30, voir docs/TECH_DEBT_NOTES.md."
)


def _apply(apps, from_values, to_values, note_suffix):
    PaperTrade = apps.get_model('portfolio', 'PaperTrade')
    updated = 0
    for trade_id, new_values in to_values.items():
        expected_current_entry_price = from_values[trade_id]['entry_price']
        trade = PaperTrade.objects.filter(id=trade_id, entry_price=expected_current_entry_price).first()
        if trade is None:
            continue
        for field, value in new_values.items():
            setattr(trade, field, value)
        update_fields = list(new_values.keys())
        if note_suffix:
            trade.notes = f"{trade.notes or ''}{note_suffix}"
            update_fields.append('notes')
        trade.save(update_fields=update_fields)
        updated += 1
    if updated not in (0, len(to_values)):
        raise RuntimeError(
            f"Correction split AXG/ATYR : {updated}/{len(to_values)} lignes mises à jour -- "
            "au moins une n'avait plus la valeur attendue (entry_price). Arrêt pour investigation."
        )


def apply_correction(apps, schema_editor):
    _apply(apps, from_values=ORIGINAL, to_values=CORRECTED, note_suffix=SPLIT_NOTE_SUFFIX)


def revert_correction(apps, schema_editor):
    _apply(apps, from_values=CORRECTED, to_values=ORIGINAL, note_suffix=None)


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0041_modelevaluationdaily_dollar_metrics'),
    ]

    operations = [
        migrations.RunPython(apply_correction, revert_correction),
    ]
