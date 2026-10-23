# Generated manually (split from Django's auto-generated migration) on 2026-10-22.
# Volet A : précision des prix (A1) + traçabilité de la position d'origine (A2).
# Voir docs/TECH_DEBT_NOTES.md items 25-26.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0039_papertrade_stop_order_id'),
    ]

    operations = [
        migrations.AddField(
            model_name='papertrade',
            name='initial_quantity',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='papertrade',
            name='initial_stop_loss',
            field=models.DecimalField(blank=True, decimal_places=6, max_digits=14, null=True),
        ),
        migrations.AddField(
            model_name='papertrade',
            name='realized_partial_pnl',
            field=models.DecimalField(decimal_places=2, default=0, max_digits=10),
        ),
        migrations.AlterField(
            model_name='papertrade',
            name='entry_price',
            field=models.DecimalField(decimal_places=6, max_digits=14),
        ),
        migrations.AlterField(
            model_name='papertrade',
            name='exit_price',
            field=models.DecimalField(blank=True, decimal_places=6, max_digits=14, null=True),
        ),
        migrations.AlterField(
            model_name='papertrade',
            name='stop_loss',
            field=models.DecimalField(decimal_places=6, max_digits=14),
        ),
        migrations.AlterField(
            model_name='papertrade',
            name='take_profit',
            field=models.DecimalField(blank=True, decimal_places=6, max_digits=14, null=True),
        ),
    ]
