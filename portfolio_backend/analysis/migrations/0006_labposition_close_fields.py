# Généré manuellement (2026-08-14, Partie 2 -- structurer la collecte pour
# un futur modèle ML, cf. docs/ML_LAB_FUTURE_MODEL.md) -- additions simples,
# aucune ambiguïté de schéma.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0005_scanresult_sanity_guardrail'),
    ]

    operations = [
        migrations.AddField(
            model_name='fundamentallabposition',
            name='exit_price',
            field=models.DecimalField(blank=True, decimal_places=4, max_digits=12, null=True),
        ),
        migrations.AddField(
            model_name='fundamentallabposition',
            name='exit_date',
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='fundamentallabposition',
            name='close_reason',
            field=models.CharField(
                blank=True, default='', max_length=20,
                choices=[('stop_loss', 'stop_loss'), ('manual', 'manual')],
            ),
        ),
        migrations.AddField(
            model_name='fundamentallabposition',
            name='realized_return_pct',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
