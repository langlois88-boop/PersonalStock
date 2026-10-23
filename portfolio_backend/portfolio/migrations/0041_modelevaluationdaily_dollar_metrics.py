# Generated manually (split from Django's auto-generated migration) on 2026-10-22.
# Volet B3 : gain/perte moyens $ + espérance $ en complément de win_rate/avg_pnl.
# Voir docs/TECH_DEBT_NOTES.md item 27.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0040_papertrade_precision_and_tracking'),
    ]

    operations = [
        migrations.AddField(
            model_name='modelevaluationdaily',
            name='avg_gain_dollars',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='modelevaluationdaily',
            name='avg_loss_dollars',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='modelevaluationdaily',
            name='expectancy_dollars',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
