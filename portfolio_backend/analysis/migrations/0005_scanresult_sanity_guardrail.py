# Généré manuellement (2026-08-14, Partie 1 -- garde-fou de sécurité contre
# les cas extrêmes, cf. cas HAIN) -- environnement Django local indisponible
# pour makemigrations, mais ces opérations sont des additions simples
# (2 AddField + 1 AlterField sur les choices), sans ambiguïté de schéma.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0004_manualtickercheck'),
    ]

    operations = [
        migrations.AddField(
            model_name='scanresult',
            name='claude_confidence',
            field=models.CharField(blank=True, default='', max_length=10),
        ),
        migrations.AddField(
            model_name='scanresult',
            name='anomaly_reason',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AlterField(
            model_name='scanresult',
            name='final_verdict',
            field=models.CharField(
                choices=[
                    ('confirmed', 'confirmed'),
                    ('uncertain', 'uncertain'),
                    ('rejected_deepseek', 'rejected_deepseek'),
                    ('rejected_claude', 'rejected_claude'),
                    ('flagged_anomaly', 'flagged_anomaly'),
                ],
                db_index=True,
                max_length=20,
            ),
        ),
    ]
