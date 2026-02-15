from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0016_papertrade_entry_features_papertrade_entry_signal'),
    ]

    operations = [
        migrations.AddField(
            model_name='papertrade',
            name='sandbox',
            field=models.CharField(
                choices=[('WATCHLIST', 'WATCHLIST'), ('AI_BLUECHIP', 'AI_BLUECHIP'), ('AI_PENNY', 'AI_PENNY')],
                db_index=True,
                default='WATCHLIST',
                max_length=20,
            ),
        ),
    ]
