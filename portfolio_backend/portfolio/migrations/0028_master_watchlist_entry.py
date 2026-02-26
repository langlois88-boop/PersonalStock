from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0027_active_signal'),
    ]

    operations = [
        migrations.CreateModel(
            name='MasterWatchlistEntry',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('symbol', models.CharField(db_index=True, max_length=12, unique=True)),
                ('category', models.CharField(choices=[('HIGH_VOL', 'High Volatility'), ('WEAK_SHORT', 'Weak Stocks'), ('SWING', 'Swing Pipeline')], db_index=True, max_length=20)),
                ('stop_loss_pct', models.FloatField(blank=True, null=True)),
                ('high_risk', models.BooleanField(default=False)),
                ('block_if_market_sentiment_lt', models.FloatField(blank=True, null=True)),
                ('volume_scan_enabled', models.BooleanField(default=False)),
                ('source', models.CharField(blank=True, default='', max_length=120)),
                ('notes', models.TextField(blank=True, default='')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'ordering': ['category', 'symbol'],
            },
        ),
    ]
