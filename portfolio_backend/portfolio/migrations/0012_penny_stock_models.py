from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0011_stock_day_low_day_high'),
    ]

    operations = [
        migrations.CreateModel(
            name='PennyStockUniverse',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('symbol', models.CharField(max_length=20, unique=True)),
                ('name', models.CharField(blank=True, max_length=200)),
                ('exchange', models.CharField(blank=True, max_length=50)),
                ('sector', models.CharField(blank=True, max_length=100)),
                ('industry', models.CharField(blank=True, max_length=120)),
                ('price', models.FloatField(blank=True, null=True)),
                ('market_cap', models.FloatField(blank=True, null=True)),
                ('volume', models.FloatField(blank=True, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('raw', models.JSONField(blank=True, default=dict)),
            ],
        ),
        migrations.CreateModel(
            name='PennyStockSnapshot',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('as_of', models.DateField()),
                ('price', models.FloatField(blank=True, null=True)),
                ('market_cap', models.FloatField(blank=True, null=True)),
                ('volume', models.FloatField(blank=True, null=True)),
                ('revenue', models.FloatField(blank=True, null=True)),
                ('debt', models.FloatField(blank=True, null=True)),
                ('cash', models.FloatField(blank=True, null=True)),
                ('burn_rate', models.FloatField(blank=True, null=True)),
                ('rsi', models.FloatField(blank=True, null=True)),
                ('macd_hist', models.FloatField(blank=True, null=True)),
                ('sentiment_score', models.FloatField(blank=True, null=True)),
                ('social_mentions', models.IntegerField(blank=True, null=True)),
                ('dilution_score', models.FloatField(blank=True, null=True)),
                ('ai_score', models.FloatField(blank=True, null=True)),
                ('flags', models.JSONField(blank=True, default=dict)),
                ('raw', models.JSONField(blank=True, default=dict)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('stock', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='portfolio.pennystockuniverse')),
            ],
            options={
                'ordering': ['-as_of', '-ai_score'],
                'unique_together': {('stock', 'as_of')},
            },
        ),
    ]
