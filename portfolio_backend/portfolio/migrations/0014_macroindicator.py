from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0013_pennystocksnapshot_shares_outstanding'),
    ]

    operations = [
        migrations.CreateModel(
            name='MacroIndicator',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField(unique=True)),
                ('sp500_close', models.FloatField()),
                ('vix_index', models.FloatField()),
                ('interest_rate_10y', models.FloatField()),
                ('inflation_rate', models.FloatField()),
                ('oil_price', models.FloatField(blank=True, null=True)),
            ],
            options={
                'ordering': ['-date'],
            },
        ),
    ]
