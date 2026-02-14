from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0012_penny_stock_models'),
    ]

    operations = [
        migrations.AddField(
            model_name='pennystocksnapshot',
            name='shares_outstanding',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
