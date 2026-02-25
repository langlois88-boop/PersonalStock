from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("portfolio", "0027_active_signal"),
    ]

    operations = [
        migrations.AddField(
            model_name="papertrade",
            name="broker",
            field=models.CharField(choices=[("SIM", "SIM"), ("ALPACA", "ALPACA")], default="SIM", max_length=20, db_index=True),
        ),
        migrations.AddField(
            model_name="papertrade",
            name="broker_order_id",
            field=models.CharField(blank=True, default="", max_length=120),
        ),
        migrations.AddField(
            model_name="papertrade",
            name="broker_status",
            field=models.CharField(blank=True, default="", max_length=40),
        ),
        migrations.AddField(
            model_name="papertrade",
            name="broker_side",
            field=models.CharField(blank=True, default="", max_length=10),
        ),
        migrations.AddField(
            model_name="papertrade",
            name="broker_filled_qty",
            field=models.DecimalField(blank=True, decimal_places=4, max_digits=12, null=True),
        ),
        migrations.AddField(
            model_name="papertrade",
            name="broker_avg_price",
            field=models.DecimalField(blank=True, decimal_places=4, max_digits=12, null=True),
        ),
        migrations.AddField(
            model_name="papertrade",
            name="broker_updated_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
