from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0023_model_calibration_drift'),
    ]

    operations = [
        migrations.AddField(
            model_name='papertrade',
            name='entry_explanations',
            field=models.JSONField(blank=True, null=True),
        ),
    ]
