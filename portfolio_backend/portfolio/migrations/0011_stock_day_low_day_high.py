from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0010_pennysignal'),
    ]

    operations = [
        migrations.AddField(
            model_name='stock',
            name='day_low',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='stock',
            name='day_high',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
