from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0031_alter_papertrade_broker'),
    ]

    operations = [
        migrations.AddField(
            model_name='prediction',
            name='confidence',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
