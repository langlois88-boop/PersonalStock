from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0024_papertrade_entry_explanations'),
    ]

    operations = [
        migrations.AddIndex(
            model_name='papertrade',
            index=models.Index(fields=['sandbox', 'status', 'entry_date'], name='papertrade_sbx_status_dt'),
        ),
        migrations.AddIndex(
            model_name='modelevaluationdaily',
            index=models.Index(fields=['as_of', 'sandbox'], name='modeleval_asof_sandbox'),
        ),
    ]
