from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0027_active_signal'),
    ]

    operations = [
        migrations.CreateModel(
            name='SystemLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.DateTimeField(auto_now_add=True, db_index=True)),
                ('category', models.CharField(choices=[('AI_PENNY', 'IA Penny'), ('AI_BLUECHIP', 'IA Bluechip'), ('AI_CRYPTO', 'IA Crypto'), ('PAPER_TRADE', 'Paper Trade'), ('SYSTEM', 'System'), ('TELEGRAM', 'Telegram')], db_index=True, max_length=20)),
                ('level', models.CharField(choices=[('INFO', 'INFO'), ('SUCCESS', 'SUCCESS'), ('WARNING', 'WARNING'), ('ERROR', 'ERROR')], db_index=True, max_length=10)),
                ('symbol', models.CharField(blank=True, db_index=True, max_length=20, null=True)),
                ('message', models.TextField()),
                ('metadata', models.JSONField(blank=True, null=True)),
            ],
            options={
                'ordering': ['-timestamp'],
            },
        ),
        migrations.AddIndex(
            model_name='systemlog',
            index=models.Index(fields=['category', 'timestamp'], name='systemlog_cat_ts'),
        ),
        migrations.AddIndex(
            model_name='systemlog',
            index=models.Index(fields=['level', 'timestamp'], name='systemlog_lvl_ts'),
        ),
    ]
