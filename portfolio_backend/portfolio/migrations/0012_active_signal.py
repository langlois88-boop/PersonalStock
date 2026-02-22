from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0026_alter_newsarticle_url_alter_stocknews_url'),
    ]

    operations = [
        migrations.CreateModel(
            name='ActiveSignal',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ticker', models.CharField(db_index=True, max_length=12)),
                ('pattern', models.CharField(blank=True, default='', max_length=80)),
                ('rvol', models.FloatField(blank=True, null=True)),
                ('entry_price', models.FloatField()),
                ('target_price', models.FloatField()),
                ('stop_loss', models.FloatField()),
                ('confidence', models.FloatField(blank=True, null=True)),
                ('status', models.CharField(choices=[('OPEN', 'OPEN'), ('TARGET', 'TARGET'), ('STOP', 'STOP'), ('TIMEOUT', 'TIMEOUT'), ('CLOSED', 'CLOSED')], db_index=True, default='OPEN', max_length=10)),
                ('opened_at', models.DateTimeField(auto_now_add=True)),
                ('closed_at', models.DateTimeField(blank=True, null=True)),
                ('closed_price', models.FloatField(blank=True, null=True)),
                ('outcome', models.CharField(blank=True, default='', max_length=10)),
                ('liquidity_note', models.CharField(blank=True, default='', max_length=120)),
                ('meta', models.JSONField(blank=True, default=dict)),
            ],
            options={
                'ordering': ['-opened_at'],
            },
        ),
        migrations.AddIndex(
            model_name='activesignal',
            index=models.Index(fields=['status', 'opened_at'], name='activesignal_status_dt'),
        ),
    ]
