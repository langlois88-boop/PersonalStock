from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0017_papertrade_sandbox'),
    ]

    operations = [
        migrations.CreateModel(
            name='SandboxWatchlist',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('sandbox', models.CharField(choices=[('WATCHLIST', 'WATCHLIST'), ('AI_BLUECHIP', 'AI_BLUECHIP'), ('AI_PENNY', 'AI_PENNY')], max_length=20, unique=True)),
                ('symbols', models.JSONField(blank=True, default=list)),
                ('source', models.CharField(blank=True, default='', max_length=100)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'ordering': ['sandbox'],
            },
        ),
    ]
