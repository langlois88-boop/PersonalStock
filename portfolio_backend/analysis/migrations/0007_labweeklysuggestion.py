# Généré manuellement (2026-08-14, Partie 3 -- résumé hebdomadaire
# automatique, cf. analysis.tasks.generate_lab_weekly_suggestions).

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0006_labposition_close_fields'),
    ]

    operations = [
        migrations.CreateModel(
            name='LabWeeklySuggestion',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('week_start', models.DateField(db_index=True)),
                ('generated_at', models.DateTimeField(auto_now_add=True)),
                ('positions_closed_analyzed', models.IntegerField(default=0)),
                ('has_pattern', models.BooleanField(default=False)),
                ('summary', models.TextField(blank=True, default='')),
            ],
            options={
                'ordering': ['-generated_at'],
            },
        ),
    ]
