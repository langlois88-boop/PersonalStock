from django.db import migrations, models


class Migration(migrations.Migration):

	dependencies = [
		('portfolio', '0019_papertrade_evaluation_metrics'),
	]

	operations = [
		migrations.CreateModel(
			name='ModelRegistry',
			fields=[
				('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
				('model_name', models.CharField(choices=[('BLUECHIP', 'BLUECHIP'), ('PENNY', 'PENNY')], db_index=True, max_length=20)),
				('model_version', models.CharField(db_index=True, max_length=120)),
				('model_path', models.CharField(blank=True, default='', max_length=255)),
				('status', models.CharField(choices=[('ACTIVE', 'ACTIVE'), ('CANDIDATE', 'CANDIDATE'), ('ARCHIVED', 'ARCHIVED')], db_index=True, default='CANDIDATE', max_length=20)),
				('trained_at', models.DateTimeField(auto_now_add=True)),
				('backtest_win_rate', models.FloatField(blank=True, null=True)),
				('backtest_sharpe', models.FloatField(blank=True, null=True)),
				('paper_win_rate', models.FloatField(blank=True, null=True)),
				('paper_trades', models.IntegerField(blank=True, null=True)),
				('notes', models.JSONField(blank=True, default=dict)),
			],
			options={
				'ordering': ['-trained_at'],
				'unique_together': {('model_name', 'model_version')},
			},
		),
	]