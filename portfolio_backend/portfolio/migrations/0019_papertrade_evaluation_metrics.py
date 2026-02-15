from django.db import migrations, models


class Migration(migrations.Migration):

	dependencies = [
		('portfolio', '0018_sandboxwatchlist'),
	]

	operations = [
		migrations.AddField(
			model_name='papertrade',
			name='model_name',
			field=models.CharField(blank=True, default='BLUECHIP', max_length=20),
		),
		migrations.AddField(
			model_name='papertrade',
			name='model_version',
			field=models.CharField(blank=True, default='', max_length=120),
		),
		migrations.AddField(
			model_name='papertrade',
			name='outcome',
			field=models.CharField(blank=True, choices=[('WIN', 'WIN'), ('LOSS', 'LOSS')], max_length=10, null=True),
		),
		migrations.CreateModel(
			name='ModelEvaluationDaily',
			fields=[
				('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
				('as_of', models.DateField(db_index=True)),
				('model_name', models.CharField(choices=[('BLUECHIP', 'BLUECHIP'), ('PENNY', 'PENNY')], max_length=20)),
				('model_version', models.CharField(blank=True, default='', max_length=120)),
				('sandbox', models.CharField(blank=True, default='', max_length=20)),
				('trades', models.IntegerField(default=0)),
				('win_rate', models.FloatField(default=0)),
				('avg_pnl', models.FloatField(default=0)),
				('total_pnl', models.FloatField(default=0)),
				('max_drawdown', models.FloatField(default=0)),
				('brier_score', models.FloatField(blank=True, null=True)),
				('mean_predicted', models.FloatField(blank=True, null=True)),
				('mean_outcome', models.FloatField(blank=True, null=True)),
				('created_at', models.DateTimeField(auto_now_add=True)),
			],
			options={
				'ordering': ['-as_of', 'model_name'],
				'unique_together': {('as_of', 'model_name', 'model_version', 'sandbox')},
			},
		),
	]