from django.db import migrations, models


class Migration(migrations.Migration):

	dependencies = [
		('portfolio', '0022_data_qa_daily'),
	]

	operations = [
		migrations.CreateModel(
			name='ModelCalibrationDaily',
			fields=[
				('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
				('as_of', models.DateField(db_index=True)),
				('model_name', models.CharField(choices=[('BLUECHIP', 'BLUECHIP'), ('PENNY', 'PENNY')], max_length=20)),
				('model_version', models.CharField(blank=True, default='', max_length=120)),
				('sandbox', models.CharField(blank=True, default='', max_length=20)),
				('bins', models.JSONField(blank=True, default=list)),
				('count', models.IntegerField(default=0)),
				('brier_score', models.FloatField(blank=True, null=True)),
				('created_at', models.DateTimeField(auto_now_add=True)),
			],
			options={
				'ordering': ['-as_of', 'model_name'],
				'unique_together': {('as_of', 'model_name', 'model_version', 'sandbox')},
			},
		),
		migrations.CreateModel(
			name='ModelDriftDaily',
			fields=[
				('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
				('as_of', models.DateField(db_index=True)),
				('model_name', models.CharField(choices=[('BLUECHIP', 'BLUECHIP'), ('PENNY', 'PENNY')], max_length=20)),
				('model_version', models.CharField(blank=True, default='', max_length=120)),
				('sandbox', models.CharField(blank=True, default='', max_length=20)),
				('psi', models.JSONField(blank=True, default=dict)),
				('feature_stats', models.JSONField(blank=True, default=dict)),
				('created_at', models.DateTimeField(auto_now_add=True)),
			],
			options={
				'ordering': ['-as_of', 'model_name'],
				'unique_together': {('as_of', 'model_name', 'model_version', 'sandbox')},
			},
		),
	]