from django.db import migrations, models


class Migration(migrations.Migration):

	dependencies = [
		('portfolio', '0021_task_run_log'),
	]

	operations = [
		migrations.CreateModel(
			name='DataQADaily',
			fields=[
				('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
				('as_of', models.DateField(db_index=True, unique=True)),
				('price_metrics', models.JSONField(blank=True, default=dict)),
				('macro_metrics', models.JSONField(blank=True, default=dict)),
				('news_metrics', models.JSONField(blank=True, default=dict)),
				('anomaly_metrics', models.JSONField(blank=True, default=dict)),
				('created_at', models.DateTimeField(auto_now_add=True)),
			],
			options={
				'ordering': ['-as_of'],
			},
		),
	]