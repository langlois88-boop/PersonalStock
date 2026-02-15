from django.db import migrations, models


class Migration(migrations.Migration):

	dependencies = [
		('portfolio', '0020_model_registry'),
	]

	operations = [
		migrations.CreateModel(
			name='TaskRunLog',
			fields=[
				('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
				('task_name', models.CharField(db_index=True, max_length=120)),
				('status', models.CharField(choices=[('SUCCESS', 'SUCCESS'), ('FAILED', 'FAILED')], db_index=True, max_length=20)),
				('started_at', models.DateTimeField(auto_now_add=True)),
				('finished_at', models.DateTimeField(blank=True, null=True)),
				('duration_ms', models.IntegerField(blank=True, null=True)),
				('error', models.TextField(blank=True, default='')),
				('payload', models.JSONField(blank=True, default=dict)),
			],
			options={
				'ordering': ['-started_at'],
			},
		),
	]