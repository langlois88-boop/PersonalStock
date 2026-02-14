import os

from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'portfolio_backend.settings')

app = Celery('portfolio_backend')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
