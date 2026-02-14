"""
ASGI config for portfolio_backend project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

import os
from pathlib import Path

try:
	from dotenv import load_dotenv
except ImportError:  # pragma: no cover
	load_dotenv = None

from django.core.asgi import get_asgi_application

if load_dotenv:
	base_dir = Path(__file__).resolve().parent.parent
	load_dotenv(base_dir / '.env')
	load_dotenv(base_dir / 'portfolio_backend' / '.env')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'portfolio_backend.settings')

application = get_asgi_application()
