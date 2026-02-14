#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def main():
    """Run administrative tasks."""
    if load_dotenv:
        base_dir = Path(__file__).resolve().parent
        load_dotenv(base_dir / '.env')
        load_dotenv(base_dir / 'portfolio_backend' / '.env')

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'portfolio_backend.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
