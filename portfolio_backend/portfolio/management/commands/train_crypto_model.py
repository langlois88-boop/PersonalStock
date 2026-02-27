from __future__ import annotations

from django.core.management.base import BaseCommand

from ...ml_engine.crypto_training import (
    DEFAULT_SYMBOLS,
    save_crypto_model,
    train_crypto_model,
)


class Command(BaseCommand):
    help = "Train crypto model using 15m data."

    def add_arguments(self, parser):
        parser.add_argument('--symbols', type=str, default='', help='Comma-separated crypto symbols')
        parser.add_argument('--days', type=int, default=60)

    def handle(self, *args, **options):
        raw = options.get('symbols') or ''
        symbols = [s.strip().upper() for s in raw.split(',') if s.strip()] or DEFAULT_SYMBOLS
        days = int(options.get('days') or 60)
        payload = train_crypto_model(symbols, days=days)
        path = save_crypto_model(payload)
        self.stdout.write(self.style.SUCCESS(f'Crypto model saved to {path}'))
