from __future__ import annotations

from django.core.management.base import BaseCommand

from portfolio.tasks import generate_penny_signals


class Command(BaseCommand):
    help = "Generate penny stock signals from Reddit sentiment + price patterns"

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=7,
            help="Lookback window in days (default: 7)",
        )

    def handle(self, *args, **options):
        result = generate_penny_signals(days=options["days"])
        self.stdout.write(
            self.style.SUCCESS(
                f"Created {result['created']} signals from {result['seen']} candidates"
            )
        )
