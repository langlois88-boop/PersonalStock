from __future__ import annotations

from datetime import timedelta

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from portfolio.models import NewsArticle


def _extract_signed_score(raw: object) -> float | None:
    if not isinstance(raw, dict):
        return None
    finbert = raw.get("finbert")
    if not isinstance(finbert, dict):
        return None
    score = finbert.get("signed_score")
    if score is None:
        return None
    try:
        return float(score)
    except Exception:
        return None


class Command(BaseCommand):
    help = "Compute average FinBERT sentiment for a symbol"

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbol",
            type=str,
            default="BTC-CAD",
            help="Symbol to scan (default: BTC-CAD)",
        )
        parser.add_argument(
            "--hours",
            type=int,
            default=24,
            help="Lookback window in hours (default: 24)",
        )

    def handle(self, *args, **options):
        symbol = (options.get("symbol") or "").strip().upper()
        if not symbol:
            raise CommandError("--symbol is required")
        hours = int(options.get("hours") or 0)
        if hours < 1:
            raise CommandError("--hours must be >= 1")

        cutoff = timezone.now() - timedelta(hours=hours)
        qs = NewsArticle.objects.filter(symbol__iexact=symbol, published_at__gte=cutoff)

        scores: list[float] = []
        for article in qs:
            score = _extract_signed_score(article.raw)
            if score is not None:
                scores.append(score)

        avg = (sum(scores) / len(scores)) if scores else 0.0
        self.stdout.write(
            f"{symbol} FinBERT sentiment (last {hours}h): {avg:.2f} | scored={len(scores)}"
        )
