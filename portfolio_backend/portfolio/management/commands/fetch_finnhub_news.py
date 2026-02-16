from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone as dt_timezone

import finnhub
from finnhub.exceptions import FinnhubAPIException
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from portfolio.models import Stock, StockNews


class Command(BaseCommand):
    help = "Fetch recent news for each Stock using Finnhub and store articles"

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=1,
            help="How many days back to search (default: 1)",
        )

    def handle(self, *args, **options):
        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            self.stdout.write(
                self.style.WARNING(
                    "Skipping Finnhub fetch: FINNHUB_API_KEY env var is missing."
                )
            )
            return

        days = int(options["days"])
        if days < 1:
            raise CommandError("--days must be >= 1")

        client = finnhub.Client(api_key=api_key)
        analyzer = SentimentIntensityAnalyzer()

        to_dt = timezone.now().date()
        from_dt = to_dt - timedelta(days=days)

        total_created = 0
        total_seen = 0

        for stock in Stock.objects.all().order_by("symbol"):
            try:
                news = client.company_news(stock.symbol, _from=str(from_dt), to=str(to_dt))
            except FinnhubAPIException as exc:
                if getattr(exc, "status_code", None) == 401:
                    self.stderr.write(
                        self.style.ERROR(
                            "Finnhub API key was rejected (401). Update FINNHUB_API_KEY and retry."
                        )
                    )
                    break
                self.stderr.write(
                    self.style.WARNING(f"Finnhub error for {stock.symbol}: {exc}")
                )
                continue

            self.stdout.write(f"{stock.symbol}: {len(news)} articles")

            for item in news:
                total_seen += 1
                url = item.get("url")
                if not url:
                    continue

                headline = (item.get("headline") or "").strip()[:300] or url
                summary = (item.get("summary") or "").strip()
                text_for_sentiment = f"{headline}. {summary}".strip()
                sentiment = analyzer.polarity_scores(text_for_sentiment).get("compound")

                published_at = None
                if item.get("datetime"):
                    published_at = datetime.fromtimestamp(item["datetime"], tz=dt_timezone.utc)

                _, created = StockNews.objects.get_or_create(
                    url=url,
                    defaults={
                        "stock": stock,
                        "headline": headline,
                        "source": "Finnhub",
                        "published_at": published_at,
                        "sentiment": sentiment,
                        "raw": item,
                    },
                )

                if created:
                    total_created += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Done. Seen {total_seen} articles, created {total_created} new rows."
            )
        )
