from __future__ import annotations

from datetime import timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import feedparser
from django.core.management.base import BaseCommand
from django.utils import timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from portfolio.models import Stock, StockNews


class Command(BaseCommand):
    help = "Fetch recent news for each Stock using Google News RSS and store articles"

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=1,
            help="How many days back to search (default: 1)",
        )

    def handle(self, *args, **options):
        days = int(options["days"])
        if days < 1:
            raise ValueError("--days must be >= 1")

        analyzer = SentimentIntensityAnalyzer()
        cutoff = timezone.now() - timedelta(days=days)

        total_created = 0
        total_seen = 0

        for stock in Stock.objects.all().order_by("symbol"):
            query = quote_plus(f"{stock.symbol} stock")
            url = (
                "https://news.google.com/rss/search"
                f"?q={query}&hl=en-US&gl=US&ceid=US:en"
            )
            feed = feedparser.parse(url)
            self.stdout.write(f"{stock.symbol}: {len(feed.entries)} articles")

            for entry in feed.entries:
                total_seen += 1
                link = entry.get("link")
                if not link:
                    continue

                headline = (entry.get("title") or "").strip()[:300] or link
                summary = (entry.get("summary") or "").strip()
                text_for_sentiment = f"{headline}. {summary}".strip()
                sentiment = analyzer.polarity_scores(text_for_sentiment).get("compound")

                published_at = None
                if entry.get("published"):
                    try:
                        published_at = parsedate_to_datetime(entry["published"])
                    except Exception:
                        published_at = None

                if published_at and published_at < cutoff:
                    continue

                _, created = StockNews.objects.get_or_create(
                    url=link,
                    defaults={
                        "stock": stock,
                        "headline": headline,
                        "source": "Google News",
                        "published_at": published_at,
                        "sentiment": sentiment,
                        "raw": entry,
                    },
                )

                if created:
                    total_created += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Done. Seen {total_seen} articles, created {total_created} new rows."
            )
        )
