from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from portfolio.models import Stock, StockNews


class Command(BaseCommand):
    help = "Fetch recent news for each Stock using NewsAPI and store articles"

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=1,
            help="How many days back to search (default: 1)",
        )
        parser.add_argument(
            "--page-size",
            type=int,
            default=10,
            help="Max articles per symbol to fetch (default: 10)",
        )
        parser.add_argument(
            "--language",
            type=str,
            default="en",
            help="News language (default: en)",
        )

    def handle(self, *args, **options):
        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            raise CommandError(
                "Missing NEWSAPI_KEY env var. Example: export NEWSAPI_KEY='...'")

        days: int = options["days"]
        page_size: int = options["page_size"]
        language: str = options["language"]

        if days < 1:
            raise CommandError("--days must be >= 1")

        newsapi = NewsApiClient(api_key=api_key)
        analyzer = SentimentIntensityAnalyzer()

        from_dt = timezone.now() - timedelta(days=days)
        from_iso = from_dt.strftime("%Y-%m-%d")

        total_created = 0
        total_seen = 0

        for stock in Stock.objects.all().order_by("symbol"):
            query = stock.symbol
            try:
                result: dict[str, Any] = newsapi.get_everything(
                    q=query,
                    language=language,
                    sort_by="publishedAt",
                    from_param=from_iso,
                    page_size=page_size,
                )
            except Exception as exc:
                raise CommandError(f"NewsAPI error for {stock.symbol}: {exc}")

            articles = result.get("articles") or []
            self.stdout.write(f"{stock.symbol}: {len(articles)} articles")

            for a in articles:
                total_seen += 1
                url = a.get("url")
                if not url:
                    continue

                published_at = None
                published_raw = a.get("publishedAt")
                if published_raw:
                    try:
                        published_at = datetime.fromisoformat(
                            published_raw.replace("Z", "+00:00")
                        )
                    except ValueError:
                        published_at = None

                headline = (a.get("title") or "").strip()[:300] or url
                description = (a.get("description") or "").strip()
                text_for_sentiment = f"{headline}. {description}".strip()
                sentiment = analyzer.polarity_scores(text_for_sentiment).get("compound")

                obj, created = StockNews.objects.get_or_create(
                    url=url,
                    defaults={
                        "stock": stock,
                        "headline": headline,
                        "source": ((a.get("source") or {}).get("name") or "").strip()[:100],
                        "published_at": published_at,
                        "sentiment": sentiment,
                        "raw": a,
                    },
                )

                if created:
                    total_created += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Done. Seen {total_seen} articles, created {total_created} new rows."
            )
        )
