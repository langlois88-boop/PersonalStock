from __future__ import annotations

import os
from datetime import timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import feedparser
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from portfolio.models import NewsArticle


def _parse_watchlist(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip().upper() for item in value.split(',') if item.strip()]


def _build_query(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if not symbol:
        return ''
    is_crypto_pair = '-' in symbol and any(token in symbol for token in ['BTC', 'ETH', 'SOL', 'DOGE', 'XRP', 'ADA', 'LTC'])
    suffix = 'crypto' if is_crypto_pair else 'stock'
    return f"{symbol} {suffix}".strip()


class Command(BaseCommand):
    help = "Fetch news articles for a watchlist and store them in NewsArticle"

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=7,
            help="How many days back to search (default: 7)",
        )
        parser.add_argument(
            "--watchlist",
            type=str,
            default="",
            help="Comma-separated symbols to scan (overrides NEWS_SCRAPER_WATCHLIST)",
        )
        parser.add_argument(
            "--full",
            action="store_true",
            help="Scan all available feed entries (ignore --days cutoff)",
        )

    def handle(self, *args, **options):
        days = int(options["days"])
        if days < 1:
            raise CommandError("--days must be >= 1")

        watchlist = _parse_watchlist(options.get("watchlist"))
        if not watchlist:
            watchlist = _parse_watchlist(os.getenv("NEWS_SCRAPER_WATCHLIST"))

        if not watchlist:
            raise CommandError("No symbols provided. Set NEWS_SCRAPER_WATCHLIST or use --watchlist.")

        cutoff = timezone.now() - timedelta(days=days)
        total_created = 0
        total_seen = 0

        for symbol in watchlist:
            query = _build_query(symbol)
            if not query:
                continue
            url = (
                "https://news.google.com/rss/search"
                f"?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
            )
            feed = feedparser.parse(url)
            entries = feed.entries or []
            self.stdout.write(f"{symbol}: {len(entries)} articles")

            for entry in entries:
                total_seen += 1
                link = (entry.get("link") or "").strip()
                if not link:
                    continue
                link = link[:500]

                title = (entry.get("title") or "").strip()[:300] or link
                description = (entry.get("summary") or "").strip()

                published_at = None
                if entry.get("published"):
                    try:
                        published_at = parsedate_to_datetime(entry["published"])
                    except Exception:
                        published_at = None

                if not options.get("full") and published_at and published_at < cutoff:
                    continue

                _, created = NewsArticle.objects.get_or_create(
                    url=link,
                    defaults={
                        "symbol": symbol,
                        "title": title,
                        "description": description,
                        "source": "Google News",
                        "published_at": published_at,
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
