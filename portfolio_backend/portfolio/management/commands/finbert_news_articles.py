from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Q
from django.utils import timezone
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from portfolio.models import NewsArticle


DEFAULT_MODEL = "ProsusAI/finbert"


def _get_model_name() -> str:
    return (os.getenv("FINBERT_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _clean_text(text: str) -> str:
    return " ".join((text or "").split())


def _build_text(article: NewsArticle) -> str:
    title = _clean_text(article.title or "")
    description = _clean_text(article.description or "")
    raw = article.raw if isinstance(article.raw, dict) else {}
    if not description:
        description = _clean_text(raw.get("summary") or "")
    if description:
        return f"{title}. {description}".strip()
    return title


def _score_to_signed(result: dict[str, Any]) -> float:
    label = (result.get("label") or "").lower()
    score = float(result.get("score") or 0.0)
    if label == "positive":
        return score
    if label == "negative":
        return -score
    return 0.0


class Command(BaseCommand):
    help = "Run FinBERT on NewsArticle rows and store scores in raw.finbert"

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbol",
            type=str,
            default="BTC-CAD",
            help="Symbol to score (default: BTC-CAD)",
        )
        parser.add_argument(
            "--days",
            type=int,
            default=30,
            help="How many days back to score (default: 30)",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=200,
            help="Max articles to score (default: 200)",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Re-score articles even if raw.finbert exists",
        )

    def handle(self, *args, **options):
        symbol = (options.get("symbol") or "").strip().upper()
        if not symbol:
            raise CommandError("--symbol is required")
        days = int(options.get("days") or 0)
        if days < 1:
            raise CommandError("--days must be >= 1")
        limit = int(options.get("limit") or 0)
        if limit < 1:
            raise CommandError("--limit must be >= 1")

        model_name = _get_model_name()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)

        cutoff = timezone.now() - timedelta(days=days)
        qs = NewsArticle.objects.filter(
            symbol__iexact=symbol
        ).filter(
            Q(published_at__gte=cutoff) | Q(published_at__isnull=True)
        ).order_by("-published_at", "-fetched_at")

        if not options.get("force"):
            qs = qs.exclude(raw__has_key="finbert")

        articles = list(qs[:limit])
        if not articles:
            self.stdout.write(self.style.WARNING("No articles to score."))
            return

        texts = [_build_text(article) for article in articles]
        results = nlp(texts, truncation=True)

        updated = 0
        for article, result in zip(articles, results):
            raw = article.raw if isinstance(article.raw, dict) else {}
            finbert_payload = {
                "model": model_name,
                "label": result.get("label"),
                "score": float(result.get("score") or 0.0),
                "signed_score": _score_to_signed(result),
            }
            raw["finbert"] = finbert_payload
            article.raw = raw
            article.save(update_fields=["raw"])
            updated += 1

        self.stdout.write(self.style.SUCCESS(f"Scored {updated} articles."))
