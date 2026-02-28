from datetime import datetime, timedelta
from typing import Dict

import feedparser
from textblob import TextBlob


def fetch_news_sentiment(ticker: str, days: int = 7) -> Dict[str, float]:
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return {"news_sentiment": 0.0, "news_count": 0}

    rss_url = (
        f"https://news.google.com/rss/search?q={ticker}+stock+when:{days}d"
        "&hl=en-CA&gl=CA&ceid=CA:en"
    )
    feed = feedparser.parse(rss_url)
    entries = feed.entries or []
    if not entries:
        return {"news_sentiment": 0.0, "news_count": 0}

    sentiments = []
    neutral_phrases = ("release results", "conference call")
    cutoff = datetime.utcnow() - timedelta(days=days)
    for entry in entries:
        title = getattr(entry, "title", "")
        title_lower = str(title).lower()
        published = getattr(entry, "published_parsed", None)
        if published:
            published_dt = datetime(*published[:6])
            if published_dt < cutoff:
                continue
        analysis = TextBlob(title)
        polarity = analysis.sentiment.polarity
        if polarity == 0 and any(phrase in title_lower for phrase in neutral_phrases):
            continue
        sentiments.append(polarity)

    if not sentiments:
        return {"news_sentiment": 0.0, "news_count": len(entries)}

    avg_sentiment = sum(sentiments) / len(sentiments)
    return {"news_sentiment": round(avg_sentiment, 4), "news_count": len(entries)}
