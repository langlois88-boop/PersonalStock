from __future__ import annotations

import json
import os
import re
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import requests
import finnhub
import joblib
from django.core.cache import cache
from django.utils import timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from portfolio import market_data as yf
from portfolio.alpaca_data import get_intraday_context
from portfolio.ml_engine.collectors.news_rss import fetch_news_sentiment
from portfolio.ml_engine import train_penny_model, train_stable_model
from portfolio.models import Stock
from portfolio.tasks import (
    _fetch_yfinance_screeners,
    _google_news_titles,
    bluechip_dip_scanner,
    penny_opportunity_scanner,
)


@dataclass
class ValidationCandidate:
    symbol: str
    price: float | None
    change_pct: float | None
    ml_signal: str
    ml_confidence: float | None
    rsi: float | None
    volume_change: float | None


class ValidationService:
    def __init__(self) -> None:
        base_url = (
            os.getenv('VALIDATION_LLM_BASE_URL')
            or os.getenv('DANAS_CHAT_BASE_URL')
            or os.getenv('OLLAMA_CHAT_BASE_URL')
            or os.getenv('OLLAMA_BASE_URL')
            or ''
        ).strip().rstrip('/')
        if base_url and '/v1' not in base_url:
            base_url = f"{base_url}/v1"
        self.base_url = base_url
        self.model = (
            os.getenv('VALIDATION_LLM_MODEL')
            or os.getenv('OLLAMA_MODEL')
            or 'deepseek-r1:8b'
        ).strip()
        self.timeout = int(os.getenv('VALIDATION_LLM_TIMEOUT', os.getenv('OLLAMA_TIMEOUT', '90')))
        self.min_score = float(os.getenv('VALIDATION_MIN_SCORE', '7'))
        self.news_days = int(os.getenv('VALIDATION_NEWS_DAYS', '3'))
        self.news_limit = int(os.getenv('VALIDATION_NEWS_LIMIT', '3'))
        self.max_candidates = int(os.getenv('VALIDATION_MAX_CANDIDATES', '5'))
        self.lag_seconds = float(os.getenv('VALIDATION_LAG_SECONDS', '-0.58'))
        self.screeners = [
            s.strip()
            for s in os.getenv('VALIDATION_SCREENERS', 'day_losers,most_actives').split(',')
            if s.strip()
        ]
        self.cache_key = 'danas_consensus_results'
        self.cache_ttl = int(os.getenv('VALIDATION_CACHE_TTL', str(60 * 30)))
        self.volatility_max = float(os.getenv('VALIDATION_VOLATILITY_MAX', '0.035'))
        self.ml_json_path = (
            os.getenv('VALIDATION_ML_JSON_PATH')
            or os.getenv('BIGGEST_LOSER_JSON')
            or ''
        ).strip()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.penny_model_path = Path(os.getenv('PENNY_MODEL_PATH', ''))
        if not self.penny_model_path:
            self.penny_model_path = Path(__file__).resolve().parents[1] / 'ml_engine' / 'scout_brain_v1.pkl'
        self.stable_model_path = Path(os.getenv('STABLE_MODEL_PATH', ''))
        if not self.stable_model_path:
            self.stable_model_path = Path(__file__).resolve().parents[1] / 'ml_engine' / 'stable_brain_v1.pkl'
        self._penny_model_cache: dict[str, Any] | None = None
        self._stable_model_cache: Any | None = None

    def _is_canadian(self, symbol: str) -> bool:
        return symbol.upper().endswith('.TO') or symbol.upper().endswith('.V')

    def _parse_float(self, value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _fetch_finnhub_titles(self, symbol: str) -> list[str]:
        api_key = os.getenv('FINNHUB_API_KEY') or os.getenv('FINNHUB_KEY')
        if not api_key:
            return []
        try:
            client = finnhub.Client(api_key=api_key)
            to_dt = timezone.now().date()
            from_dt = to_dt - timedelta(days=self.news_days)
            items = client.company_news(symbol, _from=str(from_dt), to=str(to_dt)) or []
        except Exception:
            return []
        headlines = []
        for item in items:
            headline = (item.get('headline') or '').strip()
            if headline:
                headlines.append(headline)
            if len(headlines) >= self.news_limit:
                break
        return headlines

    def _news_headlines(self, symbol: str) -> list[str]:
        headlines = self._fetch_finnhub_titles(symbol)
        if headlines:
            return headlines
        try:
            headlines = _google_news_titles(symbol, days=self.news_days, limit=self.news_limit)
        except Exception:
            headlines = []
        return headlines or []

    def _build_prompt(
        self,
        candidate: ValidationCandidate,
        news_headlines: list[str],
        patterns: list[str] | None = None,
        news_sentiment: float | None = None,
        volatility: float | None = None,
    ) -> str:
        lines = [
            "CONTEXTE DE TRADING - ANALYSE QUANT & SÉMANTIQUE",
            "--------------------------------------------------",
            f"CIBLE DETECTÉE : {candidate.symbol}",
            f"PRIX ACTUEL : {candidate.price if candidate.price is not None else 'N/A'}$ ({candidate.change_pct if candidate.change_pct is not None else 'N/A'}%)",
            f"SIGNAL ML (Scikit-Learn) : {candidate.ml_signal} (Confiance: {candidate.ml_confidence if candidate.ml_confidence is not None else 'N/A'}%)",
            f"INDICATEURS : RSI: {candidate.rsi if candidate.rsi is not None else 'N/A'} | Volume: {candidate.volume_change if candidate.volume_change is not None else 'N/A'}% | Lag Sync: {self.lag_seconds}s",
            f"PATTERNS: {', '.join(patterns or []) if patterns else 'N/A'}",
            f"NEWS_SENTIMENT: {news_sentiment if news_sentiment is not None else 'N/A'}",
            f"VOLATILITÉ_INTRA: {volatility if volatility is not None else 'N/A'}",
            "",
            "NEWS RÉCENTES (Finnhub/Yahoo) :",
            *([f"- {h}" for h in news_headlines] if news_headlines else ["- Aucune news récente disponible"]),
            "",
            "MISSION :",
            "Tu es Danas, l'expert de validation. Ton modèle Scikit-Learn veut acheter ce \"Dip\".",
            "1. Analyse si la chute est purement technique (opportunité) ou fondamentale (danger).",
            "2. Prends en compte le Lag de -0.58s : est-ce que le mouvement est trop volatil pour notre exécution ?",
            "3. Donne un score de validation FINAL de 1 à 10.",
            "",
            "RÉPONSE COURTE EN FRANÇAIS :",
            "- <think> Ta réflexion sur la corrélation News/Technique </think>",
            "- DIAGNOSTIC : (Opportunité ou Piège ?)",
            "- ACTION : (Achat immédiat / Attendre confirmation / Ignorer)",
            "- VALIDATION : [Score]/10",
        ]
        return "\n".join(lines)

    def _chat_completion(self, prompt: str) -> str | None:
        if not self.base_url:
            return None
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': (
                        "Tu es Danas. Nous sommes le "
                        f"{timezone.now().strftime('%d/%m/%Y %H:%M')}. "
                        "Réponds uniquement en français."
                    ),
                },
                {'role': 'user', 'content': prompt},
            ],
            'stream': False,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return ((data.get('choices') or [{}])[0].get('message') or {}).get('content')

    def _parse_validation_text(self, text: str | None) -> dict[str, Any]:
        if not text:
            return {'raw': '', 'diagnostic': None, 'action': None, 'validation_score': None}
        diagnostic = None
        action = None
        score = None
        for line in text.splitlines():
            if 'DIAGNOSTIC' in line.upper():
                diagnostic = line.split(':', 1)[-1].strip()
            if 'ACTION' in line.upper():
                action = line.split(':', 1)[-1].strip()
        match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', text)
        if match:
            try:
                score = float(match.group(1))
            except Exception:
                score = None
        return {
            'raw': text.strip(),
            'diagnostic': diagnostic,
            'action': action,
            'validation_score': score,
        }

    def _pattern_bias(self, patterns: list[str]) -> str:
        bullish = {'bullish', 'hammer', 'engulfing', 'morning', 'breakout', 'reversal'}
        bearish = {'bearish', 'shooting', 'dark cloud', 'evening', 'breakdown'}
        for pattern in patterns:
            name = str(pattern).lower()
            if any(key in name for key in bullish):
                return 'bullish'
            if any(key in name for key in bearish):
                return 'bearish'
        return 'neutral'

    def _news_sentiment(self, headlines: list[str]) -> float:
        if not headlines:
            return 0.0
        scores = [self.sentiment_analyzer.polarity_scores(title).get('compound', 0.0) for title in headlines]
        return float(sum(scores) / len(scores)) if scores else 0.0

    def _load_penny_model(self) -> dict[str, Any] | None:
        if self._penny_model_cache is not None:
            return self._penny_model_cache
        try:
            payload = joblib.load(self.penny_model_path)
            if isinstance(payload, dict) and payload.get('model') and payload.get('features'):
                self._penny_model_cache = payload
                return payload
        except Exception:
            return None
        return None

    def _load_stable_model(self) -> Any | None:
        if self._stable_model_cache is not None:
            return self._stable_model_cache
        try:
            model = joblib.load(self.stable_model_path)
            self._stable_model_cache = model
            return model
        except Exception:
            return None

    def _penny_prediction(self, symbol: str) -> float | None:
        payload = self._load_penny_model()
        if not payload:
            return None
        try:
            data = yf.Ticker(symbol).history(period='60d', interval='1d', timeout=10)
        except Exception:
            return None
        if data is None or data.empty:
            return None
        try:
            features = train_penny_model._build_features(data).dropna()
            if features.empty:
                return None
            last = features.tail(1)
            feature_list = payload.get('features') or []
            for col in feature_list:
                if col not in last.columns:
                    last[col] = 0.0
            X = last[feature_list].fillna(0).values
            model = payload.get('model')
            try:
                proba = float(model.predict_proba(X)[0][1])
                return proba
            except Exception:
                pred = float(model.predict(X)[0]) if hasattr(model, 'predict') else 0.0
                return pred
        except Exception:
            return None
        return None

    def _stable_prediction(self, symbol: str) -> float | None:
        model = self._load_stable_model()
        if model is None:
            return None
        try:
            data = yf.Ticker(symbol).history(period='2y', interval='1d', timeout=10)
        except Exception:
            return None
        if data is None or data.empty or 'Close' not in data:
            return None
        try:
            spy = yf.Ticker('SPY').history(period='2y', interval='1d', timeout=10)
        except Exception:
            spy = None
        if spy is None or spy.empty or 'Close' not in spy:
            return None
        close = data['Close']
        volume = data['Volume'] if 'Volume' in data else close * 0
        dividend_yield = float(Stock.objects.filter(symbol__iexact=symbol).values_list('dividend_yield', flat=True).first() or 0)
        news_payload = fetch_news_sentiment(symbol)
        sentiment_score = float(news_payload.get('news_sentiment') or 0.0)
        features = train_stable_model._build_features(
            close,
            volume,
            spy['Close'],
            dividend_yield,
            sector_close=None,
            sentiment_score=sentiment_score,
        )
        if not features:
            return None
        try:
            return float(model.predict([features])[0])
        except Exception:
            return None

    def _consensus_prompt(
        self,
        symbol: str,
        penny_pred: float | None,
        stable_pred: float | None,
        rss_sentiment: float,
        news_headlines: list[str],
    ) -> str:
        lines = [
            "CONSENSUS MULTI-MODÈLES",
            "------------------------",
            f"SYMBOL: {symbol}",
            f"PENNY_MODEL_PROBA: {penny_pred if penny_pred is not None else 'N/A'}",
            f"STABLE_MODEL_RET20J: {stable_pred if stable_pred is not None else 'N/A'}",
            f"RSS_SENTIMENT: {rss_sentiment}",
            "NEWS:",
            *([f"- {h}" for h in news_headlines] if news_headlines else ["- Aucune news disponible"]),
            "",
            "Question: Le trade est-il rationnel ou un piège ?",
            "Réponds par VALIDE ou REJETE avec une raison d'une ligne.",
        ]
        return "\n".join(lines)

    def get_final_consensus(
        self,
        symbol: str,
        news_headlines: list[str],
    ) -> dict[str, Any]:
        penny_pred = self._penny_prediction(symbol)
        stable_pred = self._stable_prediction(symbol)
        rss_payload = fetch_news_sentiment(symbol, days=self.news_days)
        rss_sentiment = float(rss_payload.get('news_sentiment') or 0.0)
        prompt = self._consensus_prompt(symbol, penny_pred, stable_pred, rss_sentiment, news_headlines)
        text = None
        try:
            text = self._chat_completion(prompt)
        except Exception:
            text = None
        verdict = 'REJETE'
        if text and 'VALIDE' in text.upper():
            verdict = 'VALIDE'
        return {
            'verdict': verdict,
            'reason': (text or '').strip(),
            'penny_pred': penny_pred,
            'stable_pred': stable_pred,
            'rss_sentiment': rss_sentiment,
            'rss_count': int(rss_payload.get('news_count') or 0),
        }

    def _candidate_from_scanner(self, row: dict[str, Any]) -> ValidationCandidate:
        symbol = (row.get('symbol') or '').strip().upper()
        price = self._parse_float(row.get('price'))
        score = self._parse_float(row.get('score'))
        rsi = self._parse_float(row.get('rsi'))
        return ValidationCandidate(
            symbol=symbol,
            price=price,
            change_pct=None,
            ml_signal='BUY',
            ml_confidence=None if score is None else round(score * 100, 2),
            rsi=rsi,
            volume_change=None,
        )

    def _load_ml_json_candidates(self) -> list[ValidationCandidate]:
        if not self.ml_json_path:
            return []
        try:
            with open(self.ml_json_path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
        except Exception:
            return []
        if isinstance(payload, dict):
            payload = payload.get('results') or payload.get('candidates') or payload.get('signals') or []
        if not isinstance(payload, list):
            return []
        candidates: list[ValidationCandidate] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            symbol = (item.get('symbol') or item.get('ticker') or '').strip().upper()
            if not symbol:
                continue
            price = self._parse_float(item.get('price') or item.get('current_price'))
            change_pct = self._parse_float(item.get('change_pct') or item.get('dip') or item.get('percent_change'))
            rsi = self._parse_float(item.get('rsi') or item.get('rsi14'))
            confidence = self._parse_float(item.get('confidence') or item.get('ml_confidence'))
            ml_signal = str(item.get('ml_signal') or item.get('signal') or 'BUY').upper()
            candidates.append(
                ValidationCandidate(
                    symbol=symbol,
                    price=price,
                    change_pct=change_pct,
                    ml_signal=ml_signal,
                    ml_confidence=confidence,
                    rsi=rsi,
                    volume_change=None,
                )
            )
        return candidates

    def _scan_candidates(self) -> list[ValidationCandidate]:
        ml_candidates = self._load_ml_json_candidates()
        cached_blue = cache.get('bluechip_dip_candidates') or []
        cached_penny = cache.get('penny_opportunity_candidates') or []
        blue = cached_blue if cached_blue else (bluechip_dip_scanner().get('results') or [])
        penny = cached_penny if cached_penny else (penny_opportunity_scanner().get('results') or [])

        screeners = []
        if self.screeners:
            try:
                screeners = _fetch_yfinance_screeners(self.screeners, count=25)
            except Exception:
                screeners = []

        candidates = list(ml_candidates)
        candidates.extend([self._candidate_from_scanner(row) for row in (blue + penny) if row.get('symbol')])
        for item in screeners:
            symbol = (item.get('symbol') or '').strip().upper()
            if not symbol:
                continue
            price = self._parse_float(item.get('regularMarketPrice') or item.get('price') or item.get('lastPrice'))
            change_pct = self._parse_float(item.get('regularMarketChangePercent') or item.get('percentChange'))
            volume = self._parse_float(item.get('regularMarketVolume') or item.get('volume'))
            candidates.append(
                ValidationCandidate(
                    symbol=symbol,
                    price=price,
                    change_pct=change_pct,
                    ml_signal='WATCH',
                    ml_confidence=None,
                    rsi=None,
                    volume_change=volume,
                )
            )

        unique: dict[str, ValidationCandidate] = {}
        for item in candidates:
            if item.symbol and item.symbol not in unique:
                unique[item.symbol] = item
        sorted_candidates = sorted(
            unique.values(),
            key=lambda c: (self._is_canadian(c.symbol), c.ml_confidence or 0),
            reverse=True,
        )
        return list(sorted_candidates)[: self.max_candidates]

    def run_consensus(self, refresh: bool = False) -> dict[str, Any]:
        if not refresh:
            cached = cache.get(self.cache_key)
            if cached:
                return cached

        results: list[dict[str, Any]] = []
        for candidate in self._scan_candidates():
            if not candidate.symbol:
                continue
            news = self._news_headlines(candidate.symbol)
            news_sentiment = self._news_sentiment(news)
            patterns: list[str] = []
            volatility = None
            try:
                ctx = get_intraday_context(candidate.symbol)
                if ctx:
                    patterns = list(ctx.get('patterns') or [])
                    volatility = float(ctx.get('volatility')) if ctx.get('volatility') is not None else None
            except Exception:
                patterns = []
                volatility = None
            consensus = self.get_final_consensus(candidate.symbol, news)
            prompt = self._build_prompt(
                candidate,
                news,
                patterns=patterns,
                news_sentiment=news_sentiment,
                volatility=volatility,
            )
            text = None
            try:
                text = self._chat_completion(prompt)
            except Exception:
                text = None

            parsed = self._parse_validation_text(text)
            if parsed.get('validation_score') is None:
                parsed['validation_score'] = 0.0
            pattern_bias = self._pattern_bias(patterns)
            if pattern_bias == 'bullish' and news_sentiment < -0.1:
                parsed['validation_score'] = float(parsed.get('validation_score') or 0) * 0.5
            if pattern_bias == 'bearish' and news_sentiment > 0.1:
                parsed['validation_score'] = float(parsed.get('validation_score') or 0) * 0.6
            if volatility is not None and volatility > self.volatility_max and self.lag_seconds < 0:
                parsed['validation_score'] = float(parsed.get('validation_score') or 0) * 0.7
            if consensus.get('verdict') == 'REJETE':
                parsed['validation_score'] = min(float(parsed.get('validation_score') or 0), 3.0)
            results.append({
                'symbol': candidate.symbol,
                'price': candidate.price,
                'change_pct': candidate.change_pct,
                'ml_signal': candidate.ml_signal,
                'ml_confidence': candidate.ml_confidence,
                'rsi': candidate.rsi,
                'patterns': patterns,
                'news_sentiment': round(news_sentiment, 4),
                'volatility': volatility,
                'consensus': consensus,
                'news': news,
                'diagnostic': parsed.get('diagnostic'),
                'action': parsed.get('action'),
                'validation_score': parsed.get('validation_score'),
                'raw': parsed.get('raw'),
            })

        filtered = [r for r in results if float(r.get('validation_score') or 0) >= self.min_score]
        payload = {
            'as_of': timezone.now().isoformat(),
            'count': len(filtered),
            'results': filtered,
        }
        cache.set(self.cache_key, payload, timeout=self.cache_ttl)
        return payload
