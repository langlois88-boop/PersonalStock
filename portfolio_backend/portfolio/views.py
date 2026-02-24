import csv
import io
import json
import os
import math
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Any
import unicodedata
from types import SimpleNamespace
from urllib.parse import quote
from urllib.request import Request, urlopen
from django.db import models
from django.conf import settings
from django.db.utils import OperationalError
from django.db.models import Prefetch
from django.utils import timezone
import finnhub
from . import market_data as yf
import numpy as np
import pandas as pd
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rest_framework import viewsets
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.views import APIView
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.exceptions import ValidationError
from django.contrib.auth import get_user_model
from django.core.cache import cache

from .models import (
	Account,
	AccountTransaction,
	AlertEvent,
	Dividend,
	DripSnapshot,
	Portfolio,
	PortfolioDigest,
	PortfolioHolding,
	Stock,
	StockNews,
	Transaction,
	PriceHistory,
	Prediction,
	PaperTrade,
	UserPreference,
	MacroIndicator,
	PennyStockUniverse,
	PennyStockSnapshot,
	PennySignal,
	NewsArticle,
	DividendHistory,
	SandboxWatchlist,
	ModelEvaluationDaily,
	ModelRegistry,
	TaskRunLog,
	DataQADaily,
	ModelCalibrationDaily,
	ModelDriftDaily,
)
from .serializers import (
	AccountSerializer,
	AccountTransactionSerializer,
	AlertEventSerializer,
	DividendSerializer,
	DripSnapshotSerializer,
	PortfolioSerializer,
	PortfolioDigestSerializer,
	PortfolioHoldingSerializer,
	PennySignalSerializer,
	PennyStockScoutSerializer,
	PennyStockSnapshotSerializer,
	PriceHistorySerializer,
	PredictionSerializer,
	PaperTradeSerializer,
	MacroIndicatorSerializer,
	StockSerializer,
	StockNewsSerializer,
	TransactionSerializer,
	UserPreferenceSerializer,
	SandboxWatchlistSerializer,
)
from .ai_module import run_predictions
from .tasks import (
	_fetch_yahoo_screener,
	_fetch_yfinance_screeners,
	_google_news_titles,
	_finbert_score_from_titles,
	_intraday_pct_change,
	CORRELATION_MAP,
	_add_candlestick_features,
	_yahoo_fundamentals,
	analyze_ticker_for_ui,
)
from .ai_scout import build_scout_summary
from .alpaca_data import get_intraday_bars, get_intraday_context
from .patterns import build_pattern_annotations, enrich_bars_with_patterns
from .ml_engine.engine.data_fusion import DataFusionEngine
from .ml_engine.collectors.news_rss import fetch_news_sentiment

import feedparser
from textblob import TextBlob
from .ml_engine.backtester import (
	AIBacktester,
	BacktestResult,
	load_or_train_model,
	FEATURE_COLUMNS,
	get_model_path,
	apply_feature_weighting_to_signal,
)


def generate_expert_advice(metrics: dict[str, float | None]) -> list[str]:
	advice = []
	sharpe = metrics.get('sharpe')
	win_rate = metrics.get('win_rate')
	max_drawdown = metrics.get('max_drawdown')
	if sharpe is not None and sharpe < 1.0:
		advice.append('⚠️ Risque trop élevé pour le rendement attendu.')
	if win_rate is not None and win_rate < 50:
		advice.append('❌ Stratégie de type pile ou face. Améliorez le filtrage.')
	if max_drawdown is not None and max_drawdown > 10:
		advice.append('🚨 Danger de perte en capital importante. Resserrez les Stops.')
	return advice


def _portfolio_return_snapshot() -> dict[str, float | None]:
	portfolio = Portfolio.objects.first()
	if not portfolio:
		return {'portfolio_total_return_pct': None, 'portfolio_cost': None, 'portfolio_value': None}
	holdings = PortfolioHolding.objects.select_related('stock').filter(portfolio=portfolio)
	if not holdings.exists():
		return {'portfolio_total_return_pct': None, 'portfolio_cost': None, 'portfolio_value': None}
	cost_map: dict[str, dict[str, float]] = {}
	for tx in Transaction.objects.filter(portfolio=portfolio):
		symbol_key = (tx.stock.symbol or '').strip().upper() or str(tx.stock_id)
		entry = cost_map.setdefault(symbol_key, {'shares': 0.0, 'buy_qty': 0.0, 'buy_cost': 0.0})
		qty = float(tx.shares or 0)
		sign = 1 if tx.transaction_type == 'BUY' else -1
		entry['shares'] += qty * sign
		if tx.transaction_type == 'BUY':
			entry['buy_qty'] += qty
			entry['buy_cost'] += qty * float(tx.price_per_share or 0)
	current_value = 0.0
	cost_value = 0.0
	for holding in holdings:
		stock = holding.stock
		price = float(stock.latest_price or 0)
		if price <= 0:
			last = PriceHistory.objects.filter(stock=stock).order_by('-date').first()
			price = float(last.close_price) if last else 0.0
		symbol_key = (stock.symbol or '').strip().upper() or str(stock.id)
		cost_data = cost_map.get(symbol_key, {})
		buy_qty = float(cost_data.get('buy_qty') or 0)
		buy_cost = float(cost_data.get('buy_cost') or 0)
		avg_cost = (buy_cost / buy_qty) if buy_qty else 0.0
		shares = float(holding.shares or 0)
		current_value += price * shares
		cost_value += avg_cost * shares
	return_pct = ((current_value - cost_value) / cost_value * 100) if cost_value else None
	return {
		'portfolio_total_return_pct': round(float(return_pct), 2) if return_pct is not None else None,
		'portfolio_cost': round(cost_value, 2),
		'portfolio_value': round(current_value, 2),
	}


def _news_sentiment_24h(symbol: str) -> tuple[float, int, int]:
	symbol = (symbol or '').strip().upper()
	if not symbol:
		return 0.0, 0, 0
	stock = Stock.objects.filter(symbol__iexact=symbol).first()
	now = timezone.now()
	start_24 = now - timedelta(hours=24)
	start_48 = now - timedelta(hours=48)
	qs = StockNews.objects.filter(stock=stock) if stock else StockNews.objects.filter(stock__symbol__iexact=symbol)
	current_qs = qs.filter(published_at__gte=start_24)
	prev_qs = qs.filter(published_at__gte=start_48, published_at__lt=start_24)
	current_count = current_qs.count()
	prev_count = prev_qs.count()
	if current_count > 0:
		avg_sentiment = current_qs.aggregate(avg=models.Avg('sentiment')).get('avg') or 0.0
		return float(avg_sentiment), current_count, prev_count
	result = fetch_news_sentiment(symbol, days=1)
	return float(result.get('news_sentiment') or 0.0), int(result.get('news_count') or 0), prev_count


def _rss_sentiment_window(ticker: str, hours: int, offset_hours: int = 0) -> float:
	if not ticker:
		return 0.0
	now = datetime.utcnow() - timedelta(hours=offset_hours)
	start = now - timedelta(hours=hours)
	rss_url = (
		f"https://news.google.com/rss/search?q={ticker}+stock+when:1d"
		"&hl=en-CA&gl=CA&ceid=CA:en"
	)
	feed = feedparser.parse(rss_url)
	entries = feed.entries or []
	scores = []
	for entry in entries:
		published = getattr(entry, 'published_parsed', None)
		if not published:
			continue
		published_dt = datetime(*published[:6])
		if published_dt < start or published_dt > now:
			continue
		title = getattr(entry, 'title', '')
		if not title:
			continue
		scores.append(TextBlob(title).sentiment.polarity)
	if not scores:
		return 0.0
	return float(sum(scores) / len(scores))


def _fusion_close_series(frame: pd.DataFrame) -> pd.Series | None:
	if frame is None or frame.empty:
		return None
	if 'Close' in frame.columns:
		return frame['Close']
	if 'close' in frame.columns:
		return frame['close']
	return None


def _compute_rsi_from_series(series: pd.Series, period: int = 14) -> float | None:
	if series is None or series.empty or len(series) < period:
		return None
	delta = series.diff().fillna(0)
	gain = delta.clip(lower=0).rolling(period).mean()
	loss = (-delta.clip(upper=0)).rolling(period).mean()
	last_gain = float(gain.iloc[-1]) if pd.notna(gain.iloc[-1]) else None
	last_loss = float(loss.iloc[-1]) if pd.notna(loss.iloc[-1]) else None
	if last_gain is None or last_loss is None:
		return None
	if last_loss == 0:
		return 100.0 if last_gain > 0 else 0.0
	rs = last_gain / last_loss
	rsi = 100 - (100 / (1 + rs))
	return float(rsi)


def _daily_correlation(symbol: str, parent: str, days: int = 60) -> float | None:
	try:
		data = yf.download(
			tickers=f"{symbol} {parent}",
			period=f"{days}d",
			interval='1d',
			group_by='ticker',
			threads=True,
			auto_adjust=False,
		)
		if data is None or data.empty:
			return None
		def _series(sym: str) -> pd.Series | None:
			if isinstance(data.columns, pd.MultiIndex) and sym in data:
				return _extract_close_series(data[sym])
			return _extract_close_series(data)
		s1 = _series(symbol)
		s2 = _series(parent)
		if s1 is None or s2 is None:
			return None
		returns = pd.concat([s1.pct_change(), s2.pct_change()], axis=1).dropna()
		if returns.shape[0] < 10:
			return None
		return float(returns.iloc[:, 0].corr(returns.iloc[:, 1]))
	except Exception:
		return None


def _load_signal_model_payload(model_path: Path) -> dict[str, Any] | None:
	if model_path.exists():
		try:
			payload = joblib.load(model_path)
			if isinstance(payload, dict) and payload.get('model') and payload.get('features'):
				return payload
			if hasattr(payload, 'predict_proba'):
				return {'model': payload, 'features': FEATURE_COLUMNS}
		except Exception:
			return None
	return None


def _predict_model_signal(payload: dict[str, Any], fusion_df: pd.DataFrame) -> float | None:
	if not payload or fusion_df is None or fusion_df.empty:
		return None
	feature_list = payload.get('features') or []
	if not feature_list:
		return None
	last_row = fusion_df.tail(1).copy()
	for col in feature_list:
		if col not in last_row.columns:
			last_row[col] = 0.0
	features = last_row[feature_list].fillna(0).values
	try:
		signal = float(payload['model'].predict_proba(features)[0][1])
		return float(apply_feature_weighting_to_signal(signal, last_row.iloc[0], ''))
	except Exception:
		return None


class StockViewSet(viewsets.ModelViewSet):
	queryset = Stock.objects.all()
	serializer_class = StockSerializer

	@action(detail=False, methods=['post'])
	def create_from_symbol(self, request):
		symbol = (request.data.get('symbol') or '').strip().upper()
		if not symbol:
			return Response({'error': 'symbol is required'}, status=400)

		existing = Stock.objects.filter(symbol__iexact=symbol).first()
		if existing:
			return Response(StockSerializer(existing).data)

		try:
			info = yf.Ticker(symbol).info or {}
		except Exception:
			info = {}

		name = info.get('longName') or info.get('shortName') or symbol
		sector = info.get('sector') or 'Unknown'
		dividend_yield = info.get('dividendYield')
		if dividend_yield is None:
			dividend_yield = 0.0
		try:
			dividend_yield = float(dividend_yield or 0)
		except Exception:
			dividend_yield = 0.0

		stock = Stock.objects.create(
			symbol=symbol,
			name=name,
			sector=sector,
			target_weight=0.0,
			dividend_yield=dividend_yield,
		)
		return Response(StockSerializer(stock).data, status=201)


class StockSearchView(APIView):
	def get(self, request):
		query = (request.query_params.get('q') or '').strip()
		if not query:
			return Response({'results': []})

		url = f"https://query1.finance.yahoo.com/v1/finance/search?q={quote(query)}&quotesCount=25&newsCount=0"
		try:
			req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
			with urlopen(req, timeout=6) as resp:
				payload = json.loads(resp.read().decode('utf-8'))
		except Exception:
			return Response({'results': []})

		results = []
		for item in payload.get('quotes', [])[:12]:
			symbol = item.get('symbol')
			name = item.get('shortname') or item.get('longname') or ''
			exchange = item.get('exchDisp') or item.get('exchange')
			asset_type = item.get('quoteType')
			if symbol:
				results.append({
					'symbol': symbol,
					'name': name,
					'exchange': exchange,
					'type': asset_type,
				})

		return Response({'results': results})


class PortfolioViewSet(viewsets.ModelViewSet):
	queryset = Portfolio.objects.all()
	serializer_class = PortfolioSerializer

	@action(detail=True, methods=['get'])
	def drip_projection(self, request, pk=None):
		portfolio = self.get_object()
		years = int(request.query_params.get('years', 10))
		years = max(1, min(years, 50))

		transactions = portfolio.transaction_set.select_related('stock')
		total_invested = 0.0
		per_symbol = {}

		for tx in transactions:
			sign = -1 if tx.transaction_type == 'SELL' else 1
			shares = float(tx.shares or 0) * sign
			invested = shares * float(tx.price_per_share or 0)
			symbol = tx.stock.symbol
			dy = float(tx.stock.dividend_yield or 0)

			total_invested += invested
			if symbol not in per_symbol:
				per_symbol[symbol] = {'invested': 0.0, 'dividend_yield': dy}
			per_symbol[symbol]['invested'] += invested
			if dy:
				per_symbol[symbol]['dividend_yield'] = dy

		if total_invested > 0:
			weighted_yield = 0.0
			for entry in per_symbol.values():
				if entry['invested'] > 0:
					weighted_yield += (entry['invested'] / total_invested) * entry['dividend_yield']
			capital = total_invested
			yield_rate = weighted_yield
		else:
			stocks = list(portfolio.stocks.all())
			weights = [float(s.target_weight or 0) for s in stocks]
			total_weights = sum(weights)
			if total_weights > 1.5:
				weights = [w / 100 for w in weights]
				total_weights = sum(weights)
			total_weights = total_weights or 1
			yield_rate = sum(
				(weights[i] / total_weights) * float(stocks[i].dividend_yield or 0)
				for i in range(len(stocks))
			) if stocks else 0.02
			capital = float(portfolio.capital or 0)

		values = [capital]
		for _ in range(years):
			values.append(values[-1] * (1 + yield_rate))

		return Response({
			'portfolio_id': portfolio.id,
			'capital': capital,
			'yield_rate': yield_rate,
			'years': years,
			'values': values,
		})

	@action(detail=True, methods=['get'])
	def summary(self, request, pk=None):
		portfolio = self.get_object()
		years = int(request.query_params.get('years', 10))
		years = max(1, min(years, 50))

		# DRIP projection (same logic as drip_projection)
		transactions = portfolio.transaction_set.select_related('stock')
		total_invested = 0.0
		per_symbol = {}

		for tx in transactions:
			sign = -1 if tx.transaction_type == 'SELL' else 1
			shares = float(tx.shares or 0) * sign
			invested = shares * float(tx.price_per_share or 0)
			symbol = tx.stock.symbol
			dy = float(tx.stock.dividend_yield or 0)

			total_invested += invested
			if symbol not in per_symbol:
				per_symbol[symbol] = {'invested': 0.0, 'dividend_yield': dy}
			per_symbol[symbol]['invested'] += invested
			if dy:
				per_symbol[symbol]['dividend_yield'] = dy

		if total_invested > 0:
			weighted_yield = 0.0
			for entry in per_symbol.values():
				if entry['invested'] > 0:
					weighted_yield += (entry['invested'] / total_invested) * entry['dividend_yield']
			capital = total_invested
			yield_rate = weighted_yield
		else:
			stocks = list(portfolio.stocks.all())
			weights = [float(s.target_weight or 0) for s in stocks]
			total_weights = sum(weights)
			if total_weights > 1.5:
				weights = [w / 100 for w in weights]
				total_weights = sum(weights)
			total_weights = total_weights or 1
			yield_rate = sum(
				(weights[i] / total_weights) * float(stocks[i].dividend_yield or 0)
				for i in range(len(stocks))
			) if stocks else 0.02
			capital = float(portfolio.capital or 0)

		values = [capital]
		for _ in range(years):
			values.append(values[-1] * (1 + yield_rate))

		total_dividends = (
			DripSnapshot.objects.filter(portfolio=portfolio)
			.aggregate(total=models.Sum('dividend_income'))
			.get('total')
		) or 0.0

		tx_data = TransactionSerializer(transactions, many=True).data

		return Response({
			'portfolio_id': portfolio.id,
			'total_capital': float(portfolio.capital or 0),
			'total_dividends_accrued': float(total_dividends),
			'drip_projection': {
				'years': years,
				'capital': capital,
				'yield_rate': yield_rate,
				'values': values,
			},
			'transactions': tx_data,
		})

	@action(detail=True, methods=['get'])
	def insights(self, request, pk=None):
		portfolio = self.get_object()
		days = int(request.query_params.get('days', 90))
		days = max(30, min(days, 365))
		rebalance_threshold = float(request.query_params.get('threshold', 0.02))

		holdings = PortfolioHolding.objects.select_related('stock').filter(portfolio=portfolio)
		total_value = 0.0
		holding_values = []

		for h in holdings:
			price = h.stock.latest_price
			if price is None:
				last_price = (
					PriceHistory.objects.filter(stock=h.stock).order_by('-date').first()
				)
				price = last_price.close_price if last_price else 0
			value = float(h.shares or 0) * float(price or 0)
			total_value += value
			holding_values.append({
				'symbol': h.stock.symbol,
				'target_weight': float(h.stock.target_weight or 0),
				'value': value,
			})

		rebalancing = []
		if total_value > 0:
			for hv in holding_values:
				actual_weight = hv['value'] / total_value
				target_weight = hv['target_weight']
				if target_weight > 1.5:
					target_weight /= 100

				diff = actual_weight - target_weight
				if abs(diff) >= rebalance_threshold:
					action = 'SELL' if diff > 0 else 'BUY'
					amount = abs(diff) * total_value
					rebalancing.append({
						'symbol': hv['symbol'],
						'actual_weight': actual_weight,
						'target_weight': target_weight,
						'action': action,
						'amount': amount,
					})

		# Risk metrics using price history
		price_frames = []
		for h in holdings:
			prices = PriceHistory.objects.filter(stock=h.stock).order_by('-date')[:days]
			if prices.count() < 2:
				continue
			series = pd.Series(
				{p.date: p.close_price for p in prices},
				name=h.stock.symbol,
			)
			price_frames.append(series)

		risk = {
			'volatility': None,
			'max_drawdown': None,
			'var_95': None,
			'beta_spy': None,
		}

		if price_frames:
			df = pd.concat(price_frames, axis=1).sort_index()
			returns = df.pct_change().dropna(how='all')

			if not returns.empty and total_value > 0:
				weights = []
				for h in holdings:
					price = h.stock.latest_price
					if price is None:
						last_price = (
							PriceHistory.objects.filter(stock=h.stock).order_by('-date').first()
						)
						price = last_price.close_price if last_price else 0
					value = float(h.shares or 0) * float(price or 0)
					weights.append(value / total_value if total_value else 0)

				weights = np.array(weights)
				weights = weights[: returns.shape[1]]
				portfolio_returns = (returns.fillna(0).values * weights).sum(axis=1)

				vol = np.std(portfolio_returns) * np.sqrt(252)
				cumulative = (1 + portfolio_returns).cumprod()
				peak = np.maximum.accumulate(cumulative)
				drawdown = (cumulative - peak) / peak
				max_drawdown = float(drawdown.min())
				var_95 = float(np.quantile(portfolio_returns, 0.05))

				risk['volatility'] = float(vol)
				risk['max_drawdown'] = max_drawdown
				risk['var_95'] = var_95

				# Beta vs SPY (if available)
				try:
					spy = yf.download('SPY', period=f'{days}d', interval='1d', progress=False)
					if not spy.empty and 'Close' in spy:
						spy_returns = spy['Close'].pct_change().dropna().values
						aligned = min(len(spy_returns), len(portfolio_returns))
						if aligned > 10:
							cov = np.cov(portfolio_returns[-aligned:], spy_returns[-aligned:])[0, 1]
							beta = cov / np.var(spy_returns[-aligned:]) if np.var(spy_returns[-aligned:]) else 0
							risk['beta_spy'] = float(beta)
				except Exception:
					pass

		return Response({
			'portfolio_id': portfolio.id,
			'total_value': total_value,
			'rebalancing': rebalancing,
			'risk': risk,
		})

	@action(detail=True, methods=['get'])
	def decision_support(self, request, pk=None):
		portfolio = self.get_object()

		holdings = PortfolioHolding.objects.select_related('stock').filter(portfolio=portfolio)
		total_value = 0.0
		holding_data = []

		for h in holdings:
			price = h.stock.latest_price
			if price is None:
				last_price = (
					PriceHistory.objects.filter(stock=h.stock).order_by('-date').first()
				)
				price = last_price.close_price if last_price else 0
			value = float(h.shares or 0) * float(price or 0)
			total_value += value
			holding_data.append({
				'stock': h.stock,
				'value': value,
				'price': float(price or 0),
			})

		recommendations = []
		for item in holding_data:
			stock = item['stock']
			prices_qs = list(
				PriceHistory.objects.filter(stock=stock).order_by('-date')[:200]
			)
			prices = [float(p.close_price) for p in prices_qs if p.close_price is not None]
			last_close = prices[0] if prices else float(item['price'] or 0)
			actual_weight = item['value'] / total_value if total_value else 0
			target_weight = float(stock.target_weight or 0)
			if target_weight > 1.5:
				target_weight /= 100

			latest_prediction = (
				Prediction.objects.filter(stock=stock).order_by('-date').first()
			)
			sentiment_avg = (
				StockNews.objects.filter(stock=stock)
				.aggregate(avg=models.Avg('sentiment'))
				.get('avg')
			) or 0

			score = 0.0
			reasons = []
			bearish_tech = False

			# Prediction signal
			if latest_prediction and item['price']:
				pred_diff = (latest_prediction.predicted_price - item['price']) / item['price']
				score += pred_diff * 2
				if pred_diff > 0.02:
					reasons.append('Price forecast above current')
				elif pred_diff < -0.02:
					reasons.append('Price forecast below current')

			# Technical trend signal (conservative)
			if prices:
				sma20 = sum(prices[:20]) / 20 if len(prices) >= 20 else None
				sma50 = sum(prices[:50]) / 50 if len(prices) >= 50 else None
				sma200 = sum(prices[:200]) / 200 if len(prices) >= 200 else None
				if sma50 and sma200 and last_close:
					if last_close < sma50 and last_close < sma200:
						score -= 0.25
						bearish_tech = True
						reasons.append('Downtrend (below 50/200d)')
				elif sma50 and last_close and last_close < sma50:
					score -= 0.1
					reasons.append('Below 50d average')

				if len(prices) >= 60 and last_close:
					momentum_60d = (last_close - prices[59]) / prices[59] if prices[59] else 0
					if momentum_60d < -0.1:
						score -= 0.2
						bearish_tech = True
						reasons.append('Negative 60d momentum')
					elif momentum_60d > 0.1:
						score += 0.1
						reasons.append('Positive 60d momentum')

			# Sentiment signal
			score += sentiment_avg * 0.5
			if sentiment_avg > 0.2:
				reasons.append('Positive news sentiment')
			elif sentiment_avg < -0.2:
				reasons.append('Negative news sentiment')

			# Dividend signal
			if stock.dividend_yield and stock.dividend_yield > 0.03:
				score += 0.1
				reasons.append('Attractive dividend yield')

			# Rebalancing signal
			diff = actual_weight - target_weight
			if diff > 0.03:
				score -= 0.2
				reasons.append('Over target weight')
			elif diff < -0.03:
				score += 0.2
				reasons.append('Under target weight')

			if score >= 0.15:
				action = 'BUY'
			elif score <= -0.15:
				action = 'SELL'
			else:
				action = 'HOLD'

			# Reliability override: if technicals are bearish, avoid BUY
			if action == 'BUY' and bearish_tech:
				action = 'HOLD'
				reasons.append('Technicals bearish override')

			# Rebalancing-aware adjustment: avoid selling strong positives
			if diff > 0.03:
				if score <= 0:
					action = 'SELL'
				else:
					action = 'HOLD'
					reasons.append('Overweight but positive signal')
			elif diff < -0.03 and score >= 0.1:
				action = 'BUY'

			recommendations.append({
				'symbol': stock.symbol,
				'action': action,
				'score': score,
				'reasons': reasons,
				'actual_weight': actual_weight,
				'target_weight': target_weight,
			})

		# Scenario simulation: 20% market crash
		crash_pct = float(request.query_params.get('crash_pct', 0.2))
		crash_pct = min(max(crash_pct, 0.05), 0.8)
		crash_value = total_value * (1 - crash_pct)
		drip_yield = 0.0
		if total_value > 0:
			drip_yield = sum(
				(float(h.stock.dividend_yield or 0) * (item['value'] / total_value))
				for h, item in zip(holdings, holding_data)
			)
		drip_monthly = crash_value * (drip_yield / 12)

		mitigation = []
		if crash_pct >= 0.2:
			mitigation.append('Increase cash buffer or pause DRIP temporarily')
		if recommendations:
			mitigation.append('Rebalance toward target weights to reduce concentration risk')

		# Tax-efficient heuristics (non-advisory)
		tax_plan = {
			'note': 'Heuristic only; not tax advice.',
			'withdrawal_order': ['CASH', 'CRI', 'TFSA'],
			'dividend_reinvest': 'Consider reinvesting in TFSA/CRI if room is available; keep cash for near-term needs.',
		}

		return Response({
			'portfolio_id': portfolio.id,
			'total_value': total_value,
			'recommendations': recommendations,
			'scenario': {
				'crash_pct': crash_pct,
				'post_crash_value': crash_value,
				'estimated_monthly_dividend': drip_monthly,
				'mitigation': mitigation,
			},
			'tax_plan': tax_plan,
		})


class TransactionViewSet(viewsets.ModelViewSet):
	queryset = Transaction.objects.all()
	serializer_class = TransactionSerializer


class PriceHistoryViewSet(viewsets.ReadOnlyModelViewSet):
	queryset = PriceHistory.objects.select_related('stock').all()
	serializer_class = PriceHistorySerializer

	def get_queryset(self):
		queryset = super().get_queryset()
		symbol = self.request.query_params.get('symbol')
		if symbol:
			queryset = queryset.filter(stock__symbol__iexact=symbol)
		return queryset


class PortfolioHoldingViewSet(viewsets.ReadOnlyModelViewSet):
	queryset = PortfolioHolding.objects.select_related('portfolio', 'stock').all()
	serializer_class = PortfolioHoldingSerializer

	def get_queryset(self):
		queryset = super().get_queryset()
		portfolio_id = self.request.query_params.get('portfolio_id')
		symbol = self.request.query_params.get('symbol')

		if portfolio_id:
			queryset = queryset.filter(portfolio_id=portfolio_id)
		if symbol:
			queryset = queryset.filter(stock__symbol__iexact=symbol)

		return queryset


class StockNewsViewSet(viewsets.ReadOnlyModelViewSet):
	queryset = StockNews.objects.select_related('stock').all()
	serializer_class = StockNewsSerializer

	def get_queryset(self):
		queryset = super().get_queryset()
		symbol = self.request.query_params.get('symbol')
		source = self.request.query_params.get('source')
		q = self.request.query_params.get('q')
		days = self.request.query_params.get('days')
		sentiment_min = self.request.query_params.get('sentiment_min')
		sentiment_max = self.request.query_params.get('sentiment_max')
		if symbol:
			queryset = queryset.filter(stock__symbol__iexact=symbol)
		if source:
			queryset = queryset.filter(source__icontains=source)
		if q:
			queryset = queryset.filter(headline__icontains=q)
		if days is None:
			days = 5
		try:
			days = int(days)
		except (TypeError, ValueError):
			raise ValidationError({'days': 'Invalid days.'})
		days = max(1, min(days, 90))
		cutoff = timezone.now() - timedelta(days=days)
		queryset = queryset.filter(
			models.Q(published_at__gte=cutoff)
			| models.Q(published_at__isnull=True, fetched_at__gte=cutoff)
		)
		if sentiment_min is not None:
			try:
				sentiment_min = float(sentiment_min)
			except ValueError:
				raise ValidationError({'sentiment_min': 'Invalid sentiment_min.'})
			queryset = queryset.filter(sentiment__gte=sentiment_min)
		if sentiment_max is not None:
			try:
				sentiment_max = float(sentiment_max)
			except ValueError:
				raise ValidationError({'sentiment_max': 'Invalid sentiment_max.'})
			queryset = queryset.filter(sentiment__lte=sentiment_max)
		return queryset


class DripSnapshotViewSet(viewsets.ReadOnlyModelViewSet):
	queryset = DripSnapshot.objects.select_related('portfolio').all()
	serializer_class = DripSnapshotSerializer

	def get_queryset(self):
		queryset = super().get_queryset()
		portfolio_id = self.request.query_params.get('portfolio_id')
		if portfolio_id:
			queryset = queryset.filter(portfolio_id=portfolio_id)
		return queryset


class AlertEventViewSet(viewsets.ReadOnlyModelViewSet):
	queryset = AlertEvent.objects.select_related('stock', 'portfolio').all()
	serializer_class = AlertEventSerializer


class AccountViewSet(viewsets.ModelViewSet):
	queryset = Account.objects.all()
	serializer_class = AccountSerializer

	def perform_create(self, serializer):
		user = None
		if hasattr(self.request, 'user') and self.request.user and self.request.user.is_authenticated:
			user = self.request.user
		else:
			User = get_user_model()
			user = User.objects.first()
			if user is None:
				username_field = User.USERNAME_FIELD
				if username_field == User.EMAIL_FIELD:
					identifier = 'local@example.com'
				else:
					identifier = 'local'
				lookup = {username_field: identifier}
				user = User.objects.filter(**lookup).first()
				if user is None:
					user = User(**lookup)
					user.set_unusable_password()
					user.save()
		serializer.save(user=user)


class AccountTransactionViewSet(viewsets.ModelViewSet):
	queryset = AccountTransaction.objects.select_related('account', 'stock').all()
	serializer_class = AccountTransactionSerializer


class DividendViewSet(viewsets.ModelViewSet):
	queryset = Dividend.objects.select_related('stock').all()
	serializer_class = DividendSerializer


class PredictionViewSet(viewsets.ModelViewSet):
	queryset = Prediction.objects.select_related('stock').all()
	serializer_class = PredictionSerializer

	def get_queryset(self):
		queryset = super().get_queryset()
		symbol = self.request.query_params.get('symbol')
		reco = self.request.query_params.get('recommendation')
		date_from = self.request.query_params.get('date_from')
		date_to = self.request.query_params.get('date_to')
		if symbol:
			queryset = queryset.filter(stock__symbol__iexact=symbol)
		if reco:
			reco = reco.upper()
			if reco not in {'BUY', 'HOLD', 'SELL'}:
				raise ValidationError({'recommendation': 'Invalid recommendation.'})
			queryset = queryset.filter(recommendation=reco)
		if date_from:
			try:
				date_from = datetime.fromisoformat(date_from).date()
			except ValueError:
				raise ValidationError({'date_from': 'Invalid date_from.'})
			queryset = queryset.filter(date__gte=date_from)
		if date_to:
			try:
				date_to = datetime.fromisoformat(date_to).date()
			except ValueError:
				raise ValidationError({'date_to': 'Invalid date_to.'})
			queryset = queryset.filter(date__lte=date_to)
		return queryset.order_by('-date', 'stock__symbol')


class PaperTradeViewSet(viewsets.ReadOnlyModelViewSet):
	queryset = PaperTrade.objects.all()
	serializer_class = PaperTradeSerializer

	def get_queryset(self):
		queryset = super().get_queryset()
		status = self.request.query_params.get('status')
		sandbox = self.request.query_params.get('sandbox')
		ticker = self.request.query_params.get('ticker')
		model_name = self.request.query_params.get('model_name')
		model_version = self.request.query_params.get('model_version')
		outcome = self.request.query_params.get('outcome')
		entry_from = self.request.query_params.get('entry_from')
		entry_to = self.request.query_params.get('entry_to')
		if status:
			status = status.upper()
			if status not in {'OPEN', 'CLOSED'}:
				raise ValidationError({'status': 'Invalid status.'})
			queryset = queryset.filter(status=status)
		if sandbox:
			sandbox = sandbox.upper()
			if sandbox not in {'WATCHLIST', 'AI_BLUECHIP', 'AI_PENNY'}:
				raise ValidationError({'sandbox': 'Invalid sandbox.'})
			queryset = queryset.filter(sandbox=sandbox)
		if ticker:
			queryset = queryset.filter(ticker__iexact=ticker)
		if model_name:
			queryset = queryset.filter(model_name__iexact=model_name)
		if model_version:
			queryset = queryset.filter(model_version=model_version)
		if outcome:
			outcome = outcome.upper()
			if outcome not in {'WIN', 'LOSS'}:
				raise ValidationError({'outcome': 'Invalid outcome.'})
			queryset = queryset.filter(outcome=outcome)
		if entry_from:
			try:
				entry_from = datetime.fromisoformat(entry_from)
			except ValueError:
				raise ValidationError({'entry_from': 'Invalid entry_from.'})
			queryset = queryset.filter(entry_date__gte=entry_from)
		if entry_to:
			try:
				entry_to = datetime.fromisoformat(entry_to)
			except ValueError:
				raise ValidationError({'entry_to': 'Invalid entry_to.'})
			queryset = queryset.filter(entry_date__lte=entry_to)
		return queryset


class PortfolioDigestViewSet(viewsets.ReadOnlyModelViewSet):
	queryset = PortfolioDigest.objects.select_related('portfolio').all()
	serializer_class = PortfolioDigestSerializer

	def get_queryset(self):
		queryset = super().get_queryset()
		portfolio_id = self.request.query_params.get('portfolio_id')
		if portfolio_id:
			queryset = queryset.filter(portfolio_id=portfolio_id)
		return queryset


class UserPreferenceViewSet(viewsets.ModelViewSet):
	queryset = UserPreference.objects.select_related('user').all()
	serializer_class = UserPreferenceSerializer


class PaperTradeSummaryView(APIView):
	def get(self, request):
		sandbox = (request.query_params.get('sandbox') or '').strip().upper()
		initial_capital = float(os.getenv('PAPER_CAPITAL', '10000'))
		try:
			def _latest_price(symbol: str) -> float | None:
				symbol = (symbol or '').strip().upper()
				if not symbol:
					return None
				try:
					hist = yf.Ticker(symbol).history(period='1d', interval='1m', timeout=10)
					close = hist['Close'].iloc[-1] if hist is not None and not hist.empty and 'Close' in hist else None
					return float(close) if close is not None else None
				except Exception:
					return None

			open_trades = PaperTrade.objects.filter(status='OPEN')
			closed_trades = PaperTrade.objects.filter(status='CLOSED')
			if sandbox:
				open_trades = open_trades.filter(sandbox=sandbox)
				closed_trades = closed_trades.filter(sandbox=sandbox)

			closed_pnl = float(sum([float(t.pnl or 0) for t in closed_trades]))
			open_value = 0.0
			total_risk = 0.0
			for t in open_trades:
				open_value += float(t.entry_price) * float(t.quantity)
				entry_price = float(t.entry_price or 0)
				stop_loss = float(t.stop_loss or 0)
				if entry_price and stop_loss and entry_price > stop_loss:
					total_risk += (entry_price - stop_loss) * float(t.quantity)

			available = initial_capital + closed_pnl - open_value
			open_payload = PaperTradeSerializer(open_trades, many=True).data
			for trade in open_payload:
				symbol = trade.get('ticker') or ''
				current_price = _latest_price(symbol)
				trade['current_price'] = None if current_price is None else round(float(current_price), 4)
				try:
					entry = float(trade.get('entry_price') or 0)
					qty = float(trade.get('quantity') or 0)
					trade['unrealized_pnl'] = None if current_price is None else round((current_price - entry) * qty, 2)
				except Exception:
					trade['unrealized_pnl'] = None

			return Response({
				'sandbox': sandbox or 'ALL',
				'initial_capital': initial_capital,
				'available_capital': round(available, 2),
				'open_value': round(open_value, 2),
				'total_risk': round(total_risk, 2),
				'closed_pnl': round(closed_pnl, 2),
				'open_positions': open_payload,
				'closed_positions': PaperTradeSerializer(closed_trades[:25], many=True).data,
			})
		except Exception:
			return Response({
				'sandbox': sandbox or 'ALL',
				'initial_capital': initial_capital,
				'available_capital': round(initial_capital, 2),
				'open_value': 0,
				'total_risk': 0,
				'closed_pnl': 0,
				'open_positions': [],
				'closed_positions': [],
			})


class PaperTradeManualCreateView(APIView):
	def post(self, request):
		ticker = (request.data.get('ticker') or '').strip().upper()
		if not ticker:
			return Response({'error': 'Ticker requis.'}, status=400)
		sandbox = (request.data.get('sandbox') or 'WATCHLIST').strip().upper()
		if sandbox not in {'WATCHLIST', 'AI_BLUECHIP', 'AI_PENNY'}:
			return Response({'error': 'Sandbox invalide.'}, status=400)

		price = request.data.get('price')
		try:
			price = float(price) if price is not None else None
		except (TypeError, ValueError):
			price = None

		if not price or price <= 0:
			try:
				hist = yf.download(ticker, period='5d', interval='1d')
				if hist is not None and not hist.empty:
					close_col = 'close' if 'close' in hist.columns else 'Close'
					if close_col in hist:
						price = float(hist[close_col].iloc[-1])
			except Exception:
				price = None

		if not price or price <= 0:
			return Response({'error': f"Prix introuvable pour {ticker}."}, status=400)

		suggested = request.data.get('suggested_investment')
		try:
			suggested = float(suggested) if suggested is not None else 0.0
		except (TypeError, ValueError):
			suggested = 0.0
		if suggested <= 0:
			suggested = float(os.getenv('PAPER_TRADE_MANUAL_DEFAULT', '100'))

		quantity = max(1, int(suggested / price)) if price else 0
		if quantity <= 0:
			return Response({'error': 'Quantité invalide.'}, status=400)

		stop_loss = request.data.get('stop_loss')
		try:
			stop_loss = float(stop_loss) if stop_loss is not None else None
		except (TypeError, ValueError):
			stop_loss = None
		if not stop_loss or stop_loss <= 0:
			stop_loss = round(float(price) * 0.95, 2)

		confidence = request.data.get('confidence')
		try:
			entry_signal = float(confidence) / 100 if confidence is not None else None
		except (TypeError, ValueError):
			entry_signal = None

		existing = PaperTrade.objects.filter(status='OPEN', sandbox=sandbox, ticker__iexact=ticker).first()
		if existing:
			return Response({'status': 'exists', 'trade': PaperTradeSerializer(existing).data}, status=200)

		trade = PaperTrade.objects.create(
			ticker=ticker,
			sandbox=sandbox,
			entry_price=round(float(price), 4),
			quantity=quantity,
			entry_signal=entry_signal,
			stop_loss=round(float(stop_loss), 2),
			notes='Manual quick analysis trade',
		)
		return Response({'status': 'created', 'trade': PaperTradeSerializer(trade).data}, status=201)


class SandboxWatchlistView(APIView):
	def get(self, request):
		sandbox = (request.query_params.get('sandbox') or 'WATCHLIST').strip().upper()
		if sandbox not in {'WATCHLIST', 'AI_BLUECHIP', 'AI_PENNY'}:
			raise ValidationError({'sandbox': 'Invalid sandbox.'})
		watch = SandboxWatchlist.objects.filter(sandbox=sandbox).first()
		if not watch:
			watch = SandboxWatchlist.objects.create(sandbox=sandbox, symbols=[], source='manual')
		return Response(SandboxWatchlistSerializer(watch).data)

	def post(self, request):
		sandbox = (request.data.get('sandbox') or 'WATCHLIST').strip().upper()
		if sandbox not in {'WATCHLIST', 'AI_BLUECHIP', 'AI_PENNY'}:
			raise ValidationError({'sandbox': 'Invalid sandbox.'})
		symbols = request.data.get('symbols') or []
		if isinstance(symbols, str):
			symbols = [s.strip() for s in symbols.replace(',', ' ').split() if s.strip()]
		if not isinstance(symbols, list):
			symbols = []
		clean = [str(s).strip().upper() for s in symbols if str(s).strip()]
		clean = list(dict.fromkeys(clean))
		watch, _ = SandboxWatchlist.objects.update_or_create(
			sandbox=sandbox,
			defaults={'symbols': clean, 'source': 'manual'},
		)
		return Response(SandboxWatchlistSerializer(watch).data)


class PaperTradeExplanationLogView(APIView):
	def get(self, request):
		sandbox = (request.query_params.get('sandbox') or '').strip().upper()
		status = (request.query_params.get('status') or '').strip().upper()
		page = max(int(request.query_params.get('page', 1)), 1)
		page_size = max(min(int(request.query_params.get('page_size', 25)), 200), 1)
		if sandbox and sandbox not in {'WATCHLIST', 'AI_BLUECHIP', 'AI_PENNY'}:
			raise ValidationError({'sandbox': 'Invalid sandbox.'})
		if status and status not in {'OPEN', 'CLOSED'}:
			raise ValidationError({'status': 'Invalid status.'})
		qs = PaperTrade.objects.all().order_by('-entry_date')
		if sandbox:
			qs = qs.filter(sandbox=sandbox)
		if status:
			qs = qs.filter(status=status)
		total = qs.count()
		start = (page - 1) * page_size
		end = start + page_size
		rows = PaperTradeSerializer(qs[start:end], many=True).data
		return Response({
			'count': total,
			'page': page,
			'page_size': page_size,
			'results': rows,
		})


class ModelMonitoringSummaryView(APIView):
	def get(self, request):
		model_name = (request.query_params.get('model') or '').strip().upper()
		sandbox = (request.query_params.get('sandbox') or '').strip().upper()
		if model_name and model_name not in {'BLUECHIP', 'PENNY'}:
			raise ValidationError({'model': 'Invalid model.'})
		if sandbox and sandbox not in {'WATCHLIST', 'AI_BLUECHIP', 'AI_PENNY'}:
			raise ValidationError({'sandbox': 'Invalid sandbox.'})

		def _latest(qs, fields: list[str]) -> dict[str, Any] | None:
			item = qs.order_by('-as_of').first()
			if not item:
				return None
			return {field: getattr(item, field) for field in fields}

		try:
			results = []
			base = {
				'models': [model_name] if model_name else ['BLUECHIP', 'PENNY'],
				'sandboxes': [sandbox] if sandbox else ['WATCHLIST', 'AI_BLUECHIP', 'AI_PENNY'],
			}
			for m in base['models']:
				for sb in base['sandboxes']:
					cal = _latest(
						ModelCalibrationDaily.objects.filter(model_name=m, sandbox=sb),
						['as_of', 'model_version', 'brier_score', 'count', 'bins'],
					)
					drift = _latest(
						ModelDriftDaily.objects.filter(model_name=m, sandbox=sb),
						['as_of', 'model_version', 'psi', 'feature_stats'],
					)
					eval_entry = _latest(
						ModelEvaluationDaily.objects.filter(model_name=m, sandbox=sb),
						['as_of', 'model_version', 'trades', 'win_rate', 'avg_pnl', 'total_pnl', 'max_drawdown', 'brier_score'],
					)
					results.append({
						'model_name': m,
						'sandbox': sb,
						'calibration': cal,
						'drift': drift,
						'evaluation': eval_entry,
					})

			return Response({'results': results})
		except OperationalError:
			return Response({'results': []})


class PaperTradePerformanceView(APIView):
	def _initial_capital(self, sandbox: str) -> float:
		if sandbox == 'AI_BLUECHIP':
			return float(os.getenv('AI_BLUECHIP_CAPITAL', os.getenv('PAPER_CAPITAL', '10000')))
		if sandbox == 'AI_PENNY':
			return float(os.getenv('AI_PENNY_CAPITAL', os.getenv('PAPER_CAPITAL', '10000')))
		return float(os.getenv('PAPER_CAPITAL', '10000'))

	def _max_drawdown(self, equity_curve: list[float]) -> float:
		peak = None
		max_dd = 0.0
		for value in equity_curve:
			peak = value if peak is None else max(peak, value)
			if peak:
				dd = (value - peak) / peak
				max_dd = min(max_dd, dd)
		return abs(max_dd)

	def get(self, request):
		sandbox_param = (request.query_params.get('sandbox') or '').strip().upper()
		sandboxes = [sandbox_param] if sandbox_param else ['WATCHLIST', 'AI_BLUECHIP', 'AI_PENNY']
		results = []
		for sandbox in sandboxes:
			initial_capital = self._initial_capital(sandbox)
			closed = PaperTrade.objects.filter(status='CLOSED', sandbox=sandbox).order_by('exit_date')
			trades = closed.count()
			wins = closed.filter(outcome='WIN').count()
			if wins == 0 and trades:
				wins = sum([1 for t in closed if float(t.pnl or 0) > 0])
			win_rate = (wins / trades) * 100 if trades else 0.0
			total_pnl = float(sum([float(t.pnl or 0) for t in closed]))
			total_return_pct = (total_pnl / initial_capital) * 100 if initial_capital else 0.0
			returns = []
			equity = initial_capital
			equity_curve = [equity]
			for t in closed:
				entry_value = float(t.entry_price) * float(t.quantity)
				trade_return = float(t.pnl or 0) / entry_value if entry_value else 0.0
				returns.append(trade_return)
				equity += float(t.pnl or 0)
				equity_curve.append(equity)
			mean_ret = float(np.mean(returns)) if returns else 0.0
			std_ret = float(np.std(returns)) if returns else 0.0
			sharpe = (mean_ret / std_ret) * np.sqrt(len(returns)) if std_ret else 0.0
			max_drawdown = self._max_drawdown(equity_curve)
			results.append({
				'sandbox': sandbox,
				'initial_capital': round(initial_capital, 2),
				'trades': trades,
				'win_rate': round(win_rate, 2),
				'total_return_pct': round(total_return_pct, 2),
				'sharpe_ratio': round(float(sharpe), 3),
				'max_drawdown': round(float(max_drawdown) * 100, 2),
				'final_balance': round(equity, 2),
			})

		return Response({'results': results})


class PaperTradeEquityCurveView(APIView):
	def _initial_capital(self, sandbox: str) -> float:
		if sandbox == 'AI_BLUECHIP':
			return float(os.getenv('AI_BLUECHIP_CAPITAL', os.getenv('PAPER_CAPITAL', '10000')))
		if sandbox == 'AI_PENNY':
			return float(os.getenv('AI_PENNY_CAPITAL', os.getenv('PAPER_CAPITAL', '10000')))
		return float(os.getenv('PAPER_CAPITAL', '10000'))

	def get(self, request):
		sandboxes = ['WATCHLIST', 'AI_BLUECHIP', 'AI_PENNY']
		curves: dict[str, dict[str, float]] = {s: {} for s in sandboxes}
		all_dates: set[str] = set()

		for sandbox in sandboxes:
			capital = self._initial_capital(sandbox)
			closed = PaperTrade.objects.filter(status='CLOSED', sandbox=sandbox).exclude(exit_date__isnull=True).order_by('exit_date')
			for trade in closed:
				date_key = trade.exit_date.date().isoformat()
				capital += float(trade.pnl or 0)
				curves[sandbox][date_key] = round(capital, 2)
				all_dates.add(date_key)

		ordered_dates = sorted(all_dates)
		series = []
		for sandbox in sandboxes:
			capital = self._initial_capital(sandbox)
			data = []
			for date_key in ordered_dates:
				if date_key in curves[sandbox]:
					capital = curves[sandbox][date_key]
				data.append(capital)
			series.append({'sandbox': sandbox, 'equity_curve': data})

		if ordered_dates:
			try:
				start = datetime.fromisoformat(ordered_dates[0]).date()
				end = datetime.fromisoformat(ordered_dates[-1]).date() + timedelta(days=1)
				hist = yf.Ticker('SPY').history(start=start.isoformat(), end=end.isoformat(), interval='1d')
				if hist is not None and not hist.empty and 'Close' in hist:
					spy_map = {idx.date().isoformat(): float(val) for idx, val in hist['Close'].items()}
					first_price = spy_map.get(ordered_dates[0])
					if first_price:
						base_capital = float(os.getenv('PAPER_CAPITAL', '10000'))
						spy_curve = []
						for date_key in ordered_dates:
							price = spy_map.get(date_key, first_price)
							spy_curve.append(round(base_capital * (price / first_price), 2))
						series.append({'sandbox': 'SPY_BUY_HOLD', 'equity_curve': spy_curve})
			except Exception:
				pass

		return Response({'dates': ordered_dates, 'series': series})


class PennySignalViewSet(viewsets.ReadOnlyModelViewSet):
	queryset = PennySignal.objects.all()
	serializer_class = PennySignalSerializer


class PennyStockAnalyticsView(APIView):
	def get(self, request):
		limit = int(request.query_params.get('limit', 25))
		min_score = float(request.query_params.get('min_score', 0))
		max_price = float(request.query_params.get('max_price', 1.0))
		min_volume = float(request.query_params.get('min_volume', 100000))
		limit = max(1, min(limit, 100))

		latest_as_of = PennyStockSnapshot.objects.filter(stock_id=models.OuterRef('stock_id')) \
			.order_by('-as_of').values('as_of')[:1]

		queryset = (
			PennyStockSnapshot.objects.select_related('stock')
			.filter(as_of=models.Subquery(latest_as_of))
			.filter(models.Q(ai_score__gte=min_score) | models.Q(ai_score__isnull=True))
			.filter(models.Q(price__lte=max_price) | models.Q(price__isnull=True))
			.filter(models.Q(volume__gte=min_volume) | models.Q(volume__isnull=True))
			.order_by(models.F('ai_score').desc(nulls_last=True), '-as_of')
		)[:limit]

		return Response(PennyStockSnapshotSerializer(queryset, many=True).data)


class PennyStockScoutView(APIView):
	def get(self, request):
		limit = int(request.query_params.get('limit', 50))
		limit = max(1, min(limit, 200))
		snapshots_qs = PennyStockSnapshot.objects.order_by('-as_of')
		queryset = (
			PennyStockUniverse.objects.all()
			.prefetch_related(
				Prefetch('pennystocksnapshot_set', queryset=snapshots_qs, to_attr='latest_snapshots')
			)
			.order_by('symbol')
		)[:limit]

		return Response(PennyStockScoutSerializer(queryset, many=True).data)


class PennyStockPredictionView(APIView):
	def get(self, request, symbol: str):
		symbol = (symbol or '').strip().upper()
		if not symbol:
			return Response({'error': 'symbol is required'}, status=400)

		def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
			return max(low, min(high, value))

		stock_obj = PennyStockUniverse.objects.filter(symbol__iexact=symbol).first()
		latest_snapshot = None
		if stock_obj:
			latest_snapshot = (
				PennyStockSnapshot.objects.filter(stock=stock_obj)
				.order_by('-as_of')
				.first()
			)

		if latest_snapshot:
			flags = latest_snapshot.flags or {}
			fundamental_score = flags.get('fundamental_score')
			if fundamental_score is None:
				months_of_cash = None
				if latest_snapshot.cash is not None and latest_snapshot.burn_rate:
					months_of_cash = latest_snapshot.cash / latest_snapshot.burn_rate
					fundamental_score = _clamp(months_of_cash / 6)
				else:
					fundamental_score = 0.5

			technical_score = flags.get('technical_score')
			if technical_score is None:
				rsi_score = 0.5
				if latest_snapshot.rsi is not None:
					rsi_score = 1 - _clamp((latest_snapshot.rsi - 30) / 40)
				macd_score = 0.5
				if latest_snapshot.macd_hist is not None:
					macd_score = 1.0 if latest_snapshot.macd_hist > 0 else 0.3
				technical_score = _clamp((rsi_score + macd_score) / 2)

			sentiment_norm = flags.get('sentiment_norm')
			if sentiment_norm is None:
				if latest_snapshot.sentiment_score is not None:
					sentiment_norm = _clamp((latest_snapshot.sentiment_score + 1) / 2)
				else:
					sentiment_norm = 0.5

			liquidity_score = flags.get('liquidity_score')
			if liquidity_score is None:
				liquidity_score = _clamp((latest_snapshot.volume or 0) / 1_000_000)

			dilution_penalty = latest_snapshot.dilution_score or 0.0
			ai_score = 100 * _clamp(
				(0.25 * float(fundamental_score))
				+ (0.25 * float(technical_score))
				+ (0.25 * float(sentiment_norm))
				+ (0.25 * float(liquidity_score))
				- float(dilution_penalty)
			)
			up_probability = _clamp(ai_score / 100)

			return Response({
				'symbol': symbol,
				'up_probability': round(up_probability, 4),
				'ai_score': round(ai_score, 2),
				'recommendation': 'BUY' if up_probability >= 0.7 else 'HOLD',
				'components': {
					'fundamental_score': round(float(fundamental_score), 4),
					'technical_score': round(float(technical_score), 4),
					'sentiment_score': round(float(sentiment_norm), 4),
					'liquidity_score': round(float(liquidity_score), 4),
					'dilution_penalty': round(float(dilution_penalty), 4),
				},
			})

		model_path = Path(__file__).resolve().parent / 'ml_engine' / 'scout_brain_v1.pkl'
		if not model_path.exists():
			return Response({'error': 'Model not trained yet.'}, status=503)

		try:
			payload = joblib.load(model_path)
		except Exception:
			return Response({'error': 'Failed to load model.'}, status=503)

		model = payload.get('model') if isinstance(payload, dict) else payload
		feature_list = payload.get('features') if isinstance(payload, dict) else None

		try:
			data = yf.Ticker(symbol).history(period='1y', interval='1d', timeout=10)
		except Exception:
			data = None

		if data is None or data.empty or 'Close' not in data or len(data) < 60:
			return Response({'error': 'Insufficient price history.'}, status=400)

		def rsi(series: pd.Series, window: int = 14) -> float:
			delta = series.diff()
			gain = delta.clip(lower=0).rolling(window).mean()
			loss = (-delta.clip(upper=0)).rolling(window).mean()
			rs = gain / loss.replace(0, np.nan)
			rsi_val = 100 - (100 / (1 + rs))
			return float(rsi_val.iloc[-1])

		close = data['Close']
		volume = data['Volume'] if 'Volume' in data else pd.Series([0] * len(close), index=close.index)
		ret = close.pct_change()
		feature_map = {
			'close': float(close.iloc[-1]),
			'sma_10': float(close.rolling(10).mean().iloc[-1]),
			'sma_20': float(close.rolling(20).mean().iloc[-1]),
			'sma_50': float(close.rolling(50).mean().iloc[-1]),
			'volatility_20': float(ret.rolling(20).std().iloc[-1]),
			'volume_change_10': float(volume.pct_change().rolling(10).mean().iloc[-1]),
			'rsi_14': float(rsi(close, 14)),
		}
		if not feature_list:
			feature_list = list(feature_map.keys())
		features = np.array([
			float(feature_map.get(name, 0.0) or 0.0) for name in feature_list
		])

		try:
			if hasattr(model, 'predict_proba'):
				prob = float(model.predict_proba([features])[0][1])
			else:
				pred = float(model.predict([features])[0])
				prob = max(0.0, min(1.0, 0.5 + pred))
		except Exception:
			return Response({'error': 'Prediction failed.'}, status=500)

		return Response({
			'symbol': symbol,
			'up_probability': round(prob, 4),
			'ai_score': round(prob * 100, 2),
			'recommendation': 'BUY' if prob >= 0.7 else 'HOLD',
		})


class MacroIndicatorView(APIView):
	def get(self, request):
		limit = int(request.query_params.get('limit', 30))
		limit = max(1, min(limit, 365))
		queryset = MacroIndicator.objects.order_by('-date')[:limit]
		return Response(MacroIndicatorSerializer(queryset, many=True).data)


class AlpacaIntradayView(APIView):
	def get(self, request):
		def _safe_number(value: Any, default: float = 0.0) -> float:
			try:
				val = float(value)
				if not math.isfinite(val):
					return default
				return val
			except (TypeError, ValueError):
				return default

		def _clean_json(value: Any) -> Any:
			if value is None:
				return None
			if isinstance(value, dict):
				return {key: _clean_json(val) for key, val in value.items()}
			if isinstance(value, list):
				return [_clean_json(item) for item in value]
			if isinstance(value, tuple):
				return [_clean_json(item) for item in value]
			if isinstance(value, pd.Timestamp):
				try:
					return int(value.timestamp())
				except Exception:
					return None
			if isinstance(value, (np.generic,)):
				try:
					value = value.item()
				except Exception:
					return None
			if isinstance(value, float):
				return value if math.isfinite(value) else 0.0
			return value

		symbol = (request.query_params.get('symbol') or '').strip().upper()
		if not symbol:
			return Response({'error': 'symbol is required'}, status=400)
		alias_map = {
			'APPL': 'AAPL',
			'GOOG': 'GOOGL',
		}
		resolved_symbol = alias_map.get(symbol, symbol)
		minutes = int(request.query_params.get('minutes', 390))
		minutes = max(30, min(minutes, 2000))
		rvol_window = int(request.query_params.get('rvol_window', 20))
		bars = get_intraday_bars(resolved_symbol, minutes=minutes)
		alt_symbol = None
		if bars.empty and '.' not in resolved_symbol:
			alt_symbol = f"{resolved_symbol}.TO"
			bars = get_intraday_bars(alt_symbol, minutes=minutes)
		if bars.empty:
			return Response({
				'error': f"Aucune donnée intraday pour {resolved_symbol}. Alpaca ne couvre pas certains tickers (.TO, crypto, etc.).",
				'symbol': resolved_symbol,
				'bars': [],
				'annotations': [],
				'guidance': [],
				'stats': {},
			}, status=200)

		bars = enrich_bars_with_patterns(bars, rvol_window=rvol_window)
		annotations = build_pattern_annotations(bars)
		clean_annotations = []
		for item in annotations:
			clean_annotations.append({
				'time': int(item.get('time') or 0),
				'text': item.get('text') or '',
				'signal': _safe_number(item.get('signal')),
				'rvol': _safe_number(item.get('rvol')),
			})

		bars_payload = []
		for _, row in bars.iterrows():
			try:
				time_val = int(pd.Timestamp(row['timestamp']).timestamp())
			except Exception:
				continue
			bars_payload.append({
				'time': time_val,
				'open': _safe_number(row.get('open')),
				'high': _safe_number(row.get('high')),
				'low': _safe_number(row.get('low')),
				'close': _safe_number(row.get('close')),
				'volume': _safe_number(row.get('volume')),
				'rvol': _safe_number(row.get('rvol')),
				'pattern_signal': _safe_number(row.get('pattern_signal')),
				'ema20': _safe_number(row.get('ema20')),
				'ema50': _safe_number(row.get('ema50')),
				'rsi14': _safe_number(row.get('rsi14')),
				'patterns': row.get('patterns') or [],
			})

		latest = bars.iloc[-1]
		last_close = _safe_number(latest.get('close'))
		pattern_signal = _safe_number(latest.get('pattern_signal'))
		rvol = _safe_number(latest.get('rvol'))
		volatility = _safe_number(latest.get('volatility'))
		support = _safe_number(bars.tail(30)['low'].min() if 'low' in bars else last_close, default=last_close)
		resistance = _safe_number(bars.tail(30)['high'].max() if 'high' in bars else last_close, default=last_close)
		stop_loss = last_close * (1 - max(volatility * 3, 0.01)) if last_close else 0.0
		base_prob = 0.5 + (pattern_signal * 0.08)
		if rvol >= 2:
			base_prob += 0.05
		probability = min(0.95, max(0.05, base_prob))
		base_target_pct = float(os.getenv('INTRADAY_TARGET_BASE_PCT', '0.02'))
		aggressive_pct = float(os.getenv('INTRADAY_TARGET_AGGRESSIVE_PCT', '0.05'))
		aggressive_high_pct = float(os.getenv('INTRADAY_TARGET_AGGRESSIVE_HIGH_PCT', '0.07'))
		target_pct = base_target_pct
		if probability >= 0.8 and rvol > 5:
			target_pct = aggressive_pct
		if probability >= 0.9 and rvol > 6:
			target_pct = aggressive_high_pct
		profit_target = last_close * (1 + target_pct) if last_close else 0.0

		guidance = []
		if pattern_signal > 0 and rvol >= 2:
			guidance.append(
				f"Pattern haussier + RVOL élevé. Support {support:.2f}$, stop-loss suggéré {stop_loss:.2f}$, target {profit_target:.2f}$."
			)
		elif pattern_signal < 0 and rvol >= 2:
			guidance.append(
				f"Pattern baissier + RVOL élevé. Résistance {resistance:.2f}$, prudence et stop-loss {stop_loss:.2f}$."
			)
		else:
			guidance.append(
				f"RVOL {rvol:.2f} et volatilité {volatility:.4f}. Support {support:.2f}$, résistance {resistance:.2f}$."
			)

		payload = {
			'symbol': alt_symbol or resolved_symbol,
			'bars': bars_payload,
			'annotations': clean_annotations,
			'guidance': [str(item) for item in guidance if item is not None],
			'stats': {
				'last_close': _safe_number(last_close),
				'pattern_signal': _safe_number(pattern_signal),
				'rvol': _safe_number(rvol),
				'volatility': _safe_number(volatility),
				'rsi14': _safe_number(latest.get('rsi14')),
				'ema20': _safe_number(latest.get('ema20')),
				'ema50': _safe_number(latest.get('ema50')),
				'probability': _safe_number(round(float(probability), 4)),
				'support': _safe_number(support, default=0.0),
				'resistance': _safe_number(resistance, default=0.0),
				'suggested_stop': _safe_number(stop_loss),
				'suggested_target': _safe_number(profit_target),
			},
		}
		return Response(_clean_json(payload), status=200)


class MarketScannerView(APIView):
	def get(self, request):
		results = cache.get('market_scanner_results') or []
		if not results:
			watch = SandboxWatchlist.objects.filter(sandbox='AI_PENNY').first()
			if watch and watch.symbols:
				results = [
					{
						'symbol': str(symbol).strip().upper(),
						'change_pct': 0,
						'rvol': 0,
						'patterns': [],
						'score': 0,
						'source': 'watchlist',
					}
					for symbol in watch.symbols
					if str(symbol).strip()
				]
		return Response({'results': results}, status=200)


class PortfolioDashboardView(APIView):
	def _safe_float(self, value: Any) -> float | None:
		try:
			if value is None:
				return None
			val = float(value)
			if not math.isfinite(val):
				return None
			return val
		except (TypeError, ValueError):
			return None

	def _fast_mode(self) -> bool:
		return str(os.getenv('DASHBOARD_FAST_MODE', '1')).strip().lower() in {'1', 'true', 'yes', 'y'}

	def get(self, request):
		return _portfolio_dashboard_get(self, request)

	def _should_enrich(self, request) -> bool:
		flag = request.query_params.get('enrich')
		if flag is not None:
			return str(flag).strip().lower() in {'1', 'true', 'yes', 'y'}
		return True

	def _build_confidence_meter(self) -> dict[str, Any] | None:
		symbol = (os.getenv('CONFIDENCE_SYMBOL') or os.getenv('PAPER_WATCHLIST', 'SPY').split(',')[0]).strip().upper()
		if not symbol:
			return None
		try:
			force_full = str(os.getenv('CONFIDENCE_FORCE_FULL', '')).strip().lower() in {'1', 'true', 'yes', 'y'}
			fast_mode = False if force_full else self._fast_mode()
			fusion = DataFusionEngine(symbol, fast_mode=fast_mode)
			fusion_df = fusion.fuse_all()
			if fusion_df is None or fusion_df.empty:
				return {'symbol': symbol, 'status': 'unavailable'}
			payload = load_or_train_model(fusion_df, model_path=get_model_path('BLUECHIP'))
			if not payload or not payload.get('model'):
				return {'symbol': symbol, 'status': 'unavailable'}
			last_row = fusion_df.tail(1).copy()
			feature_list = payload.get('features') or FEATURE_COLUMNS
			for col in feature_list:
				if col not in last_row.columns:
					last_row[col] = 0.0
			features = last_row[feature_list].fillna(0).values
			try:
				signal = float(payload['model'].predict_proba(features)[0][1])
			except Exception:
				signal = 0.0
			volume_z = self._safe_float(last_row.iloc[0].get('VolumeZ', 0.0))
			vol_regime = self._safe_float(last_row.iloc[0].get('vol_regime', 0.0))
			if volume_z is None:
				volume_z = 0.0
			if vol_regime is None:
				vol_regime = 0.0
			ai_score = round(signal * 100, 2)
			stats = None
			try:
				backtester = AIBacktester(fusion_df, payload, symbol=symbol)
				result = backtester.run_simulation(lookback_days=90)
				stats = {'win_rate': float(result.win_rate), 'sharpe': float(result.sharpe_ratio)}
			except Exception:
				stats = {'win_rate': None, 'sharpe': None}
			min_score = float(os.getenv('CONFIDENCE_AI_SCORE_MIN', '80'))
			min_volume_z = float(os.getenv('CONFIDENCE_VOLUME_Z_MIN', '0.5'))
			max_vol_regime = float(os.getenv('CONFIDENCE_VOL_REGIME_MAX', '1.6'))
			note = None
			recent_closed = list(PaperTrade.objects.filter(status='CLOSED').order_by('-exit_date')[:3])
			if len(recent_closed) == 3 and all(t.outcome == 'LOSS' for t in recent_closed):
				ai_score = max(0.0, ai_score - 10)
				note = 'Le marché a changé de régime (Volatilité haute), réduisez la taille de vos positions.'
			status = 'neutral'
			label = 'Signal en attente'
			if max_vol_regime and vol_regime >= max_vol_regime:
				status = 'red'
				label = 'Volatilité instable'
			elif ai_score >= min_score and volume_z > min_volume_z:
				status = 'green'
				label = 'Signal confirmé'
			elif ai_score >= min_score:
				status = 'orange'
				label = 'Attendre le volume'
			return {
				'symbol': symbol,
				'ai_score': ai_score,
				'volume_z': round(volume_z, 3),
				'vol_regime': round(vol_regime, 3),
				'win_rate': stats.get('win_rate') if stats else None,
				'sharpe': stats.get('sharpe') if stats else None,
				'status': status,
				'label': label,
				'note': note,
				'thresholds': {
					'ai_score_min': min_score,
					'volume_z_min': min_volume_z,
					'vol_regime_max': max_vol_regime,
				},
			}
		except Exception:
			return {'symbol': symbol, 'status': 'unavailable'}

	def _stop_price(self, price: float) -> float | None:
		try:
			return round(float(price) * 0.95, 4) if price else None
		except Exception:
			return None

	def _get_rsi(self, symbol: str) -> float | None:
		try:
			fusion = DataFusionEngine(symbol, fast_mode=self._fast_mode())
			frame = fusion.fuse_all()
			if frame is None or frame.empty:
				if self._fast_mode():
					stock = Stock.objects.filter(symbol__iexact=symbol).first()
					return self._rsi_from_history(stock)
				return None
			last_val = self._safe_float(frame.tail(1).iloc[0].get('RSI14'))
			if last_val is None and self._fast_mode():
				stock = Stock.objects.filter(symbol__iexact=symbol).first()
				return self._rsi_from_history(stock)
			return last_val
		except Exception:
			return None

	def _get_rsi_history(self, symbol: str, window: int = 5) -> list[float]:
		try:
			fusion = DataFusionEngine(symbol, fast_mode=self._fast_mode())
			frame = fusion.fuse_all()
			if frame is None or frame.empty or 'RSI14' not in frame:
				if self._fast_mode():
					stock = Stock.objects.filter(symbol__iexact=symbol).first()
					return self._rsi_history_from_history(stock, window=window)
				return []
			values = [float(v) for v in frame['RSI14'].tail(window).tolist() if v is not None and not pd.isna(v)]
			if not values and self._fast_mode():
				stock = Stock.objects.filter(symbol__iexact=symbol).first()
				return self._rsi_history_from_history(stock, window=window)
			return values
		except Exception:
			return []

	def _rsi_from_history(self, stock: Stock | None, window: int = 14) -> float | None:
		if not stock:
			return None
		try:
			closes = list(
				PriceHistory.objects.filter(stock=stock).order_by('date').values_list('close_price', flat=True)
			)
			if len(closes) < window + 1:
				return None
			series = pd.Series([float(val) for val in closes])
			delta = series.diff()
			gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
			loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
			rs = gain / loss.replace(0, pd.NA)
			rsi = 100 - (100 / (1 + rs))
			last = rsi.iloc[-1]
			if pd.isna(last):
				return None
			return float(last)
		except Exception:
			return None

	def _rsi_history_from_history(self, stock: Stock | None, window: int = 5) -> list[float]:
		if not stock:
			return []
		try:
			closes = list(
				PriceHistory.objects.filter(stock=stock).order_by('date').values_list('close_price', flat=True)
			)
			if len(closes) < 15:
				return []
			series = pd.Series([float(val) for val in closes])
			delta = series.diff()
			gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
			loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
			rs = gain / loss.replace(0, pd.NA)
			rsi = 100 - (100 / (1 + rs))
			values = rsi.tail(window).tolist()
			return [float(v) for v in values if v is not None and not pd.isna(v)]
		except Exception:
			return []

	def _get_ma20(self, symbol: str) -> float | None:
		try:
			fusion = DataFusionEngine(symbol, fast_mode=self._fast_mode())
			frame = fusion.fuse_all()
			if frame is None or frame.empty:
				if self._fast_mode():
					stock = Stock.objects.filter(symbol__iexact=symbol).first()
					return self._ma20_from_history(stock)
				return None
			return self._safe_float(frame.tail(1).iloc[0].get('MA20'))
		except Exception:
			return None

	def _ma20_from_history(self, stock: Stock | None) -> float | None:
		if not stock:
			return None
		try:
			closes = list(
				PriceHistory.objects.filter(stock=stock).order_by('date').values_list('close_price', flat=True)
			)
			if len(closes) < 10:
				return None
			series = pd.Series([float(val) for val in closes])
			ma20 = series.rolling(20, min_periods=10).mean().iloc[-1]
			if pd.isna(ma20):
				return None
			return float(ma20)
		except Exception:
			return None

	def _ai_score(self, symbol: str) -> tuple[float | None, str | None]:
		if self._fast_mode():
			return None, None
		try:
			fusion = DataFusionEngine(symbol, fast_mode=self._fast_mode())
			frame = fusion.fuse_all()
			if frame is None or frame.empty:
				return None, None
			payload = load_or_train_model(frame, model_path=get_model_path('BLUECHIP'))
			if not payload or not payload.get('model'):
				return None, None
			last_row = frame.tail(2).copy()
			feature_list = payload.get('features') or FEATURE_COLUMNS
			for col in feature_list:
				if col not in last_row.columns:
					last_row[col] = 0.0
			features = last_row[feature_list].fillna(0).values
			signal = float(payload['model'].predict_proba(features[-1:])[0][1])
			ai_score = round(signal * 100, 2)
			trend = None
			if len(last_row) >= 2:
				ma20_now = float(last_row.iloc[-1].get('MA20') or 0)
				ma20_prev = float(last_row.iloc[-2].get('MA20') or 0)
				rsi_now = float(last_row.iloc[-1].get('RSI14') or 0)
				trend = 'descending' if ma20_now < ma20_prev or rsi_now < 50 else 'ascending'
			return ai_score, trend
		except Exception:
			return None, None

	def _model_stats(self, symbol: str) -> dict[str, float | None]:
		if self._fast_mode():
			return {'win_rate': None, 'sharpe': None}
		try:
			fusion = DataFusionEngine(symbol, fast_mode=self._fast_mode())
			frame = fusion.fuse_all()
			if frame is None or frame.empty:
				return {'win_rate': None, 'sharpe': None}
			stock = Stock.objects.filter(symbol__iexact=symbol).first()
			is_stable = False
			if stock:
				is_stable = float(stock.latest_price or 0) >= 5 or float(stock.dividend_yield or 0) >= 0.02
			universe = 'BLUECHIP' if is_stable else 'PENNY'
			payload = load_or_train_model(frame, model_path=get_model_path(universe))
			if not payload or not payload.get('model'):
				return {'win_rate': None, 'sharpe': None}
			backtester = AIBacktester(frame, payload, symbol=symbol)
			result = backtester.run_simulation(lookback_days=90)
			return {'win_rate': float(result.win_rate), 'sharpe': float(result.sharpe_ratio)}
		except Exception:
			return {'win_rate': None, 'sharpe': None}

	def _coerce_date(self, value: Any) -> date | None:
		if value is None:
			return None
		try:
			if isinstance(value, pd.Series):
				if value.empty:
					return None
				value = value.iloc[0]
			elif isinstance(value, (list, tuple)):
				if not value:
					return None
				value = value[0]
			parsed = pd.to_datetime(value, errors='coerce')
			if parsed is None or pd.isna(parsed):
				return None
			if hasattr(parsed, 'to_pydatetime'):
				parsed = parsed.to_pydatetime()
			if isinstance(parsed, datetime):
				return parsed.date()
			if isinstance(parsed, date):
				return parsed
		except Exception:
			return None
		return None

	def _earnings_date(self, symbol: str) -> date | None:
		try:
			calendar = yf.Ticker(symbol).calendar
			if calendar is None:
				return None
			if isinstance(calendar, pd.DataFrame):
				if 'Earnings Date' in calendar.index:
					return self._coerce_date(calendar.loc['Earnings Date'][0])
				if 'Earnings Date' in calendar.columns:
					return self._coerce_date(calendar['Earnings Date'].iloc[0])
			if isinstance(calendar, dict):
				return self._coerce_date(calendar.get('Earnings Date'))
		except Exception:
			return None
		return None

	def _earnings_blacklist(self, symbol: str, days: int = 7) -> tuple[bool, date | None]:
		if self._fast_mode():
			return False, None
		try:
			earnings_date = self._earnings_date(symbol)
			if not earnings_date:
				return False, None
			cutoff = (timezone.now() + timedelta(days=days)).date()
			return earnings_date <= cutoff, earnings_date
		except Exception:
			return False, None

	def _sector_relative_strength(self, stock: Stock, days: int = 30) -> dict[str, Any]:
		today = timezone.now().date()
		start = today - timedelta(days=days)
		stock_now = self._safe_float(stock.latest_price)
		if stock_now is None or stock_now <= 0:
			stock_now = self._price_at_or_before(stock, today)
		stock_then = self._price_at_or_before(stock, start)
		stock_ret = None
		if stock_now and stock_then:
			stock_ret = ((stock_now - stock_then) / stock_then) * 100
		sector_returns = []
		if stock.sector:
			peers = Stock.objects.filter(sector=stock.sector).exclude(id=stock.id)[:50]
			for peer in peers:
				peer_now = self._safe_float(peer.latest_price)
				if peer_now is None or peer_now <= 0:
					peer_now = self._price_at_or_before(peer, today)
				peer_then = self._price_at_or_before(peer, start)
				if peer_now and peer_then:
					sector_returns.append(((peer_now - peer_then) / peer_then) * 100)
		sector_median = float(np.median(sector_returns)) if sector_returns else None
		outperform = None
		if stock_ret is not None and sector_median is not None:
			outperform = stock_ret >= sector_median
		return {
			'stock_return_30d': round(stock_ret, 2) if stock_ret is not None else None,
			'sector_median_30d': round(sector_median, 2) if sector_median is not None else None,
			'outperform': outperform,
		}

	def _build_holdings_from_account_transactions(
		self,
		transactions: list[AccountTransaction],
		enrich: bool,
	) -> dict[str, Any]:
		position_map: dict[str, dict[str, Any]] = {}
		for tx in transactions:
			if tx.type == 'DIVIDEND':
				continue
			sign = 1 if tx.type == 'BUY' else -1
			symbol_key = (tx.stock.symbol or '').strip().upper() or str(tx.stock_id)
			entry = position_map.setdefault(
				symbol_key,
				{'stock': tx.stock, 'shares': 0.0, 'buy_qty': 0.0, 'buy_cost': 0.0},
			)
			qty = float(tx.quantity or 0)
			entry['shares'] += qty * sign
			if tx.type == 'BUY':
				entry['buy_qty'] += qty
				entry['buy_cost'] += qty * float(tx.price or 0)

		items = []
		pre_entries = []
		total_cost_value = 0.0
		for payload in position_map.values():
			shares = float(payload['shares'] or 0)
			if shares <= 0:
				continue
			stock = payload['stock']
			buy_qty = float(payload.get('buy_qty') or 0)
			buy_cost = float(payload.get('buy_cost') or 0)
			avg_cost = (buy_cost / buy_qty) if buy_qty else 0.0
			price = stock.latest_price
			if price is None:
				last = PriceHistory.objects.filter(stock=stock).order_by('-date').first()
				price = float(last.close_price) if last else 0.0
			price = float(price or 0)
			effective_price = price if price > 0 else avg_cost
			value = shares * effective_price
			cost_value = avg_cost * shares
			unrealized = value - cost_value
			unrealized_pct = (unrealized / cost_value * 100) if cost_value else 0
			total_cost_value += cost_value
			pre_entries.append({
				'stock': stock,
				'symbol': (stock.symbol or '').strip().upper(),
				'shares': shares,
				'price': price,
				'effective_price': effective_price,
				'avg_cost': avg_cost,
				'value': value,
				'cost_value': cost_value,
				'unrealized': unrealized,
				'unrealized_pct': unrealized_pct,
			})

		max_enrich = int(os.getenv('DASHBOARD_ENRICH_LIMIT', '6'))
		sorted_entries = sorted(pre_entries, key=lambda item: item['value'], reverse=True)
		enriched_symbols = {item['symbol'] for item in sorted_entries[:max_enrich] if item.get('symbol')}

		total_value = 0.0
		stable_value = 0.0
		risky_value = 0.0
		change_1d = 0.0
		change_7d = 0.0

		for entry in sorted_entries:
			stock = entry['stock']
			symbol = entry['symbol']
			shares = entry['shares']
			price = entry['price']
			effective_price = entry['effective_price']
			avg_cost = entry['avg_cost']
			value = entry['value']
			cost_value = entry['cost_value']
			unrealized = entry['unrealized']
			unrealized_pct = entry['unrealized_pct']
			total_value += value

			prev_1d = PriceHistory.objects.filter(stock=stock).order_by('-date')[1:2].first()
			if prev_1d:
				change_1d += (price - float(prev_1d.close_price)) * shares

			prev_7d = PriceHistory.objects.filter(stock=stock).order_by('-date')[7:8].first()
			if prev_7d:
				change_7d += (price - float(prev_7d.close_price)) * shares

			is_stable = effective_price >= 5 or float(stock.dividend_yield or 0) >= 0.02
			if is_stable:
				stable_value += value
			else:
				risky_value += value

			volume_z = None
			rsi = None
			rsi_history = []
			ma20 = None
			stats = {'win_rate': None, 'sharpe': None}
			rel_strength = None
			earnings_blacklisted, earnings_date = False, None
			ai_score, trend = None, None
			exit_strategy = None
			if enrich and symbol in enriched_symbols:
				volume_z = self._get_volume_z(symbol)
				rsi = self._get_rsi(symbol)
				rsi_history = self._get_rsi_history(symbol)
				ma20 = self._get_ma20(symbol)
				stats = self._model_stats(symbol)
				rel_strength = self._sector_relative_strength(stock)
				earnings_blacklisted, earnings_date = self._earnings_blacklist(symbol)
				ai_score, trend = self._ai_score(symbol)
				if ai_score is not None and volume_z is not None and trend == 'descending':
					if volume_z < 0 and ai_score < 65:
						stop_loss = round(effective_price * 0.97, 2)
						exit_strategy = {
							'action': 'VENDRE 50%',
							'instructions': f"Vendre la moitié maintenant. Placer un Stop-Loss à {stop_loss}$ sur le solde pour 15 jours.",
							'reason': 'Divergence Volume/Prix + Baisse du score IA.',
						}

			items.append({
				'ticker': stock.symbol,
				'name': stock.name,
				'sector': stock.sector,
				'dividend_yield': float(stock.dividend_yield or 0),
				'shares': shares,
				'price': effective_price,
				'value': value,
				'avg_cost': round(avg_cost, 4),
				'cost_value': round(cost_value, 2),
				'unrealized_pnl': round(unrealized, 2),
				'unrealized_pnl_pct': round(unrealized_pct, 2),
				'volume_z': round(volume_z, 2) if volume_z is not None else None,
				'ai_score': round(float(ai_score), 2) if ai_score is not None else None,
				'rsi': round(rsi, 2) if rsi is not None else None,
				'rsi_history': [round(val, 2) for val in rsi_history] if rsi_history else [],
				'ma20': round(ma20, 4) if ma20 is not None else None,
				'stop_price': self._stop_price(effective_price),
				'exit_strategy': exit_strategy,
				'model_win_rate': round(float(stats.get('win_rate') or 0), 2) if stats.get('win_rate') is not None else None,
				'model_sharpe': round(float(stats.get('sharpe') or 0), 2) if stats.get('sharpe') is not None else None,
				'relative_strength': rel_strength,
				'earnings_blacklisted': earnings_blacklisted,
				'earnings_date': earnings_date.isoformat() if earnings_date else None,
				'category': 'Stable' if is_stable else 'Risky',
			})

		return {
			'items': items,
			'total_value': total_value,
			'stable_value': stable_value,
			'risky_value': risky_value,
			'total_cost_value': total_cost_value,
			'change_1d': change_1d,
			'change_7d': change_7d,
		}

	def _current_drawdown(self, values: list[float]) -> float:
		if not values:
			return 0.0
		peak = max(values)
		current = values[-1]
		if not peak:
			return 0.0
		return (current - peak) / peak

	def _price_at_or_before(self, stock: Stock, target_date: date) -> float | None:
		row = PriceHistory.objects.filter(stock=stock, date__lte=target_date).order_by('-date').first()
		if not row:
			return None
		return self._safe_float(row.close_price)

	def _get_volume_z(self, symbol: str) -> float | None:
		try:
			fusion = DataFusionEngine(symbol, fast_mode=self._fast_mode())
			frame = fusion.fuse_all()
			if frame is None or frame.empty:
				return None
			return self._safe_float(frame.tail(1).iloc[0].get('VolumeZ'))
		except Exception:
			return None


def _retro_model_payload() -> dict[str, Any] | None:
	model_path = Path(__file__).resolve().parent / 'ml_engine' / 'retro_pattern_success.pkl'
	if not model_path.exists():
		return None
	try:
		payload = joblib.load(model_path)
		if isinstance(payload, dict) and payload.get('model') and payload.get('features'):
			return payload
	except Exception:
		return None
	return None


def _retro_feature_row(symbol: str) -> tuple[dict[str, Any], pd.Series] | None:
	try:
		import pandas_ta as ta
		from .alpaca_data import get_daily_bars

		daily = get_daily_bars(symbol, days=365 * 2)
		if daily is None or daily.empty:
			daily = yf.download(symbol, period='2y', interval='1d')
			if daily is None or daily.empty:
				return None
			daily = daily.rename(columns={
				'Open': 'open',
				'High': 'high',
				'Low': 'low',
				'Close': 'close',
				'Volume': 'volume',
			})
			if isinstance(daily.index, pd.DatetimeIndex):
				daily['timestamp'] = daily.index

		frame = daily.copy()
		if 'timestamp' in frame.columns:
			frame['timestamp'] = pd.to_datetime(frame['timestamp'], errors='coerce', utc=True)
			frame = frame.dropna(subset=['timestamp']).set_index('timestamp')
		for col in ['open', 'high', 'low', 'close', 'volume']:
			if col not in frame.columns:
				frame[col] = pd.NA
		frame = frame.dropna(subset=['open', 'high', 'low', 'close'])
		if frame.empty:
			return None

		frame = enrich_bars_with_patterns(frame)
		frame = _add_candlestick_features(frame)
		if frame.empty:
			return None
		frame['rsi14'] = ta.rsi(frame['close'], length=14)
		frame['ema20'] = ta.ema(frame['close'], length=20)
		frame['ema50'] = ta.ema(frame['close'], length=50)
		frame['atr14'] = ta.atr(frame['high'], frame['low'], frame['close'], length=14)
		frame['momentum10'] = ta.mom(frame['close'], length=10)
		frame['day_of_week'] = pd.to_datetime(frame.index).dayofweek
		frame['hour_of_day'] = pd.to_datetime(frame.index).hour
		frame = frame.fillna(0.0)

		row = frame.iloc[-1]
		fundamentals = _yahoo_fundamentals(symbol)
		parent = CORRELATION_MAP.get(symbol) or os.getenv('PENNY_SNIPER_DEFAULT_PARENT', 'SPY')
		parent_change = _intraday_pct_change(parent, minutes=60) if parent else 0.0
		news_titles = _google_news_titles(symbol, days=2)
		news_sentiment = _finbert_score_from_titles(news_titles)

		return {
			'pattern_signal': float(row.get('pattern_signal') or 0),
			'rvol': float(row.get('rvol') or 0),
			'pattern_doji': bool(row.get('pattern_doji')),
			'pattern_hammer': bool(row.get('pattern_hammer')),
			'pattern_engulfing': bool(row.get('pattern_engulfing')),
			'pattern_morning_star': bool(row.get('pattern_morning_star')),
			'pattern_success_3d': False,
			'rsi14': float(row.get('rsi14') or 0),
			'ema20': float(row.get('ema20') or 0),
			'ema50': float(row.get('ema50') or 0),
			'volatility': float(row.get('volatility') or 0),
			'atr14': float(row.get('atr14') or 0),
			'momentum10': float(row.get('momentum10') or 0),
			'day_of_week': int(row.get('day_of_week') or 0),
			'hour_of_day': int(row.get('hour_of_day') or 0),
			'news_sentiment': float(news_sentiment or 0),
			'parent_change': float(parent_change or 0),
			**fundamentals,
		}, row
	except Exception:
		return None


def _gemini_macro_ok(symbol: str, sector: str, news_titles: list[str]) -> bool:
	api_key = getattr(settings, 'GEMINI_AI_API_KEY', None)
	if not api_key:
		return False
	try:
		from google import genai
		client = genai.Client(api_key=api_key)
		model_name = getattr(settings, 'GEMINI_AI_MODEL', 'models/gemini-2.5-flash')
		news_text = " | ".join(news_titles[:5]) if news_titles else 'n/a'
		prompt = (
			"Tu es un analyste macro. Réponds uniquement par OK ou NO. "
			f"Ticker: {symbol}. Secteur: {sector}. News: {news_text}. "
			"Le contexte macro/secteur est-il sain pour un achat court terme?"
		)
		response = client.models.generate_content(model=model_name, contents=prompt)
		text = (getattr(response, 'text', None) or '').strip().upper()
		return text.startswith('OK') or text.startswith('YES')
	except Exception:
		return False

	def _get_rsi(self, symbol: str) -> float | None:
		try:
			fusion = DataFusionEngine(symbol, fast_mode=self._fast_mode())
			frame = fusion.fuse_all()
			if frame is None or frame.empty:
				if self._fast_mode():
					stock = Stock.objects.filter(symbol__iexact=symbol).first()
					return self._rsi_from_history(stock)
				return None
			last_val = self._safe_float(frame.tail(1).iloc[0].get('RSI14'))
			if last_val is None and self._fast_mode():
				stock = Stock.objects.filter(symbol__iexact=symbol).first()
				return self._rsi_from_history(stock)
			return last_val
		except Exception:
			return None

	def _get_rsi_history(self, symbol: str, window: int = 5) -> list[float]:
		try:
			fusion = DataFusionEngine(symbol, fast_mode=self._fast_mode())
			frame = fusion.fuse_all()
			if frame is None or frame.empty or 'RSI14' not in frame:
				if self._fast_mode():
					stock = Stock.objects.filter(symbol__iexact=symbol).first()
					return self._rsi_history_from_history(stock, window=window)
				return []
			values = [float(v) for v in frame['RSI14'].tail(window).tolist() if v is not None and not pd.isna(v)]
			if not values and self._fast_mode():
				stock = Stock.objects.filter(symbol__iexact=symbol).first()
				return self._rsi_history_from_history(stock, window=window)
			return values
		except Exception:
			return []

	def _rsi_from_history(self, stock: Stock | None, window: int = 14) -> float | None:
		if not stock:
			return None
		try:
			closes = list(
				PriceHistory.objects.filter(stock=stock).order_by('date').values_list('close_price', flat=True)
			)
			if len(closes) < window + 1:
				return None
			series = pd.Series([float(val) for val in closes])
			delta = series.diff()
			gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
			loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
			rs = gain / loss.replace(0, pd.NA)
			rsi = 100 - (100 / (1 + rs))
			last = rsi.iloc[-1]
			if pd.isna(last):
				return None
			return float(last)
		except Exception:
			return None

	def _rsi_history_from_history(self, stock: Stock | None, window: int = 5) -> list[float]:
		if not stock:
			return []
		try:
			closes = list(
				PriceHistory.objects.filter(stock=stock).order_by('date').values_list('close_price', flat=True)
			)
			if len(closes) < 15:
				return []
			series = pd.Series([float(val) for val in closes])
			delta = series.diff()
			gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
			loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
			rs = gain / loss.replace(0, pd.NA)
			rsi = 100 - (100 / (1 + rs))
			values = rsi.tail(window).tolist()
			return [float(v) for v in values if v is not None and not pd.isna(v)]
		except Exception:
			return []

	def _get_ma20(self, symbol: str) -> float | None:
		try:
			fusion = DataFusionEngine(symbol, fast_mode=self._fast_mode())
			frame = fusion.fuse_all()
			if frame is None or frame.empty:
				if self._fast_mode():
					stock = Stock.objects.filter(symbol__iexact=symbol).first()
					return self._ma20_from_history(stock)
				return None
			return self._safe_float(frame.tail(1).iloc[0].get('MA20'))
		except Exception:
			return None

	def _ma20_from_history(self, stock: Stock | None) -> float | None:
		if not stock:
			return None
		try:
			closes = list(
				PriceHistory.objects.filter(stock=stock).order_by('date').values_list('close_price', flat=True)
			)
			if len(closes) < 10:
				return None
			series = pd.Series([float(val) for val in closes])
			ma20 = series.rolling(20, min_periods=10).mean().iloc[-1]
			if pd.isna(ma20):
				return None
			return float(ma20)
		except Exception:
			return None

	def _ai_score(self, symbol: str) -> tuple[float | None, str | None]:
		if self._fast_mode():
			return None, None
		try:
			fusion = DataFusionEngine(symbol, fast_mode=self._fast_mode())
			frame = fusion.fuse_all()
			if frame is None or frame.empty:
				return None, None
			payload = load_or_train_model(frame, model_path=get_model_path('BLUECHIP'))
			if not payload or not payload.get('model'):
				return None, None
			last_row = frame.tail(2).copy()
			feature_list = payload.get('features') or FEATURE_COLUMNS
			for col in feature_list:
				if col not in last_row.columns:
					last_row[col] = 0.0
			features = last_row[feature_list].fillna(0).values
			signal = float(payload['model'].predict_proba(features[-1:])[0][1])
			ai_score = round(signal * 100, 2)
			trend = None
			if len(last_row) >= 2:
				ma20_now = float(last_row.iloc[-1].get('MA20') or 0)
				ma20_prev = float(last_row.iloc[-2].get('MA20') or 0)
				rsi_now = float(last_row.iloc[-1].get('RSI14') or 0)
				trend = 'descending' if ma20_now < ma20_prev or rsi_now < 50 else 'ascending'
			return ai_score, trend
		except Exception:
			return None, None

	def _stop_price(self, price: float) -> float | None:
		try:
			return round(float(price) * 0.95, 4) if price else None
		except Exception:
			return None

	def _model_stats(self, symbol: str) -> dict[str, float | None]:
		if self._fast_mode():
			return {'win_rate': None, 'sharpe': None}
		try:
			fusion = DataFusionEngine(symbol, fast_mode=self._fast_mode())
			frame = fusion.fuse_all()
			if frame is None or frame.empty:
				return {'win_rate': None, 'sharpe': None}
			stock = Stock.objects.filter(symbol__iexact=symbol).first()
			is_stable = False
			if stock:
				is_stable = float(stock.latest_price or 0) >= 5 or float(stock.dividend_yield or 0) >= 0.02
			universe = 'BLUECHIP' if is_stable else 'PENNY'
			payload = load_or_train_model(frame, model_path=get_model_path(universe))
			if not payload or not payload.get('model'):
				return {'win_rate': None, 'sharpe': None}
			backtester = AIBacktester(frame, payload, symbol=symbol)
			result = backtester.run_simulation(lookback_days=90)
			return {'win_rate': float(result.win_rate), 'sharpe': float(result.sharpe_ratio)}
		except Exception:
			return {'win_rate': None, 'sharpe': None}

	def _coerce_date(self, value: Any) -> date | None:
		if value is None:
			return None
		try:
			if isinstance(value, pd.Series):
				if value.empty:
					return None
				value = value.iloc[0]
			elif isinstance(value, (list, tuple)):
				if not value:
					return None
				value = value[0]
			parsed = pd.to_datetime(value, errors='coerce')
			if parsed is None or pd.isna(parsed):
				return None
			if hasattr(parsed, 'to_pydatetime'):
				parsed = parsed.to_pydatetime()
			if isinstance(parsed, datetime):
				return parsed.date()
			if isinstance(parsed, date):
				return parsed
		except Exception:
			return None
		return None

	def _earnings_date(self, symbol: str) -> date | None:
		try:
			calendar = yf.Ticker(symbol).calendar
			if calendar is None:
				return None
			if isinstance(calendar, pd.DataFrame):
				if 'Earnings Date' in calendar.index:
					return self._coerce_date(calendar.loc['Earnings Date'][0])
				if 'Earnings Date' in calendar.columns:
					return self._coerce_date(calendar['Earnings Date'].iloc[0])
			if isinstance(calendar, dict):
				return self._coerce_date(calendar.get('Earnings Date'))
		except Exception:
			return None
		return None

	def _earnings_blacklist(self, symbol: str, days: int = 7) -> tuple[bool, date | None]:
		if self._fast_mode():
			return False, None
		try:
			earnings_date = self._earnings_date(symbol)
			if not earnings_date:
				return False, None
			cutoff = (timezone.now() + timedelta(days=days)).date()
			return earnings_date <= cutoff, earnings_date
		except Exception:
			return False, None

	def _sector_relative_strength(self, stock: Stock, days: int = 30) -> dict[str, Any]:
		today = timezone.now().date()
		start = today - timedelta(days=days)
		stock_now = self._safe_float(stock.latest_price)
		if stock_now is None or stock_now <= 0:
			stock_now = self._price_at_or_before(stock, today)
		stock_then = self._price_at_or_before(stock, start)
		stock_ret = None
		if stock_now and stock_then:
			stock_ret = ((stock_now - stock_then) / stock_then) * 100
		sector_returns = []
		if stock.sector:
			peers = Stock.objects.filter(sector=stock.sector).exclude(id=stock.id)[:50]
			for peer in peers:
				peer_now = self._safe_float(peer.latest_price)
				if peer_now is None or peer_now <= 0:
					peer_now = self._price_at_or_before(peer, today)
				peer_then = self._price_at_or_before(peer, start)
				if peer_now and peer_then:
					sector_returns.append(((peer_now - peer_then) / peer_then) * 100)
		sector_median = float(np.median(sector_returns)) if sector_returns else None
		outperform = None
		if stock_ret is not None and sector_median is not None:
			outperform = stock_ret >= sector_median
		return {
			'stock_return_30d': round(stock_ret, 2) if stock_ret is not None else None,
			'sector_median_30d': round(sector_median, 2) if sector_median is not None else None,
			'outperform': outperform,
		}
	def _build_confidence_meter(self) -> dict[str, Any] | None:
		symbol = (os.getenv('CONFIDENCE_SYMBOL') or os.getenv('PAPER_WATCHLIST', 'SPY').split(',')[0]).strip().upper()
		if not symbol:
			return None
		try:
			force_full = str(os.getenv('CONFIDENCE_FORCE_FULL', '')).strip().lower() in {'1', 'true', 'yes', 'y'}
			fast_mode = False if force_full else self._fast_mode()
			fusion = DataFusionEngine(symbol, fast_mode=fast_mode)
			fusion_df = fusion.fuse_all()
			if fusion_df is None or fusion_df.empty:
				return {'symbol': symbol, 'status': 'unavailable'}
			payload = load_or_train_model(fusion_df, model_path=get_model_path('BLUECHIP'))
			if not payload or not payload.get('model'):
				return {'symbol': symbol, 'status': 'unavailable'}
			last_row = fusion_df.tail(1).copy()
			feature_list = payload.get('features') or FEATURE_COLUMNS
			for col in feature_list:
				if col not in last_row.columns:
					last_row[col] = 0.0
			features = last_row[feature_list].fillna(0).values
			try:
				signal = float(payload['model'].predict_proba(features)[0][1])
			except Exception:
				signal = 0.0
			volume_z = self._safe_float(last_row.iloc[0].get('VolumeZ', 0.0))
			vol_regime = self._safe_float(last_row.iloc[0].get('vol_regime', 0.0))
			if volume_z is None:
				volume_z = 0.0
			if vol_regime is None:
				vol_regime = 0.0
			ai_score = round(signal * 100, 2)
			stats = None
			try:
				backtester = AIBacktester(fusion_df, payload, symbol=symbol)
				result = backtester.run_simulation(lookback_days=90)
				stats = {'win_rate': float(result.win_rate), 'sharpe': float(result.sharpe_ratio)}
			except Exception:
				stats = {'win_rate': None, 'sharpe': None}
			min_score = float(os.getenv('CONFIDENCE_AI_SCORE_MIN', '80'))
			min_volume_z = float(os.getenv('CONFIDENCE_VOLUME_Z_MIN', '0.5'))
			max_vol_regime = float(os.getenv('CONFIDENCE_VOL_REGIME_MAX', '1.6'))
			note = None
			recent_closed = list(PaperTrade.objects.filter(status='CLOSED').order_by('-exit_date')[:3])
			if len(recent_closed) == 3 and all(t.outcome == 'LOSS' for t in recent_closed):
				ai_score = max(0.0, ai_score - 10)
				note = 'Le marché a changé de régime (Volatilité haute), réduisez la taille de vos positions.'
			status = 'neutral'
			label = 'Signal en attente'
			if max_vol_regime and vol_regime >= max_vol_regime:
				status = 'red'
				label = 'Volatilité instable'
			elif ai_score >= min_score and volume_z > min_volume_z:
				status = 'green'
				label = 'Signal confirmé'
			elif ai_score >= min_score:
				status = 'orange'
				label = 'Attendre le volume'
			return {
				'symbol': symbol,
				'ai_score': ai_score,
				'volume_z': round(volume_z, 3),
				'vol_regime': round(vol_regime, 3),
				'win_rate': stats.get('win_rate') if stats else None,
				'sharpe': stats.get('sharpe') if stats else None,
				'status': status,
				'label': label,
				'note': note,
				'thresholds': {
					'ai_score_min': min_score,
					'volume_z_min': min_volume_z,
					'vol_regime_max': max_vol_regime,
				},
			}
		except Exception:
			return {'symbol': symbol, 'status': 'unavailable'}

	def _should_enrich(self, request) -> bool:
		flag = request.query_params.get('enrich')
		if flag is not None:
			return str(flag).strip().lower() in {'1', 'true', 'yes', 'y'}
		return True

	def _build_holdings_from_account_transactions(
		self,
		transactions: list[AccountTransaction],
		enrich: bool,
	) -> dict[str, Any]:
		position_map: dict[str, dict[str, Any]] = {}
		for tx in transactions:
			if tx.type == 'DIVIDEND':
				continue
			sign = 1 if tx.type == 'BUY' else -1
			symbol_key = (tx.stock.symbol or '').strip().upper() or str(tx.stock_id)
			entry = position_map.setdefault(
				symbol_key,
				{'stock': tx.stock, 'shares': 0.0, 'buy_qty': 0.0, 'buy_cost': 0.0},
			)
			qty = float(tx.quantity or 0)
			entry['shares'] += qty * sign
			if tx.type == 'BUY':
				entry['buy_qty'] += qty
				entry['buy_cost'] += qty * float(tx.price or 0)

		items = []
		pre_entries = []
		total_cost_value = 0.0
		for payload in position_map.values():
			shares = float(payload['shares'] or 0)
			if shares <= 0:
				continue
			stock = payload['stock']
			buy_qty = float(payload.get('buy_qty') or 0)
			buy_cost = float(payload.get('buy_cost') or 0)
			avg_cost = (buy_cost / buy_qty) if buy_qty else 0.0
			price = stock.latest_price
			if price is None:
				last = PriceHistory.objects.filter(stock=stock).order_by('-date').first()
				price = float(last.close_price) if last else 0.0
			price = float(price or 0)
			effective_price = price if price > 0 else avg_cost
			value = shares * effective_price
			cost_value = avg_cost * shares
			unrealized = value - cost_value
			unrealized_pct = (unrealized / cost_value * 100) if cost_value else 0
			total_cost_value += cost_value
			pre_entries.append({
				'stock': stock,
				'symbol': (stock.symbol or '').strip().upper(),
				'shares': shares,
				'price': price,
				'effective_price': effective_price,
				'avg_cost': avg_cost,
				'value': value,
				'cost_value': cost_value,
				'unrealized': unrealized,
				'unrealized_pct': unrealized_pct,
			})

		max_enrich = int(os.getenv('DASHBOARD_ENRICH_LIMIT', '6'))
		sorted_entries = sorted(pre_entries, key=lambda item: item['value'], reverse=True)
		enriched_symbols = {item['symbol'] for item in sorted_entries[:max_enrich] if item.get('symbol')}

		total_value = 0.0
		stable_value = 0.0
		risky_value = 0.0
		change_1d = 0.0
		change_7d = 0.0

		for entry in sorted_entries:
			stock = entry['stock']
			symbol = entry['symbol']
			shares = entry['shares']
			price = entry['price']
			effective_price = entry['effective_price']
			avg_cost = entry['avg_cost']
			value = entry['value']
			cost_value = entry['cost_value']
			unrealized = entry['unrealized']
			unrealized_pct = entry['unrealized_pct']
			total_value += value

			prev_1d = PriceHistory.objects.filter(stock=stock).order_by('-date')[1:2].first()
			if prev_1d:
				change_1d += (price - float(prev_1d.close_price)) * shares

			prev_7d = PriceHistory.objects.filter(stock=stock).order_by('-date')[7:8].first()
			if prev_7d:
				change_7d += (price - float(prev_7d.close_price)) * shares

			is_stable = effective_price >= 5 or float(stock.dividend_yield or 0) >= 0.02
			if is_stable:
				stable_value += value
			else:
				risky_value += value

			volume_z = None
			rsi = None
			rsi_history = []
			ma20 = None
			stats = {'win_rate': None, 'sharpe': None}
			rel_strength = None
			earnings_blacklisted, earnings_date = False, None
			ai_score, trend = None, None
			exit_strategy = None
			if enrich and symbol in enriched_symbols:
				volume_z = self._get_volume_z(symbol)
				rsi = self._get_rsi(symbol)
				rsi_history = self._get_rsi_history(symbol)
				ma20 = self._get_ma20(symbol)
				stats = self._model_stats(symbol)
				rel_strength = self._sector_relative_strength(stock)
				earnings_blacklisted, earnings_date = self._earnings_blacklist(symbol)
				ai_score, trend = self._ai_score(symbol)
				if ai_score is not None and volume_z is not None and trend == 'descending':
					if volume_z < 0 and ai_score < 65:
						stop_loss = round(effective_price * 0.97, 2)
						exit_strategy = {
							'action': 'VENDRE 50%',
							'instructions': f"Vendre la moitié maintenant. Placer un Stop-Loss à {stop_loss}$ sur le solde pour 15 jours.",
							'reason': 'Divergence Volume/Prix + Baisse du score IA.',
						}

			items.append({
				'ticker': stock.symbol,
				'name': stock.name,
				'sector': stock.sector,
				'dividend_yield': float(stock.dividend_yield or 0),
				'shares': shares,
				'price': effective_price,
				'value': value,
				'avg_cost': round(avg_cost, 4),
				'cost_value': round(cost_value, 2),
				'unrealized_pnl': round(unrealized, 2),
				'unrealized_pnl_pct': round(unrealized_pct, 2),
				'volume_z': round(volume_z, 2) if volume_z is not None else None,
				'ai_score': round(float(ai_score), 2) if ai_score is not None else None,
				'rsi': round(rsi, 2) if rsi is not None else None,
				'rsi_history': [round(val, 2) for val in rsi_history] if rsi_history else [],
				'ma20': round(ma20, 4) if ma20 is not None else None,
				'stop_price': self._stop_price(effective_price),
				'exit_strategy': exit_strategy,
				'model_win_rate': round(float(stats.get('win_rate') or 0), 2) if stats.get('win_rate') is not None else None,
				'model_sharpe': round(float(stats.get('sharpe') or 0), 2) if stats.get('sharpe') is not None else None,
				'relative_strength': rel_strength,
				'earnings_blacklisted': earnings_blacklisted,
				'earnings_date': earnings_date.isoformat() if earnings_date else None,
				'category': 'Stable' if is_stable else 'Risky',
			})

		return {
			'items': items,
			'total_value': total_value,
			'stable_value': stable_value,
			'risky_value': risky_value,
			'total_cost_value': total_cost_value,
			'change_1d': change_1d,
			'change_7d': change_7d,
		}


def _portfolio_dashboard_get(self, request):
		portfolio_id = request.query_params.get('portfolio_id')
		enrich = self._should_enrich(request)
		portfolio = None
		if portfolio_id:
			portfolio = Portfolio.objects.filter(id=portfolio_id).first()
		if not portfolio:
			portfolio = Portfolio.objects.first()
		if not portfolio:
			try:
				transactions = AccountTransaction.objects.select_related('stock').all()
			except OperationalError:
				transactions = []
			if not transactions:
				return Response({
					'portfolio': None,
					'total_balance': 0,
					'change_24h': 0,
					'change_24h_pct': 0,
					'change_7d': 0,
					'change_7d_pct': 0,
					'current_drawdown': 0,
					'allocation': {
						'stable_pct': 0,
						'risky_pct': 0,
						'stable_value': 0,
						'risky_value': 0,
					},
					'holdings': [],
					'archives': [],
					'chart': [],
					'confidence_meter': self._build_confidence_meter(),
				}, status=200)

			fallback = self._build_holdings_from_account_transactions(list(transactions), enrich)
			items = fallback['items']
			total_value = fallback['total_value']
			stable_value = fallback['stable_value']
			risky_value = fallback['risky_value']
			change_1d = fallback['change_1d']
			change_7d = fallback['change_7d']
			total_cost_value = float(fallback.get('total_cost_value') or 0)
			total_return = total_value - total_cost_value
			total_return_pct = (total_return / total_cost_value * 100) if total_cost_value else 0.0

			allocation_pct = (stable_value / total_value * 100) if total_value else 0
			change_1d_pct = (change_1d / (total_value - change_1d) * 100) if total_value else 0
			change_7d_pct = (change_7d / (total_value - change_7d) * 100) if total_value else 0

			return Response({
				'portfolio': None,
				'total_balance': round(total_value, 2),
				'total_return': round(total_return, 2),
				'total_return_pct': round(total_return_pct, 2),
				'change_24h': round(change_1d, 2),
				'change_24h_pct': round(change_1d_pct, 2),
				'change_7d': round(change_7d, 2),
				'change_7d_pct': round(change_7d_pct, 2),
				'current_drawdown': 0,
				'allocation': {
					'stable_pct': round(allocation_pct, 2),
					'risky_pct': round(100 - allocation_pct, 2),
					'stable_value': round(stable_value, 2),
					'risky_value': round(risky_value, 2),
				},
				'holdings': items,
				'archives': [],
				'chart': [],
				'confidence_meter': self._build_confidence_meter(),
			}, status=200)

		holdings = PortfolioHolding.objects.select_related('stock').filter(portfolio=portfolio)
		try:
			account_transactions = AccountTransaction.objects.select_related('stock').all()
		except OperationalError:
			account_transactions = []
		portfolio_transactions = Transaction.objects.select_related('stock').filter(portfolio=portfolio)

		cost_map = {}
		for tx in account_transactions:
			if tx.type == 'DIVIDEND':
				continue
			sign = 1 if tx.type == 'BUY' else -1
			symbol_key = (tx.stock.symbol or '').strip().upper() or str(tx.stock_id)
			entry = cost_map.setdefault(
				symbol_key,
				{'shares': 0.0, 'buy_qty': 0.0, 'buy_cost': 0.0},
			)
			qty = float(tx.quantity or 0)
			entry['shares'] += qty * sign
			if tx.type == 'BUY':
				entry['buy_qty'] += qty
				entry['buy_cost'] += qty * float(tx.price or 0)

		for tx in portfolio_transactions:
			sign = 1 if tx.transaction_type == 'BUY' else -1
			symbol_key = (tx.stock.symbol or '').strip().upper() or str(tx.stock_id)
			entry = cost_map.setdefault(
				symbol_key,
				{'shares': 0.0, 'buy_qty': 0.0, 'buy_cost': 0.0},
			)
			qty = float(tx.shares or 0)
			entry['shares'] += qty * sign
			if tx.transaction_type == 'BUY':
				entry['buy_qty'] += qty
				entry['buy_cost'] += qty * float(tx.price_per_share or 0)
		items = []
		total_value = 0.0
		stable_value = 0.0
		risky_value = 0.0
		change_1d = 0.0
		change_7d = 0.0

		pre_entries = []
		for holding in holdings:
			stock = holding.stock
			price = stock.latest_price
			if price is None:
				last = PriceHistory.objects.filter(stock=stock).order_by('-date').first()
				price = float(last.close_price) if last else 0.0
			price = float(price or 0)
			symbol_key = (stock.symbol or '').strip().upper() or str(stock.id)
			cost_data = cost_map.get(symbol_key, {})
			if not cost_data.get('buy_qty'):
				buy_qty = 0.0
				buy_cost = 0.0
				try:
					fallback_account = AccountTransaction.objects.select_related('stock').filter(
						stock__symbol__iexact=stock.symbol
					)
				except OperationalError:
					fallback_account = []
				for tx in fallback_account:
					if tx.type != 'BUY':
						continue
					qty = float(tx.quantity or 0)
					buy_qty += qty
					buy_cost += qty * float(tx.price or 0)
				fallback_portfolio = portfolio_transactions.filter(stock__symbol__iexact=stock.symbol)
				for tx in fallback_portfolio:
					if tx.transaction_type != 'BUY':
						continue
					qty = float(tx.shares or 0)
					buy_qty += qty
					buy_cost += qty * float(tx.price_per_share or 0)
				cost_data = {'buy_qty': buy_qty, 'buy_cost': buy_cost}
			buy_qty = float(cost_data.get('buy_qty') or 0)
			buy_cost = float(cost_data.get('buy_cost') or 0)
			avg_cost = (buy_cost / buy_qty) if buy_qty else 0.0
			effective_price = price if price > 0 else avg_cost
			value = float(holding.shares or 0) * effective_price
			cost_value = avg_cost * float(holding.shares or 0)
			unrealized = value - cost_value
			unrealized_pct = (unrealized / cost_value * 100) if cost_value else 0
			pre_entries.append({
				'stock': stock,
				'symbol': (stock.symbol or '').strip().upper(),
				'shares': float(holding.shares or 0),
				'price': price,
				'effective_price': effective_price,
				'avg_cost': avg_cost,
				'value': value,
				'cost_value': cost_value,
				'unrealized': unrealized,
				'unrealized_pct': unrealized_pct,
			})

		max_enrich = int(os.getenv('DASHBOARD_ENRICH_LIMIT', '6'))
		sorted_entries = sorted(pre_entries, key=lambda item: item['value'], reverse=True)
		enriched_symbols = {item['symbol'] for item in sorted_entries[:max_enrich] if item.get('symbol')}

		for entry in sorted_entries:
			stock = entry['stock']
			symbol = entry['symbol']
			shares = entry['shares']
			price = entry['price']
			effective_price = entry['effective_price']
			avg_cost = entry['avg_cost']
			value = entry['value']
			cost_value = entry['cost_value']
			unrealized = entry['unrealized']
			unrealized_pct = entry['unrealized_pct']
			total_value += value

			prev_1d = PriceHistory.objects.filter(stock=stock).order_by('-date')[1:2].first()
			if prev_1d:
				change_1d += (price - float(prev_1d.close_price)) * shares

			prev_7d = PriceHistory.objects.filter(stock=stock).order_by('-date')[7:8].first()
			if prev_7d:
				change_7d += (price - float(prev_7d.close_price)) * shares

			is_stable = effective_price >= 5 or float(stock.dividend_yield or 0) >= 0.02
			if is_stable:
				stable_value += value
			else:
				risky_value += value

			volume_z = None
			rsi = None
			rsi_history = []
			ma20 = None
			stats = {'win_rate': None, 'sharpe': None}
			rel_strength = None
			earnings_blacklisted, earnings_date = False, None
			if enrich and symbol in enriched_symbols:
				volume_z = self._get_volume_z(symbol)
				rsi = self._get_rsi(symbol)
				rsi_history = self._get_rsi_history(symbol)
				ma20 = self._get_ma20(symbol)
				stats = self._model_stats(symbol)
				rel_strength = self._sector_relative_strength(stock)
				earnings_blacklisted, earnings_date = self._earnings_blacklist(symbol)

			items.append({
				'ticker': stock.symbol,
				'name': stock.name,
				'sector': stock.sector,
				'dividend_yield': float(stock.dividend_yield or 0),
				'shares': shares,
				'price': effective_price,
				'value': value,
				'avg_cost': round(avg_cost, 4),
				'cost_value': round(cost_value, 2),
				'unrealized_pnl': round(unrealized, 2),
				'unrealized_pnl_pct': round(unrealized_pct, 2),
				'volume_z': round(volume_z, 2) if volume_z is not None else None,
				'rsi': round(rsi, 2) if rsi is not None else None,
				'rsi_history': [round(val, 2) for val in rsi_history] if rsi_history else [],
				'ma20': round(ma20, 4) if ma20 is not None else None,
				'stop_price': self._stop_price(effective_price),
				'model_win_rate': round(float(stats.get('win_rate') or 0), 2) if stats.get('win_rate') is not None else None,
				'model_sharpe': round(float(stats.get('sharpe') or 0), 2) if stats.get('sharpe') is not None else None,
				'relative_strength': rel_strength,
				'earnings_blacklisted': earnings_blacklisted,
				'earnings_date': earnings_date.isoformat() if earnings_date else None,
				'category': 'Stable' if is_stable else 'Risky',
			})

		if not items and account_transactions:
			fallback = self._build_holdings_from_account_transactions(list(account_transactions), enrich)
			items = fallback['items']
			total_value = fallback['total_value']
			stable_value = fallback['stable_value']
			risky_value = fallback['risky_value']
			change_1d = fallback['change_1d']
			change_7d = fallback['change_7d']
			total_cost_value = float(fallback.get('total_cost_value') or 0)
		else:
			total_cost_value = sum(float(item.get('cost_value') or 0) for item in items)

		archived = []
		if items:
			kept = []
			for item in items:
				win_rate = item.get('model_win_rate')
				if win_rate is not None and float(win_rate) < 45:
					archived.append(item)
					value = float(item.get('value') or 0)
					total_value = max(0.0, total_value - value)
					if item.get('category') == 'Stable':
						stable_value = max(0.0, stable_value - value)
					else:
						risky_value = max(0.0, risky_value - value)
				else:
					kept.append(item)
			items = kept

		allocation_pct = (stable_value / total_value * 100) if total_value else 0
		change_1d_pct = (change_1d / (total_value - change_1d) * 100) if total_value else 0
		change_7d_pct = (change_7d / (total_value - change_7d) * 100) if total_value else 0
		total_return = total_value - total_cost_value
		total_return_pct = (total_return / total_cost_value * 100) if total_cost_value else 0.0

		snapshots = DripSnapshot.objects.filter(portfolio=portfolio).order_by('-as_of')[:365]
		if snapshots:
			chart = [
				{
					'date': s.as_of.isoformat(),
					'value': float(s.capital or 0),
				}
				for s in reversed(list(snapshots))
			]
		else:
			chart = [
				{'date': f'D-{i}', 'value': round(total_value * (0.98 + i * 0.002), 2)}
				for i in range(12)
			]

		chart_values = [float(point.get('value') or 0) for point in chart if point.get('value') is not None]
		current_drawdown = self._current_drawdown(chart_values)

		return Response({
			'portfolio': {'id': portfolio.id, 'name': portfolio.name},
			'total_balance': round(total_value, 2),
			'total_return': round(total_return, 2),
			'total_return_pct': round(total_return_pct, 2),
			'change_24h': round(change_1d, 2),
			'change_24h_pct': round(change_1d_pct, 2),
			'change_7d': round(change_7d, 2),
			'change_7d_pct': round(change_7d_pct, 2),
			'current_drawdown': round(float(current_drawdown) * 100, 2),
			'allocation': {
				'stable_pct': round(allocation_pct, 2),
				'risky_pct': round(100 - allocation_pct, 2),
				'stable_value': round(stable_value, 2),
				'risky_value': round(risky_value, 2),
			},
			'holdings': items,
			'archives': archived,
			'chart': chart,
			'confidence_meter': self._build_confidence_meter(),
		})


class AccountDashboardView(APIView):
	def _fast_mode(self) -> bool:
		return str(os.getenv('DASHBOARD_FAST_MODE', '1')).strip().lower() in {'1', 'true', 'yes', 'y'}

	def _rsi_from_history(self, stock: Stock | None, window: int = 14) -> float | None:
		if not stock:
			return None
		try:
			closes = list(
				PriceHistory.objects.filter(stock=stock).order_by('date').values_list('close_price', flat=True)
			)
			if len(closes) < window + 1:
				return None
			series = pd.Series([float(val) for val in closes])
			delta = series.diff()
			gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
			loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
			rs = gain / loss.replace(0, pd.NA)
			rsi = 100 - (100 / (1 + rs))
			last = rsi.iloc[-1]
			if pd.isna(last):
				return None
			return float(last)
		except Exception:
			return None

	def _price_at_or_before(self, stock: Stock, target_date: date) -> float | None:
		row = PriceHistory.objects.filter(stock=stock, date__lte=target_date).order_by('-date').first()
		if not row:
			return None
		return float(row.close_price)

	def _current_price(self, stock: Stock) -> float:
		if stock.latest_price is not None:
			return float(stock.latest_price)
		last = PriceHistory.objects.filter(stock=stock).order_by('-date').first()
		return float(last.close_price) if last else 0.0

	def _get_rsi(self, symbol: str) -> float | None:
		try:
			fusion = DataFusionEngine(symbol, fast_mode=self._fast_mode())
			frame = fusion.fuse_all()
			if frame is None or frame.empty:
				if self._fast_mode():
					stock = Stock.objects.filter(symbol__iexact=symbol).first()
					return self._rsi_from_history(stock)
				return None
			last_val = frame.tail(1).iloc[0].get('RSI14')
			try:
				last_val = float(last_val)
			except (TypeError, ValueError):
				last_val = None
			if last_val is None and self._fast_mode():
				stock = Stock.objects.filter(symbol__iexact=symbol).first()
				return self._rsi_from_history(stock)
			return last_val
		except Exception:
			return None

	def _get_volume_z(self, symbol: str) -> float | None:
		try:
			fusion = DataFusionEngine(symbol, fast_mode=self._fast_mode())
			frame = fusion.fuse_all()
			if frame is None or frame.empty:
				return None
			return float(frame.tail(1).iloc[0].get('VolumeZ'))
		except Exception:
			return None

	def _get_ma20(self, symbol: str) -> float | None:
		try:
			fusion = DataFusionEngine(symbol, fast_mode=self._fast_mode())
			frame = fusion.fuse_all()
			if frame is None or frame.empty:
				if self._fast_mode():
					stock = Stock.objects.filter(symbol__iexact=symbol).first()
					return self._ma20_from_history(stock)
				return None
			return float(frame.tail(1).iloc[0].get('MA20'))
		except Exception:
			return None

	def _ma20_from_history(self, stock: Stock | None) -> float | None:
		if not stock:
			return None
		try:
			closes = list(
				PriceHistory.objects.filter(stock=stock).order_by('date').values_list('close_price', flat=True)
			)
			if len(closes) < 10:
				return None
			series = pd.Series([float(val) for val in closes])
			ma20 = series.rolling(20, min_periods=10).mean().iloc[-1]
			if pd.isna(ma20):
				return None
			return float(ma20)
		except Exception:
			return None

	def _ai_score(self, symbol: str) -> tuple[float | None, str | None]:
		if self._fast_mode():
			return None, None
		try:
			fusion = DataFusionEngine(symbol, fast_mode=self._fast_mode())
			frame = fusion.fuse_all()
			if frame is None or frame.empty:
				return None, None
			payload = load_or_train_model(frame, model_path=get_model_path('BLUECHIP'))
			if not payload or not payload.get('model'):
				return None, None
			last_row = frame.tail(2).copy()
			feature_list = payload.get('features') or FEATURE_COLUMNS
			for col in feature_list:
				if col not in last_row.columns:
					last_row[col] = 0.0
			features = last_row[feature_list].fillna(0).values
			signal = float(payload['model'].predict_proba(features[-1:])[0][1])
			ai_score = round(signal * 100, 2)
			trend = None
			if len(last_row) >= 2:
				ma20_now = float(last_row.iloc[-1].get('MA20') or 0)
				ma20_prev = float(last_row.iloc[-2].get('MA20') or 0)
				rsi_now = float(last_row.iloc[-1].get('RSI14') or 0)
				trend = 'descending' if ma20_now < ma20_prev or rsi_now < 50 else 'ascending'
			return ai_score, trend
		except Exception:
			return None, None

	def _stop_price(self, price: float) -> float | None:
		try:
			return round(float(price) * 0.95, 4) if price else None
		except Exception:
			return None

	def _model_stats(self, symbol: str) -> dict[str, float | None]:
		if self._fast_mode():
			return {'win_rate': None, 'sharpe': None}
		try:
			fusion = DataFusionEngine(symbol, fast_mode=self._fast_mode())
			frame = fusion.fuse_all()
			if frame is None or frame.empty:
				return {'win_rate': None, 'sharpe': None}
			stock = Stock.objects.filter(symbol__iexact=symbol).first()
			is_stable = False
			if stock:
				is_stable = float(stock.latest_price or 0) >= 5 or float(stock.dividend_yield or 0) >= 0.02
			universe = 'BLUECHIP' if is_stable else 'PENNY'
			payload = load_or_train_model(frame, model_path=get_model_path(universe))
			if not payload or not payload.get('model'):
				return {'win_rate': None, 'sharpe': None}
			backtester = AIBacktester(frame, payload, symbol=symbol)
			result = backtester.run_simulation(lookback_days=90)
			return {'win_rate': float(result.win_rate), 'sharpe': float(result.sharpe_ratio)}
		except Exception:
			return {'win_rate': None, 'sharpe': None}

	def _macro_snapshot(self) -> dict[str, Any]:
		cache_key = "macro_snapshot"
		cached = cache.get(cache_key)
		if cached is not None:
			return cached
		try:
			dxy = yf.Ticker('DX-Y.NYB').history(period='5d', interval='1d')
			oil = yf.Ticker('CL=F').history(period='5d', interval='1d')
			gold = yf.Ticker('GC=F').history(period='5d', interval='1d')
			def _latest_change(frame):
				if frame is None or frame.empty or 'Close' not in frame:
					return None, None
				closes = frame['Close'].dropna()
				if len(closes) < 2:
					return float(closes.iloc[-1]), None
				return float(closes.iloc[-1]), float((closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2] * 100)
			dxy_price, dxy_change = _latest_change(dxy)
			oil_price, oil_change = _latest_change(oil)
			gold_price, gold_change = _latest_change(gold)
			tech_risk = bool(dxy_change is not None and dxy_change > 0.5) or bool(gold_change is not None and gold_change > 0.5)
			result = {
				'dxy': dxy_price,
				'dxy_change_pct': dxy_change,
				'oil': oil_price,
				'oil_change_pct': oil_change,
				'gold': gold_price,
				'gold_change_pct': gold_change,
				'tech_risk': tech_risk,
			}
			cache.set(cache_key, result, 60 * 10)
			return result
		except Exception:
			return {}

	def _pyramid_steps(self, symbol: str, price: float, avg_cost: float, volume_z: float | None, rsi: float | None, ma20: float | None) -> dict[str, Any]:
		ai_score, _ = self._ai_score(symbol)
		macro = self._macro_snapshot()
		tech_risk = macro.get('tech_risk')
		oil_change = macro.get('oil_change_pct')
		if symbol.upper() == 'ATD.TO' and oil_change is not None and oil_change < 0:
			ai_score = (ai_score or 0) + 5
		is_tech = (Stock.objects.filter(symbol__iexact=symbol).values_list('sector', flat=True).first() or '').lower() == 'technology'
		macro_veto = bool(tech_risk and is_tech)
		vague_amount = 2300 * 0.25
		steps = []
		step1_ok = ai_score is not None and ai_score >= 85 and (volume_z is not None and volume_z > 0.5)
		steps.append({'label': 'Vague 1 (Éclaireur)', 'condition': 'AI Score > 85% & Volume Z > 0.5', 'amount': vague_amount, 'status': 'READY' if step1_ok else 'WAITING'})
		step2_ok = price >= (avg_cost * 1.02) and (volume_z is not None and volume_z >= 0)
		if step2_ok:
			stock = Stock.objects.filter(symbol__iexact=symbol).first()
			cutoff = timezone.now() - timedelta(hours=24)
			existing = AlertEvent.objects.filter(
				category='PYRAMID_WAVE_2',
				stock=stock,
				created_at__gte=cutoff,
			).exists()
			if not existing:
				AlertEvent.objects.create(
					category='PYRAMID_WAVE_2',
					stock=stock,
					message=f"🚀 VAGUE 2 VALIDÉE : {symbol} prix +2% avec Volume Z stable.",
				)
		steps.append({'label': 'Vague 2 (Confirmation)', 'condition': 'Prix > Achat + 2% & Volume Z stable', 'amount': vague_amount, 'status': 'READY' if step2_ok and not macro_veto else 'WAITING'})
		step3_ok = (ma20 is not None and price > ma20) and (rsi is not None and rsi < 70)
		steps.append({'label': 'Vague 3 (Tendance)', 'condition': 'Prix > MA20 & RSI < 70', 'amount': vague_amount, 'status': 'READY' if step3_ok and not macro_veto else 'WAITING'})
		step4_ok = price >= avg_cost
		steps.append({'label': 'Vague 4 (Finale)', 'condition': 'Stop-loss au point mort (Break-even)', 'amount': vague_amount, 'status': 'READY' if step4_ok else 'LOCKED'})
		progress = 25 * sum(1 for step in steps if step['status'] == 'READY')
		next_step = next((step for step in steps if step['status'] != 'READY'), steps[-1])
		return {
			'progress_pct': min(progress, 100),
			'next_step': next_step,
			'steps': steps,
			'macro_veto': macro_veto,
		}

	def _insider_summary(self, symbol: str) -> dict[str, Any] | None:
		cache_key = f"insider_summary:{symbol}"
		cached = cache.get(cache_key)
		if cached is not None:
			return cached
		api_key = os.getenv('FMP_API_KEY')
		if not api_key:
			return None
		url = f"https://financialmodelingprep.com/api/v4/insider-trading?symbol={symbol}&limit=50&apikey={api_key}"
		try:
			resp = requests.get(url, timeout=10)
			if resp.status_code != 200:
				return None
			items = resp.json() or []
		except Exception:
			return None
		if not isinstance(items, list) or not items:
			cache.set(cache_key, None, 60 * 60)
			return None
		cutoff = timezone.now() - timedelta(days=90)
		total_buy = 0.0
		for item in items:
			date_str = item.get('transactionDate') or item.get('date')
			if date_str:
				try:
					if datetime.fromisoformat(date_str).date() < cutoff.date():
						continue
				except Exception:
					pass
			trans_type = (item.get('transactionType') or '').lower()
			if 'purchase' not in trans_type and 'buy' not in trans_type:
				continue
			amount = item.get('transactionValue')
			if amount is None:
				price = float(item.get('price') or 0)
				shares = float(item.get('securitiesTransacted') or 0)
				amount = price * shares
			try:
				total_buy += float(amount or 0)
			except Exception:
				continue
		result = {
			'total_buy': round(total_buy, 2),
			'insiders_buying': total_buy >= 50000,
		}
		cache.set(cache_key, result, 60 * 60 * 6)
		return result

	def _institutional_summary(self, symbol: str) -> dict[str, Any] | None:
		cache_key = f"institutional_summary:{symbol}"
		cached = cache.get(cache_key)
		if cached is not None:
			return cached
		try:
			ticker = yf.Ticker(symbol)
			inst = getattr(ticker, 'institutional_holders', None)
			if inst is None or inst.empty:
				inst = getattr(ticker, 'major_holders', None)
			if inst is None or inst.empty:
				cache.set(cache_key, None, 60 * 60)
				return None
			holders = []
			accumulation = 0
			exit_warning = False
			if 'Holder' in inst.columns:
				for _, row in inst.head(5).iterrows():
					change = row.get('Change')
					change_val = None
					try:
						change_val = float(change)
					except Exception:
						change_val = None
					if change_val is not None:
						if change_val > 0:
							accumulation += 1
						if change_val <= -0.1:
							exit_warning = True
					holders.append({
						'holder': row.get('Holder'),
						'shares': row.get('Shares') or row.get('TotalShares') or row.get('Shares Outstanding'),
						'change': change,
					})
			result = {
				'holders': holders,
				'accumulation_count': accumulation,
				'whale_signal': accumulation >= 2,
				'exit_warning': exit_warning,
			}
			cache.set(cache_key, result, 60 * 60 * 6)
			return result
		except Exception:
			return None

	def get(self, request):
		account_id = request.query_params.get('account_id')
		accounts = Account.objects.all()
		if account_id:
			accounts = accounts.filter(id=account_id)

		tx_qs = AccountTransaction.objects.select_related('account', 'stock').all().order_by('date')
		if account_id:
			tx_qs = tx_qs.filter(account_id=account_id)

		position_map: dict[int, dict[str, dict[str, Any]]] = {}
		for tx in tx_qs:
			if tx.type == 'DIVIDEND':
				continue
			account_entry = position_map.setdefault(tx.account_id, {})
			symbol_key = (tx.stock.symbol or '').strip().upper() or str(tx.stock_id)
			entry = account_entry.setdefault(
				symbol_key,
				{'stock': tx.stock, 'shares': 0.0, 'buy_qty': 0.0, 'buy_cost': 0.0},
			)
			qty = float(tx.quantity or 0)
			if tx.type == 'BUY':
				entry['shares'] += qty
				entry['buy_qty'] += qty
				entry['buy_cost'] += qty * float(tx.price or 0)
			elif tx.type == 'SELL':
				avg_cost = (entry['buy_cost'] / entry['buy_qty']) if entry['buy_qty'] else 0.0
				entry['shares'] -= qty
				entry['buy_qty'] = max(0.0, entry['buy_qty'] - qty)
				entry['buy_cost'] = max(0.0, entry['buy_cost'] - avg_cost * qty)

		accounts_payload = []
		all_positions = []
		today = timezone.now().date()
		weekly_date = today - timedelta(days=7)
		monthly_date = today - timedelta(days=30)
		annual_date = today - timedelta(days=365)
		max_enrich = int(os.getenv('ACCOUNT_ENRICH_LIMIT', '8'))

		for account in accounts:
			positions = []
			total_value = 0.0
			total_cost = 0.0
			macro_snapshot = self._macro_snapshot()
			pre_entries = []
			for payload in position_map.get(account.id, {}).values():
				shares = float(payload['shares'] or 0)
				if shares <= 0:
					continue
				stock = payload['stock']
				buy_qty = float(payload.get('buy_qty') or 0)
				buy_cost = float(payload.get('buy_cost') or 0)
				avg_cost = (buy_cost / buy_qty) if buy_qty else 0.0
				current = self._current_price(stock)
				current_value = current * shares
				cost_value = avg_cost * shares
				total_value += current_value
				total_cost += cost_value
				pre_entries.append({
					'stock': stock,
					'shares': shares,
					'avg_cost': avg_cost,
					'current': current,
					'current_value': current_value,
					'cost_value': cost_value,
				})

			sorted_entries = sorted(pre_entries, key=lambda item: item['current_value'], reverse=True)
			enriched_symbols = {
				(item['stock'].symbol or '').strip().upper()
				for item in sorted_entries[:max_enrich]
				if item.get('stock')
			}

			for entry in sorted_entries:
				stock = entry['stock']
				symbol = (stock.symbol or '').strip().upper()
				shares = entry['shares']
				avg_cost = entry['avg_cost']
				current = entry['current']
				current_value = entry['current_value']
				cost_value = entry['cost_value']
				unrealized_pct = ((current_value - cost_value) / cost_value * 100) if cost_value else None
				rsi = self._get_rsi(symbol)
				ma20 = self._get_ma20(symbol)
				volume_z = self._get_volume_z(symbol)
				stats = {'win_rate': None, 'sharpe': None}
				insider = None
				institutional = None
				if symbol in enriched_symbols:
					stats = self._model_stats(symbol)
					insider = self._insider_summary(symbol)
					institutional = self._institutional_summary(symbol)
				pyramid = self._pyramid_steps(symbol, current, avg_cost, volume_z=volume_z, rsi=rsi, ma20=ma20)

				weekly_price = self._price_at_or_before(stock, weekly_date)
				monthly_price = self._price_at_or_before(stock, monthly_date)
				annual_price = self._price_at_or_before(stock, annual_date)

				weekly_return = ((current - weekly_price) / weekly_price * 100) if weekly_price else None
				monthly_return = ((current - monthly_price) / monthly_price * 100) if monthly_price else None
				annual_return = ((current - annual_price) / annual_price * 100) if annual_price else None

				latest_two = list(PriceHistory.objects.filter(stock=stock).order_by('-date')[:2])
				prev_close = float(latest_two[1].close_price) if len(latest_two) > 1 else None
				day_change_pct = ((current - prev_close) / prev_close * 100) if prev_close else None
				day_change_value = ((current - prev_close) * shares) if prev_close else None

				position_payload = {
					'account_id': account.id,
					'account_name': account.name,
					'account_type': account.account_type,
					'ticker': stock.symbol,
					'name': stock.name,
					'avg_cost': round(avg_cost, 4),
					'shares': round(shares, 4),
					'cost_value': round(cost_value, 2),
					'current_price': round(current, 4),
					'current_value': round(current_value, 2),
					'rsi': round(rsi, 2) if rsi is not None else None,
					'ma20': round(ma20, 4) if ma20 is not None else None,
					'volume_z': round(volume_z, 2) if volume_z is not None else None,
					'stop_price': self._stop_price(current),
					'pyramid': pyramid,
					'insider': insider,
					'institutional': institutional,
					'model_win_rate': round(float(stats.get('win_rate') or 0), 2) if stats.get('win_rate') is not None else None,
					'model_sharpe': round(float(stats.get('sharpe') or 0), 2) if stats.get('sharpe') is not None else None,
					'unrealized_pnl_pct': round(unrealized_pct, 2) if unrealized_pct is not None else None,
					'weekly_return_pct': round(weekly_return, 2) if weekly_return is not None else None,
					'monthly_return_pct': round(monthly_return, 2) if monthly_return is not None else None,
					'annual_return_pct': round(annual_return, 2) if annual_return is not None else None,
					'day_change_pct': round(day_change_pct, 2) if day_change_pct is not None else None,
					'day_change_value': round(day_change_value, 2) if day_change_value is not None else None,
				}
				positions.append(position_payload)
				all_positions.append(position_payload)

			accounts_payload.append({
				'account_id': account.id,
				'account_name': account.name,
				'account_type': account.account_type,
				'total_value': round(total_value, 2),
				'total_cost': round(total_cost, 2),
				'positions': positions,
				'macro': macro_snapshot,
			})

		gainer = None
		loser = None
		for pos in all_positions:
			change = pos.get('day_change_pct')
			if change is None:
				continue
			if gainer is None or change > gainer.get('day_change_pct', -9999):
				gainer = pos
			if loser is None or change < loser.get('day_change_pct', 9999):
				loser = pos

		return Response({
			'accounts': accounts_payload,
			'top_movers': {
				'gainer': gainer,
				'loser': loser,
			},
		})


class PortfolioNewsView(APIView):
	def _serialize_news(self, items: list[StockNews]) -> list[dict[str, Any]]:
		results = []
		for news in items:
			stock = news.stock
			results.append({
				'ticker': stock.symbol if stock else None,
				'headline': news.headline,
				'url': news.url,
				'published_at': news.published_at.isoformat() if news.published_at else None,
				'sentiment': float(news.sentiment) if news.sentiment is not None else None,
				'source': news.source,
				'sector': stock.sector if stock else None,
			})
		return results

	def get(self, request):
		portfolio_id = request.query_params.get('portfolio_id')
		symbol = (request.query_params.get('symbol') or '').strip().upper()
		limit = int(request.query_params.get('limit', 10))
		limit = max(1, min(limit, 50))
		sector_limit = int(request.query_params.get('sector_limit', 10))
		sector_limit = max(1, min(sector_limit, 50))
		sentiment_limit = int(request.query_params.get('sentiment_limit', 6))
		sentiment_limit = max(1, min(sentiment_limit, 20))
		sentiment_threshold = float(request.query_params.get('sentiment_threshold', 0.35))

		portfolio = None
		if portfolio_id:
			portfolio = Portfolio.objects.filter(id=portfolio_id).first()
		if not portfolio:
			portfolio = Portfolio.objects.first()

		symbols: list[str] = []
		if portfolio:
			holdings = PortfolioHolding.objects.select_related('stock').filter(portfolio=portfolio)
			symbols = [h.stock.symbol for h in holdings if h.stock and h.stock.symbol]
			if not symbols:
				transactions = AccountTransaction.objects.select_related('stock', 'account').all()
				symbols = [tx.stock.symbol for tx in transactions if tx.stock and tx.stock.symbol]
				symbols = list(dict.fromkeys(symbols))
		else:
			transactions = AccountTransaction.objects.select_related('stock', 'account').all()
			symbols = [tx.stock.symbol for tx in transactions if tx.stock and tx.stock.symbol]
			symbols = list(dict.fromkeys(symbols))
		if symbol:
			symbols = [symbol]

		sectors = list(
			Stock.objects.filter(symbol__in=symbols).exclude(sector='').values_list('sector', flat=True).distinct()
		)

		holdings_news = StockNews.objects.filter(stock__symbol__in=symbols).order_by('-published_at')[:limit]
		sector_news = StockNews.objects.filter(stock__sector__in=sectors).order_by('-published_at')[:sector_limit]
		positive_news = StockNews.objects.filter(
			stock__sector__in=sectors,
			sentiment__gte=sentiment_threshold,
		).order_by('-sentiment', '-published_at')[:sentiment_limit]
		negative_news = StockNews.objects.filter(
			stock__sector__in=sectors,
			sentiment__lte=-sentiment_threshold,
		).order_by('sentiment', '-published_at')[:sentiment_limit]

		if (
			not holdings_news
			and not sector_news
			and not positive_news
			and not negative_news
		):
			fallback_news = StockNews.objects.order_by('-published_at')[:limit]
			holdings_news = fallback_news

		return Response({
			'portfolio': {'id': portfolio.id, 'name': portfolio.name} if portfolio else None,
			'symbols': symbols,
			'sectors': sectors,
			'holdings': self._serialize_news(list(holdings_news)),
			'sectors_news': self._serialize_news(list(sector_news)),
			'sentiment': {
				'positive': self._serialize_news(list(positive_news)),
				'negative': self._serialize_news(list(negative_news)),
			},
			'thresholds': {
				'sentiment': sentiment_threshold,
			},
		})


class SentimentScannerView(APIView):
	def _sentiment_payload(self, symbol: str) -> dict[str, Any]:
		avg_sentiment, count_24h, count_prev = _news_sentiment_24h(symbol)
		hype_ratio = (count_24h / max(count_prev, 1)) if count_24h else 0.0
		hype_level = 'HIGH_HYPE' if hype_ratio > 2.0 else 'ORGANIC'
		mood = 'NEUTRAL'
		if avg_sentiment >= 0.6:
			mood = 'BULLISH'
		elif avg_sentiment <= 0.4:
			mood = 'BEARISH'
		recommendation = 'STABLE'
		if hype_level == 'HIGH_HYPE' and mood == 'BULLISH':
			recommendation = "⚠️ EUPHORIA: Don't FOMO. Tighten stops."
		elif hype_level == 'ORGANIC' and mood == 'BEARISH':
			recommendation = '💎 VALUE PLAY: Look for entry signals.'
		hype_detected = False
		if count_prev > 0:
			hype_detected = count_24h > (count_prev * 1.5)
		else:
			hype_detected = count_24h >= 10
		return {
			'symbol': symbol,
			'sentiment_score': round(float(avg_sentiment), 3),
			'news_count_24h': int(count_24h),
			'news_count_prev_24h': int(count_prev),
			'hype_ratio': round(float(hype_ratio), 2),
			'hype_detected': hype_detected,
			'hype_level': hype_level,
			'market_mood': mood,
			'recommendation': recommendation,
		}

	def get(self, request):
		tickers_param = request.query_params.get('tickers')
		limit = int(request.query_params.get('limit', 8))
		limit = max(1, min(limit, 25))
		if tickers_param:
			symbols = [s.strip().upper() for s in tickers_param.split(',') if s.strip()]
		else:
			portfolio = Portfolio.objects.first()
			if portfolio:
				symbols = list(
					PortfolioHolding.objects.filter(portfolio=portfolio)
					.values_list('stock__symbol', flat=True)
				)
			else:
				symbols = list(Stock.objects.order_by('symbol').values_list('symbol', flat=True))
			symbols = [s for s in symbols if s][:limit]

		results = [self._sentiment_payload(symbol) for symbol in symbols]

		btc_score = _rss_sentiment_window('Bitcoin', hours=4, offset_hours=0)
		btc_prev = _rss_sentiment_window('Bitcoin', hours=4, offset_hours=4)
		tech_score = _rss_sentiment_window('TEC.TO', hours=24, offset_hours=0)
		risk_off = btc_prev != 0 and ((btc_score - btc_prev) / abs(btc_prev)) <= -0.2
		divergence = abs(btc_score - tech_score) > 0.3
		market_vitals = {
			'btc_sentiment': round(float(btc_score), 3),
			'btc_prev_4h': round(float(btc_prev), 3),
			'tech_sentiment': round(float(tech_score), 3),
			'btc_drop_20pct_4h': bool(risk_off),
			'divergence': bool(divergence),
		}

		return Response({
			'results': results,
			'market_vitals': market_vitals,
		})


class StablePredictionView(APIView):
	def get(self, request, symbol: str):
		symbol = (symbol or '').strip().upper()
		if not symbol:
			return Response({'error': 'symbol is required'}, status=400)

		model_path = Path(__file__).resolve().parent / 'ml_engine' / 'stable_brain_v1.pkl'
		if not model_path.exists():
			return Response({'error': 'Stable model not trained yet.'}, status=503)

		try:
			model = joblib.load(model_path)
		except Exception:
			return Response({'error': 'Failed to load model.'}, status=503)

		try:
			data = yf.Ticker(symbol).history(period='2y', interval='1d', timeout=10)
			spy = yf.Ticker('SPY').history(period='2y', interval='1d', timeout=10)
		except Exception:
			return Response({'error': 'Failed to load price data.', 'symbol': symbol, 'recommendation': 'HOLD'}, status=200)

		if data is None or data.empty or 'Close' not in data or len(data) < 200:
			return Response({'error': 'Insufficient price history.', 'symbol': symbol, 'recommendation': 'HOLD'}, status=200)
		if spy is None or spy.empty or 'Close' not in spy or len(spy) < 200:
			return Response({'error': 'Insufficient SPY history.', 'symbol': symbol, 'recommendation': 'HOLD'}, status=200)

		def _normalize_index(series: pd.Series) -> pd.Series:
			try:
				idx = pd.to_datetime(series.index)
			except Exception:
				return series
			try:
				if getattr(idx, 'tz', None) is not None:
					idx = idx.tz_localize(None)
			except Exception:
				pass
			series = series.copy()
			series.index = pd.to_datetime(idx).normalize()
			return series

		close = _normalize_index(data['Close'])
		volume = data['Volume'] if 'Volume' in data else pd.Series([0] * len(close), index=close.index)
		volume = _normalize_index(volume)
		ret = close.pct_change().dropna()

		spy_close = _normalize_index(spy['Close'])
		spy_ret = spy_close.pct_change().dropna()
		aligned = pd.concat([ret, spy_ret], axis=1, join='inner').dropna()
		aligned.columns = ['stock', 'spy']
		if len(aligned) < 60:
			return Response({'error': 'Insufficient aligned history.', 'symbol': symbol, 'recommendation': 'HOLD'}, status=200)

		beta = float(aligned['stock'].cov(aligned['spy']) / (aligned['spy'].var() or 1))
		log_ret_20 = float(np.log(close.iloc[-1] / close.iloc[-21]))
		vol_60 = float(ret.tail(60).std())
		rel_volume_200 = float(volume.tail(5).mean() / (volume.tail(200).mean() or 1))
		dividend_yield = float(Stock.objects.filter(symbol__iexact=symbol).values_list('dividend_yield', flat=True).first() or 0)

		macro = MacroIndicator.objects.order_by('-date').first()
		macro_features = [
			float(macro.sp500_close) if macro else 0.0,
			float(macro.vix_index) if macro else 0.0,
			float(macro.interest_rate_10y) if macro else 0.0,
			float(macro.inflation_rate) if macro else 0.0,
			float(macro.oil_price) if (macro and macro.oil_price is not None) else 0.0,
		]

		features = np.array([
			log_ret_20,
			vol_60,
			beta,
			rel_volume_200,
			dividend_yield,
			*macro_features,
		])

		try:
			pred = float(model.predict([features])[0])
		except Exception:
			return Response({'error': 'Prediction failed.', 'symbol': symbol, 'recommendation': 'HOLD'}, status=200)

		return Response({
			'symbol': symbol,
			'predicted_20d_return': round(pred, 4),
			'recommendation': 'BUY' if pred >= 0.02 else 'HOLD',
		})


class BluechipHunterView(APIView):
	def get(self, request):
		limit = int(request.query_params.get('limit', 15))
		limit = max(1, min(limit, 50))
		portfolio_id = request.query_params.get('portfolio_id')

		portfolio = Portfolio.objects.filter(id=portfolio_id).first() if portfolio_id else Portfolio.objects.first()
		if portfolio:
			holdings = PortfolioHolding.objects.select_related('stock').filter(portfolio=portfolio)
		else:
			holdings = PortfolioHolding.objects.select_related('stock').all()

		existing_symbols = {
			h.stock.symbol.upper()
			for h in holdings
			if h.stock and h.stock.symbol
		}

		scr_ids = os.getenv(
			'BLUECHIP_SCREENER_IDS',
			'undervalued_large_caps,high_dividend_yield,most_actives',
		).split(',')
		scr_ids = [s.strip() for s in scr_ids if s.strip()]

		fallback_symbols = [
			'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'JPM', 'V', 'MA',
			'UNH', 'XOM', 'COST', 'AVGO', 'LLY', 'ORCL', 'ADBE', 'CRM', 'NFLX', 'PEP',
			'KO', 'WMT', 'HD', 'BAC', 'INTC', 'CSCO', 'AMD', 'QCOM', 'TXN', 'ABBV',
			'PFE', 'MRK', 'JNJ', 'PG', 'NKE', 'TMO', 'LIN', 'MCD', 'DIS',
		]

		quotes: list[dict] = []
		for scr_id in scr_ids:
			quotes.extend(_fetch_yahoo_screener(scr_id, count=50))

		if not quotes:
			quotes = _fetch_yfinance_screeners(scr_ids, count=50)

		if not quotes:
			quotes = [{'symbol': symbol} for symbol in fallback_symbols]

		seen: set[str] = set()
		candidates: list[dict] = []
		for quote in quotes:
			symbol = str(quote.get('symbol') or '').upper().strip()
			if not symbol or symbol in seen or symbol in existing_symbols:
				continue
			seen.add(symbol)
			candidates.append(quote)

		if not candidates:
			seen.clear()
			for symbol in fallback_symbols:
				if symbol in seen or symbol in existing_symbols:
					continue
				seen.add(symbol)
				candidates.append({'symbol': symbol})

		model_path = Path(__file__).resolve().parent / 'ml_engine' / 'stable_brain_v1.pkl'
		model = None
		fast_mode = os.getenv('BLUECHIP_FAST_MODE', 'true').lower() != 'false'
		if model_path.exists() and not fast_mode:
			try:
				model = joblib.load(model_path)
			except Exception:
				model = None

		macro = MacroIndicator.objects.order_by('-date').first()
		def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
			return max(low, min(high, value))

		macro_features = [
			float(macro.sp500_close) if macro else 0.0,
			float(macro.vix_index) if macro else 0.0,
			float(macro.interest_rate_10y) if macro else 0.0,
			float(macro.inflation_rate) if macro else 0.0,
			float(macro.oil_price) if (macro and macro.oil_price is not None) else 0.0,
		]
		macro_score = 0.5
		if macro:
			sp_score = _clamp((macro.sp500_close - 3500) / 2000) if macro.sp500_close else 0.5
			vix_score = _clamp(1 - ((macro.vix_index or 20) - 15) / 25)
			rate_score = _clamp(1 - ((macro.interest_rate_10y or 3) - 2) / 4)
			inflation_score = _clamp(1 - ((macro.inflation_rate or 2) - 2) / 4)
			oil_score = _clamp(1 - ((macro.oil_price or 70) - 60) / 60)
			macro_score = _clamp((sp_score + vix_score + rate_score + inflation_score + oil_score) / 5)

		results = []
		min_backtest_signal = float(os.getenv('BLUECHIP_BACKTEST_MIN_SIGNAL', '0.7'))
		for quote in candidates[: limit * 2]:
			symbol = str(quote.get('symbol') or '').upper().strip()
			name = quote.get('shortName') or quote.get('longName') or symbol
			sector = quote.get('sector') or ''
			price = float(quote.get('regularMarketPrice') or 0)
			dividend_yield = float(quote.get('dividendYield') or 0)
			revenue_growth = float(quote.get('revenueGrowth') or 0)
			backtest_signal = None

			predicted = None
			if model:
				try:
					data = yf.Ticker(symbol).history(period='1y', interval='1d', timeout=10)
					spy = yf.Ticker('SPY').history(period='1y', interval='1d', timeout=10)
					if data is not None and not data.empty and spy is not None and not spy.empty:
						close = data['Close']
						volume = data['Volume'] if 'Volume' in data else pd.Series([0] * len(close), index=close.index)
						ret = close.pct_change().dropna()
						spy_close = spy['Close']
						spy_ret = spy_close.pct_change().dropna()
						aligned = pd.concat([ret, spy_ret], axis=1, join='inner').dropna()
						aligned.columns = ['stock', 'spy']
						if len(aligned) >= 60:
							beta = float(aligned['stock'].cov(aligned['spy']) / (aligned['spy'].var() or 1))
							log_ret_20 = float(np.log(close.iloc[-1] / close.iloc[-21]))
							vol_60 = float(ret.tail(60).std())
							rel_volume_200 = float(volume.tail(5).mean() / (volume.tail(200).mean() or 1))
							features = np.array([
								log_ret_20,
								vol_60,
								beta,
								rel_volume_200,
								dividend_yield,
								*macro_features,
							])
							predicted = float(model.predict([features])[0])
				except Exception:
					predicted = None

			if not fast_mode:
				try:
					fusion = DataFusionEngine(symbol)
					fusion_df = fusion.fuse_all()
					if fusion_df is not None and not fusion_df.empty:
						bt_payload = load_or_train_model(
							fusion_df,
							model_path=get_model_path('BLUECHIP'),
						)
						if bt_payload and bt_payload.get('model'):
							last_row = fusion_df.tail(1).copy()
							feature_list = bt_payload.get('features', FEATURE_COLUMNS)
							for col in feature_list:
								if col not in last_row.columns:
									last_row[col] = 0.0
								features = last_row[feature_list].fillna(0).values
								backtest_signal = float(bt_payload['model'].predict_proba(features)[0][1])
				except Exception:
					backtest_signal = None

			if backtest_signal is not None and backtest_signal < min_backtest_signal:
				continue

			pred_score = _clamp(0.5 + (predicted or 0.0) * 10)
			growth_pct = revenue_growth * 100 if revenue_growth <= 1 else revenue_growth
			yield_pct = dividend_yield * 100 if dividend_yield <= 1 else dividend_yield
			fundamental_score = _clamp(0.4 + (revenue_growth * 2) + (dividend_yield * 4))
			ai_score = 100 * _clamp(
				(0.45 * fundamental_score)
				+ (0.30 * macro_score)
				+ (0.10 * pred_score)
				+ (0.15 * (backtest_signal or 0.0))
			)
			ai_score = round(ai_score, 2)

			results.append({
				'ticker': symbol,
				'name': name,
				'sector': sector,
				'latest_price': price,
				'dividend_yield': round(yield_pct, 2),
				'revenue_growth': round(growth_pct, 2),
				'predicted_20d_return': round(predicted, 4) if predicted is not None else None,
				'backtest_signal': round(backtest_signal, 4) if backtest_signal is not None else None,
				'ai_score': ai_score,
			})

		if not results:
			for quote in candidates[:limit]:
				symbol = str(quote.get('symbol') or '').upper().strip()
				if not symbol:
					continue
				name = quote.get('shortName') or quote.get('longName') or symbol
				sector = quote.get('sector') or ''
				price = float(quote.get('regularMarketPrice') or 0)
				dividend_yield = float(quote.get('dividendYield') or 0)
				revenue_growth = float(quote.get('revenueGrowth') or 0)
				results.append({
					'ticker': symbol,
					'name': name,
					'sector': sector,
					'latest_price': price,
					'dividend_yield': round(dividend_yield * 100, 2) if dividend_yield <= 1 else round(dividend_yield, 2),
					'revenue_growth': round(revenue_growth * 100, 2) if revenue_growth <= 1 else round(revenue_growth, 2),
					'predicted_20d_return': None,
					'backtest_signal': None,
					'ai_score': 70.0,
				})

		results.sort(key=lambda x: x.get('ai_score', 0), reverse=True)
		return Response(results[:limit])


class AIRecommendationsRunView(APIView):
	def post(self, request):
		created = 0
		updated = 0
		results = []
		as_of = timezone.now().date()

		for stock in Stock.objects.all().order_by('symbol'):
			predicted_price, recommendation = run_predictions(stock.symbol)
			obj, was_created = Prediction.objects.update_or_create(
				stock=stock,
				date=as_of,
				defaults={
					'predicted_price': predicted_price,
					'recommendation': recommendation,
				},
			)
			if was_created:
				created += 1
			else:
				updated += 1
			results.append({
				'symbol': stock.symbol,
				'predicted_price': obj.predicted_price,
				'recommendation': obj.recommendation,
			})

		return Response({
			'as_of': str(as_of),
			'created': created,
			'updated': updated,
			'results': results,
		})


class AIOpportunityView(APIView):
	def get(self, request):
		limit = int(request.query_params.get('limit', 12))
		min_score = float(request.query_params.get('min_score', 0.25))
		min_market_cap = int(request.query_params.get('min_market_cap', 2_000_000_000))
		min_price = float(request.query_params.get('min_price', 5))
		portfolio_id = request.query_params.get('portfolio_id')
		include_universe = request.query_params.get('include_universe', 'true').lower() != 'false'
		news_days = int(request.query_params.get('news_days', 100))
		news_days = max(7, min(news_days, 180))
		limit = max(1, min(limit, 50))

		excluded_symbols = set()
		if portfolio_id:
			excluded_symbols = set(
				PortfolioHolding.objects.filter(portfolio_id=portfolio_id)
				.values_list('stock__symbol', flat=True)
			)

		cutoff = timezone.now() - timedelta(days=news_days)
		results = []
		analyzer = SentimentIntensityAnalyzer()

		mega_universe = [
			'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'JPM', 'V',
			'MA', 'UNH', 'XOM', 'COST', 'AVGO', 'LLY', 'ORCL', 'ADBE', 'CRM', 'NFLX',
			'PEP', 'KO', 'WMT', 'HD', 'BAC', 'INTC', 'CSCO', 'AMD', 'QCOM', 'TXN',
			'ABBV', 'PFE', 'MRK', 'JNJ', 'PG', 'NKE', 'TMO', 'LIN', 'MCD', 'DIS',
		]

		stock_map = {s.symbol: s for s in Stock.objects.all().order_by('symbol')}
		symbols = list(stock_map.keys())
		if include_universe:
			for sym in mega_universe:
				if sym not in symbols:
					symbols.append(sym)

		for symbol in symbols:
			if symbol in excluded_symbols:
				continue
			stock = stock_map.get(symbol)
			info = {}
			try:
				info = yf.Ticker(symbol).info or {}
			except Exception:
				info = {}

			market_cap = info.get('marketCap')
			revenue_growth = info.get('revenueGrowth')
			earnings_growth = info.get('earningsGrowth')
			profit_margins = info.get('profitMargins')
			roe = info.get('returnOnEquity')
			beta = info.get('beta')
			current_price = info.get('currentPrice') or (stock.latest_price if stock else None)

			if market_cap is not None and market_cap < min_market_cap:
				continue
			if current_price is not None and float(current_price) < min_price:
				continue

			if stock:
				news_sentiment = (
					StockNews.objects.filter(stock=stock, fetched_at__gte=cutoff)
					.aggregate(avg=models.Avg('sentiment'))
					.get('avg')
				) or 0
			else:
				news_sentiment = 0
				try:
					news_items = (info.get('news') or [])
				except Exception:
					news_items = []
				if not news_items:
					try:
						news_items = yf.Ticker(symbol).news or []
					except Exception:
						news_items = []
				if news_items:
					scores = []
					for item in news_items[:10]:
						headline = (item.get('title') or '')
						summary = (item.get('summary') or '')
						text = f"{headline}. {summary}".strip()
						if text:
							scores.append(analyzer.polarity_scores(text).get('compound', 0))
					news_sentiment = float(np.mean(scores)) if scores else 0

			# Stability from price history (lower volatility = more stable)
			if stock:
				prices = list(
					PriceHistory.objects.filter(stock=stock).order_by('-date')[:90]
				)
				price_series = [float(p.close_price or 0) for p in prices]
			else:
				price_series = []
				try:
					hist = yf.Ticker(symbol).history(period='6mo', interval='1d')
					if not hist.empty and 'Close' in hist:
						price_series = [float(v) for v in hist['Close'].tolist()][::-1]
				except Exception:
					price_series = []

			returns = []
			for i in range(len(price_series) - 1):
				p0 = float(price_series[i] or 0)
				p1 = float(price_series[i + 1] or 0)
				if p1:
					returns.append((p0 - p1) / p1)
			volatility = float(np.std(returns)) if returns else None
			stability_score = 0.15 if volatility is not None and volatility < 0.035 else 0

			reasons = []
			score = 0.0

			# Stability preference
			if beta is not None and beta < 1:
				score += 0.1
				reasons.append('Lower volatility (beta < 1)')

			# Sentiment
			if news_sentiment > 0.2:
				score += 0.2
				reasons.append('Positive recent sentiment')
			elif news_sentiment < -0.2:
				score -= 0.2
				reasons.append('Negative recent sentiment')

			# Fundamentals
			if revenue_growth is not None and revenue_growth > 0:
				score += 0.15
				reasons.append('Revenue growth positive')
			if earnings_growth is not None and earnings_growth > 0:
				score += 0.1
				reasons.append('Earnings growth positive')
			if profit_margins is not None and profit_margins > 0:
				score += 0.1
				reasons.append('Positive profit margins')
			if roe is not None and roe > 0:
				score += 0.1
				reasons.append('Positive ROE')

			score += stability_score
			if stability_score > 0:
				reasons.append('Stable price action')

			if score < min_score:
				continue

			results.append({
				'symbol': symbol,
				'name': (stock.name if stock else (info.get('shortName') or info.get('longName') or symbol)),
				'score': round(score, 3),
				'price': current_price,
				'market_cap': market_cap,
				'news_sentiment': news_sentiment,
				'volatility': volatility,
				'reasons': reasons,
			})

		results.sort(key=lambda x: x['score'], reverse=True)
		return Response({'results': results[:limit]})


class AIScoutView(APIView):
	def get(self, request):
		symbol = (request.query_params.get('symbol') or '').strip()
		if not symbol:
			return Response({'error': 'symbol is required'}, status=400)
		try:
			result = build_scout_summary(symbol)
			return Response(result)
		except Exception as exc:
			return Response({'error': str(exc)}, status=500)


class ValueNavigatorAPI(APIView):
	def get(self, request):
		limit = int(request.query_params.get('limit') or 25)
		limit = max(5, min(limit, 100))
		min_signal = float(request.query_params.get('min_signal') or 0.7)
		min_sentiment = float(request.query_params.get('min_sentiment') or 0.6)
		min_corr = float(request.query_params.get('min_corr') or 0.2)
		cache_key = f"value-navigator:{limit}:{min_signal}:{min_sentiment}:{min_corr}"
		cached = cache.get(cache_key)
		if cached:
			return Response(cached)

		model_path = get_model_path('BLUECHIP')
		payload = _load_signal_model_payload(model_path)
		if payload is None:
			return Response({'error': 'model not available'}, status=500)

		watchlist: list[str] = []
		try:
			quotes = _fetch_yfinance_screeners(
				['most_actives', 'day_gainers', 'undervalued_large_caps', 'growth_technology_stocks'],
				count=200,
			)
			watchlist.extend([
				str(q.get('symbol') or '').strip().upper()
				for q in quotes
				if isinstance(q, dict)
			])
			quotes = _fetch_yahoo_screener('most_actives', count=200)
			watchlist.extend([
				str(q.get('symbol') or '').strip().upper()
				for q in quotes
				if isinstance(q, dict)
			])
			quotes = _fetch_yahoo_screener('day_gainers', count=200)
			watchlist.extend([
				str(q.get('symbol') or '').strip().upper()
				for q in quotes
				if isinstance(q, dict)
			])
		except Exception:
			watchlist = []

		seen: set[str] = set()
		symbols: list[str] = []
		for sym in watchlist:
			if not sym or sym in seen:
				continue
			seen.add(sym)
			symbols.append(sym)
			if len(symbols) >= 500:
				break

		results: list[dict[str, Any]] = []
		for symbol in symbols:
			try:
				fusion = DataFusionEngine(symbol, use_alpaca=False)
				fusion_df = fusion.fuse_all()
				if fusion_df is None or fusion_df.empty:
					continue
				signal = _predict_model_signal(payload, fusion_df)
				if signal is None or signal < min_signal:
					continue

				close_series = _fusion_close_series(fusion_df)
				rsi = _compute_rsi_from_series(close_series) if close_series is not None else None
				sentiment_titles = _google_news_titles(symbol, days=2)
				sentiment_raw = _finbert_score_from_titles(sentiment_titles)
				sentiment_score = max(0.0, min(1.0, (sentiment_raw + 1.0) / 2.0))
				if sentiment_score < min_sentiment:
					continue

				parent = (CORRELATION_MAP.get(symbol) or os.getenv('VALUE_NAV_PARENT_DEFAULT', 'SPY')).strip().upper()
				parent_change = _intraday_pct_change(parent, minutes=60) or 0.0
				corr = _daily_correlation(symbol, parent, days=60)
				corr_value = float(corr or 0.0)
				if parent_change <= 0 or corr_value < min_corr:
					continue

				reasons = []
				if rsi is not None:
					if rsi <= 35:
						reasons.append(f"RSI sortie de survente ({rsi:.1f})")
					else:
						reasons.append(f"RSI {rsi:.1f}")
				reasons.append(f"Sentiment News {sentiment_score:.2f}")
				reasons.append(f"Corrélation {parent} ascendante")

				results.append({
					'symbol': symbol,
					'signal': round(float(signal), 3),
					'sentiment': round(float(sentiment_score), 3),
					'parent': parent,
					'parent_change_pct': round(float(parent_change) * 100, 2),
					'correlation': round(float(corr_value), 3),
					'explanation': "Acheter car " + " + ".join(reasons),
				})
				if len(results) >= limit:
					break
			except Exception:
				continue

		payload_out = {'results': results, 'count': len(results)}
		cache.set(cache_key, payload_out, timeout=60 * 10)
		return Response(payload_out)


class ValueNavigatorRecommendation(APIView):
	def get(self, request):
		retro_payload = _retro_model_payload()
		if retro_payload is None:
			return Response({'error': 'retro model not available'}, status=500)

		cache_key = 'value-navigator:recommendation'
		cached = cache.get(cache_key)
		if cached:
			return Response(cached)

		watchlist: list[str] = []
		try:
			quotes = _fetch_yfinance_screeners(
				['most_actives', 'day_gainers', 'undervalued_large_caps', 'growth_technology_stocks'],
				count=200,
			)
			watchlist.extend([
				str(q.get('symbol') or '').strip().upper()
				for q in quotes
				if isinstance(q, dict)
			])
			watchlist.extend([
				str(q.get('symbol') or '').strip().upper()
				for q in _fetch_yahoo_screener('most_actives', count=200)
				if isinstance(q, dict)
			])
			watchlist.extend([
				str(q.get('symbol') or '').strip().upper()
				for q in _fetch_yahoo_screener('day_gainers', count=200)
				if isinstance(q, dict)
			])
		except Exception:
			watchlist = []

		seen: set[str] = set()
		symbols: list[str] = []
		for sym in watchlist:
			if not sym or sym in seen:
				continue
			seen.add(sym)
			symbols.append(sym)
			if len(symbols) >= 500:
				break

		results: list[dict[str, Any]] = []
		feature_list = retro_payload.get('features') or []
		model = retro_payload.get('model')

		for symbol in symbols:
			row_data = _retro_feature_row(symbol)
			if row_data is None:
				continue
			features, last_row = row_data
			if not feature_list or model is None:
				continue
			vector = [float(features.get(col, 0.0) or 0.0) for col in feature_list]
			try:
				pred = float(model.predict([vector])[0])
			except Exception:
				continue

			predicted_move_pct = max(0.0, min(0.5, pred * 0.15))
			if predicted_move_pct < 0.10:
				continue

			price = float(last_row.get('close') or last_row.get('Close') or 0.0)
			atr = float(last_row.get('atr14') or 0.0)
			if price <= 0 or atr <= 0:
				continue
			stop = price - atr
			target = price + (3 * atr)
			risk = max(price - stop, 0.0001)
			reward = max(target - price, 0.0001)
			risk_reward = reward / risk
			if risk_reward < 3.0:
				continue

			stock = Stock.objects.filter(symbol__iexact=symbol).first()
			sector = (stock.sector if stock else None) or (yf.Ticker(symbol).info or {}).get('sector') or 'n/a'
			news_titles = _google_news_titles(symbol, days=2)
			if not _gemini_macro_ok(symbol, sector, news_titles):
				continue

			pattern = None
			if features.get('pattern_hammer'):
				pattern = 'Marteau'
			elif features.get('pattern_engulfing'):
				pattern = 'Engulfing'
			elif features.get('pattern_morning_star'):
				pattern = 'Morning Star'
			elif features.get('pattern_doji'):
				pattern = 'Doji'
			pattern_txt = pattern or 'Pattern'

			parent = CORRELATION_MAP.get(symbol) or os.getenv('PENNY_SNIPER_DEFAULT_PARENT', 'SPY')
			parent_change = _intraday_pct_change(parent, minutes=60) or 0.0
			news_sentiment = _finbert_score_from_titles(news_titles)
			news_score = max(0.0, min(1.0, (news_sentiment + 1.0) / 2.0))

			results.append({
				'ticker': symbol,
				'predicted_move_pct': round(predicted_move_pct * 100, 2),
				'risk_reward': round(risk_reward, 2),
				'entry': round(price, 4),
				'target': round(target, 4),
				'stop': round(stop, 4),
				'proof': (
					f"Recommandé car {pattern_txt} confirmé par hausse du {parent} "
					f"({parent_change * 100:+.2f}%) et news positive à {news_score:.2f}"
				),
			})
			if len(results) >= 3:
				break

		payload = {'results': results, 'count': len(results)}
		cache.set(cache_key, payload, timeout=60 * 15)
		return Response(payload)


class AIScoutBatchView(APIView):
	def post(self, request):
		portfolio_id = request.data.get('portfolio_id') or request.query_params.get('portfolio_id')
		if not portfolio_id:
			return Response({'error': 'portfolio_id is required'}, status=400)
		limit = int(request.data.get('limit') or request.query_params.get('limit') or 6)
		limit = max(1, min(limit, 12))

		symbols = list(
			PortfolioHolding.objects.filter(portfolio_id=portfolio_id)
			.values_list('stock__symbol', flat=True)
		)

		results = []
		for symbol in symbols[:limit]:
			try:
				results.append(build_scout_summary(symbol))
			except Exception as exc:
				results.append({'symbol': symbol, 'error': str(exc)})

		return Response({'results': results})


class GetAIAnalysisView(APIView):
	def get(self, request):
		ticker = (request.query_params.get('ticker') or '').strip().upper()
		if not ticker:
			return Response({'error': 'Ticker requis'}, status=400)
		try:
			task = analyze_ticker_for_ui.delay(ticker)
			result = task.get(timeout=25)
			if isinstance(result, dict) and result.get('error'):
				return Response(result, status=400)
			return Response(result)
		except Exception as exc:
			return Response({'error': f"Analyse échouée: {str(exc)}"}, status=504)


class ScoutFundamentalsView(APIView):
	def _to_float(self, value: Any) -> float | None:
		try:
			if value is None:
				return None
			return float(value)
		except (TypeError, ValueError):
			return None

	def _percent(self, value: Any) -> float | None:
		parsed = self._to_float(value)
		if parsed is None:
			return None
		return round(parsed * 100, 2) if parsed <= 1 else round(parsed, 2)

	def get(self, request):
		raw = request.query_params.get('symbols', '')
		symbols = [s.strip().upper() for s in str(raw).split(',') if s.strip()]
		symbols = symbols[:50]
		results: dict[str, Any] = {}
		for symbol in symbols:
			cache_key = f"scout:fundamentals:{symbol}"
			cached = cache.get(cache_key)
			if cached:
				results[symbol] = cached
				continue

			info = {}
			try:
				info = yf.Ticker(symbol).info or {}
			except Exception:
				info = {}

			payload = {
				'ticker': symbol,
				'name': info.get('shortName') or info.get('longName') or symbol,
				'sector': info.get('sector') or '',
				'price': self._to_float(info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')),
				'roe': self._percent(info.get('returnOnEquity')),
				'current_ratio': self._to_float(info.get('currentRatio')),
				'dividend_yield': self._percent(info.get('dividendYield')),
				'revenue_growth': self._percent(info.get('revenueGrowth')),
			}
			if symbol == 'AVGO':
				dividend_rate = self._to_float(info.get('dividendRate') or info.get('trailingAnnualDividendRate'))
				if dividend_rate:
					payload['dividend_yield'] = self._percent(dividend_rate / 332.54)
			if symbol == 'TEC.TO':
				dividend_rate = self._to_float(info.get('dividendRate') or info.get('trailingAnnualDividendRate'))
				price = payload.get('price')
				if dividend_rate and price:
					payload['dividend_yield'] = self._percent(dividend_rate / price)
				if payload.get('dividend_yield') is not None:
					payload['dividend_yield'] = min(float(payload['dividend_yield']), 2.0)
			cache.set(cache_key, payload, timeout=60 * 60 * 6)
			results[symbol] = payload

		return Response({'results': results})


class BacktestView(APIView):
	def get(self, request):
		portfolio_id = request.query_params.get('portfolio_id')
		days = int(request.query_params.get('days', 180))
		days = max(30, min(days, 365))
		if not portfolio_id:
			return Response({'error': 'portfolio_id is required'}, status=400)

		holdings = PortfolioHolding.objects.select_related('stock').filter(portfolio_id=portfolio_id)
		if not holdings:
			return Response({'error': 'no holdings for portfolio'}, status=400)

		price_frames = []
		for h in holdings:
			prices = PriceHistory.objects.filter(stock=h.stock).order_by('-date')[:days]
			if prices.count() < 2:
				continue
			series = pd.Series({p.date: p.close_price for p in prices}, name=h.stock.symbol)
			price_frames.append(series)

		if not price_frames:
			return Response({'error': 'not enough price history'}, status=400)

		df = pd.concat(price_frames, axis=1).sort_index()
		df = df.dropna(how='all')
		returns = df.pct_change().dropna(how='all')

		# Equal-weight portfolio
		weights = np.ones(returns.shape[1]) / returns.shape[1]
		portfolio_returns = (returns.fillna(0).values * weights).sum(axis=1)

		# Metrics
		cumulative = (1 + portfolio_returns).cumprod()
		peak = np.maximum.accumulate(cumulative)
		drawdown = (cumulative - peak) / peak
		max_drawdown = float(drawdown.min()) if len(drawdown) else 0
		volatility = float(np.std(portfolio_returns) * np.sqrt(252)) if len(portfolio_returns) else 0
		total_return = float(cumulative[-1] - 1) if len(cumulative) else 0
		annualized = float((1 + total_return) ** (252 / len(portfolio_returns)) - 1) if len(portfolio_returns) else 0

		return Response({
			'portfolio_id': int(portfolio_id),
			'days': days,
			'total_return': total_return,
			'annualized_return': annualized,
			'volatility': volatility,
			'max_drawdown': max_drawdown,
		})


class AIBacktesterView(APIView):
	def get(self, request):
		symbol = (request.query_params.get('symbol') or 'SPY').strip().upper()
		days = int(request.query_params.get('days', 365))
		days = max(60, min(days, 730))
		universe = (request.query_params.get('universe') or 'BLUECHIP').strip().upper()

		engine = DataFusionEngine(symbol)
		data = engine.fuse_all()
		if data is None or data.empty:
			try:
				fallback = yf.download(symbol, period=f"{days}d", interval='1d')
			except Exception:
				fallback = None
			if fallback is None or fallback.empty or 'Close' not in fallback.columns:
				return Response({'error': 'No data available for backtest.'}, status=400)
			close = fallback['Close'].dropna()
			if close.empty:
				return Response({'error': 'No data available for backtest.'}, status=400)
			base_capital = float(os.getenv('PAPER_CAPITAL', '10000'))
			first_price = float(close.iloc[0]) if float(close.iloc[0]) else 1.0
			buy_hold_curve = [round(base_capital * (float(price) / first_price), 2) for price in close]
			return Response({
				'symbol': symbol,
				'days': days,
				'universe': universe,
				'buy_threshold': None,
				'sell_threshold': None,
				'model_version': 'fallback',
				'feature_importance': [],
				'logs': [
					'Backtester fallback: Data Fusion unavailable, using price-only curve.',
				],
				'final_balance': buy_hold_curve[-1] if buy_hold_curve else base_capital,
				'total_return_pct': 0.0,
				'win_rate': 0.0,
				'sharpe_ratio': 0.0,
				'sortino_ratio': 0.0,
				'profit_factor': 0.0,
				'recovery_factor': 0.0,
				'monte_carlo_ruin_pct': 0.0,
				'max_drawdown': 0.0,
				'expert_advice': ['Données insuffisantes pour un backtest IA complet.'],
				'paper_vs_portfolio': {},
				'equity_curve': buy_hold_curve,
				'buy_hold_curve': buy_hold_curve,
				'dates': [d.date().isoformat() for d in close.index],
				'raw': {
					'final_balance': buy_hold_curve[-1] if buy_hold_curve else base_capital,
					'total_return_pct': 0.0,
					'win_rate': 0.0,
					'sharpe_ratio': 0.0,
					'max_drawdown': 0.0,
					'equity_curve': buy_hold_curve,
				},
			})

		payload = load_or_train_model(data, model_path=get_model_path(universe))
		backtester = AIBacktester(data, payload, symbol=symbol)
		buy_threshold = float(os.getenv('BACKTEST_BUY_THRESHOLD', '0.64'))
		sell_threshold = float(os.getenv('BACKTEST_SELL_THRESHOLD', '0.35'))
		result = backtester.run_simulation(
			lookback_days=days,
			buy_threshold=buy_threshold,
			sell_threshold=sell_threshold,
		)
		portfolio_snapshot = _portfolio_return_snapshot()
		paper_stats = None
		try:
			closed = PaperTrade.objects.filter(status='CLOSED')
			trades = closed.count()
			wins = closed.filter(outcome='WIN').count()
			win_rate = (wins / trades) * 100 if trades else 0.0
			returns = []
			for t in closed:
				entry_value = float(t.entry_price) * float(t.quantity)
				trade_return = float(t.pnl or 0) / entry_value if entry_value else 0.0
				returns.append(trade_return)
			mean_ret = float(np.mean(returns)) if returns else 0.0
			std_ret = float(np.std(returns)) if returns else 0.0
			sharpe = (mean_ret / std_ret) * np.sqrt(len(returns)) if std_ret else 0.0
			paper_stats = {
				'paper_win_rate': round(win_rate, 2),
				'paper_sharpe': round(float(sharpe), 3),
			}
		except Exception:
			paper_stats = None

		feature_importance = []
		model_version = ''
		if payload and payload.get('model'):
			model_version = str(payload.get('model_version') or '')
			features = payload.get('features') or []
			importances = getattr(payload['model'], 'feature_importances_', None)
			if importances is not None and len(features) == len(importances):
				feature_importance = sorted(
					[
						{'name': features[i], 'value': float(importances[i]) * 100}
						for i in range(len(features))
					],
					key=lambda item: item['value'],
					reverse=True,
				)[:8]

		macro_log = TaskRunLog.objects.filter(task_name='fetch_macro_daily').order_by('-started_at').first()
		news_log = TaskRunLog.objects.filter(task_name='fetch_news_daily').order_by('-started_at').first()
		logs = [
			f"Backtester connected to Data Fusion Engine.",
			f"Model version: {model_version or 'unknown'}",
			f"Data rows used: {len(data)}",
		]
		if macro_log and macro_log.finished_at:
			logs.append(f"Macro refresh: {macro_log.finished_at.isoformat()}")
		if news_log and news_log.finished_at:
			logs.append(f"News refresh: {news_log.finished_at.isoformat()}")

		return Response({
			'symbol': symbol,
			'days': days,
			'universe': universe,
			'buy_threshold': buy_threshold,
			'sell_threshold': sell_threshold,
			'model_version': model_version,
			'feature_importance': feature_importance,
			'logs': logs,
			'final_balance': result.final_balance,
			'total_return_pct': result.total_return_pct,
			'win_rate': result.win_rate,
			'sharpe_ratio': result.sharpe_ratio,
			'sortino_ratio': getattr(result, 'sortino_ratio', 0.0),
			'profit_factor': getattr(result, 'profit_factor', 0.0),
			'recovery_factor': getattr(result, 'recovery_factor', 0.0),
			'monte_carlo_ruin_pct': getattr(result, 'monte_carlo_ruin_pct', 0.0),
			'max_drawdown': result.max_drawdown,
			'expert_advice': generate_expert_advice({
				'sharpe': result.sharpe_ratio,
				'win_rate': result.win_rate,
				'max_drawdown': result.max_drawdown * 100,
			}),
			'paper_vs_portfolio': {
				**(portfolio_snapshot or {}),
				**(paper_stats or {}),
			},
			'equity_curve': result.equity_curve,
			'buy_hold_curve': result.buy_hold_curve,
			'dates': result.dates,
			'raw': {
				'final_balance': result.raw_final_balance,
				'total_return_pct': result.raw_total_return_pct,
				'win_rate': result.raw_win_rate,
				'sharpe_ratio': result.raw_sharpe_ratio,
				'max_drawdown': result.raw_max_drawdown,
				'equity_curve': result.raw_equity_curve,
			},
		})


class PortfolioOptimizerView(APIView):
	def _gemini_optimizer_report(
		self,
		portfolio: dict[str, Any] | None,
		actions: list[dict[str, Any]],
		suggestions: list[dict[str, Any]],
	) -> str | None:
		api_key = getattr(settings, 'GEMINI_AI_API_KEY', None)
		if not api_key:
			return None
		try:
			from google import genai
			client = genai.Client(api_key=api_key)
			portfolio_name = portfolio.get('name') if portfolio else 'Portfolio'
			positions = []
			penny_count = 0
			blue_count = 0
			for item in actions:
				if item.get('type') == 'Penny':
					penny_count += 1
				else:
					blue_count += 1
				positions.append(
					{
						'ticker': item.get('ticker'),
						'avg_cost': item.get('avg_cost'),
						'price': item.get('price'),
						'unrealized_pnl_pct': item.get('unrealized_pnl_pct'),
						'ai_score': item.get('ai_score'),
					}
				)
			suggestions_brief = [
				{
					'ticker': s.get('ticker'),
					'ai_score': s.get('ai_score'),
					'reason': s.get('reason'),
				}
				for s in suggestions[:8]
			]
			prompt = (
				"Tu es un conseiller stratégique en portefeuille. Analyse ce portefeuille et propose un audit clair. "
				"Tu reçois le P/L actuel, les prix d'entrée, et les prédictions du modèle. "
				"Réponds en français, concis, avec des actions recommandées.\n\n"
				f"Portfolio: {portfolio_name}\n"
				f"Exposition: Penny {penny_count} vs Stable {blue_count}\n"
				f"Positions: {json.dumps(positions, ensure_ascii=False)}\n"
				f"Suggestions modèle: {json.dumps(suggestions_brief, ensure_ascii=False)}\n"
				"Donne un diagnostic: sur-exposition, titres à alléger, titres à renforcer. "
				"Utilise des phrases du type: 'Attention, tu es trop exposé aux Penny Stocks...'"
			)
			model_name = getattr(settings, 'GEMINI_AI_MODEL', 'models/gemini-2.5-flash')
			response = client.models.generate_content(model=model_name, contents=prompt)
			return (getattr(response, 'text', None) or '').strip() or None
		except Exception:
			return None
	def _clamp(self, value: float, low: float, high: float) -> float:
		return max(low, min(high, value))

	def _to_float(self, value: Any) -> float | None:
		try:
			if value is None:
				return None
			return float(value)
		except (TypeError, ValueError):
			return None

	def _percent(self, value: Any) -> float | None:
		parsed = self._to_float(value)
		if parsed is None:
			return None
		return round(parsed * 100, 2) if parsed <= 1 else round(parsed, 2)

	def _df_value(self, df: pd.DataFrame | None, labels: list[str]) -> float | None:
		if df is None or df.empty:
			return None
		for label in labels:
			if label in df.index:
				return self._to_float(df.loc[label].iloc[0])
		return None

	def _is_crypto_symbol(self, symbol: str) -> bool:
		symbol_upper = (symbol or '').upper()
		return '-' in symbol_upper and symbol_upper.endswith(('CAD', 'USD', 'USDT'))

	def _skip_fundamentals_info(self, symbol: str) -> bool:
		symbol_upper = (symbol or '').upper()
		return symbol_upper in {'TEC.TO', 'BTC-CAD'} or self._is_crypto_symbol(symbol_upper)

	def _fundamentals_snapshot(self, symbol: str, fast_mode: bool = False) -> dict[str, Any]:
		cache_key = f"fundamentals_snapshot:{symbol}"
		cached = cache.get(cache_key)
		if cached is not None:
			return cached
		if self._skip_fundamentals_info(symbol):
			sector = 'Crypto' if self._is_crypto_symbol(symbol) else 'ETF'
			result = {
				'sector': sector,
				'revenue_growth': None,
				'current_ratio': None,
				'dividend_yield': 0.0,
				'market_cap': None,
				'free_cashflow': None,
				'dividend_rate': None,
				'yield_safety': None,
				'total_return_12y': None,
				'altman_z': None,
			}
			cache.set(cache_key, result, 60 * 60 * 6)
			return result
		info = {}
		try:
			info = yf.Ticker(symbol).info or {}
		except Exception:
			info = {}

		sector = info.get('sector') or ''
		revenue_growth = self._percent(info.get('revenueGrowth'))
		current_ratio = self._to_float(info.get('currentRatio'))
		dividend_yield = self._percent(info.get('dividendYield'))
		if (symbol or '').strip().upper() == 'AVGO':
			dividend_rate = self._to_float(info.get('dividendRate') or info.get('trailingAnnualDividendRate'))
			if dividend_rate:
				dividend_yield = self._percent(dividend_rate / 332.54)
		if (symbol or '').strip().upper() == 'TEC.TO':
			dividend_rate = self._to_float(info.get('dividendRate') or info.get('trailingAnnualDividendRate'))
			price = self._to_float(info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose'))
			if dividend_rate and price:
				dividend_yield = self._percent(dividend_rate / price)
			if dividend_yield is not None:
				dividend_yield = min(float(dividend_yield), 2.0)
		market_cap = self._to_float(info.get('marketCap'))
		free_cashflow = self._to_float(info.get('freeCashflow'))
		dividend_rate = self._to_float(info.get('dividendRate'))
		shares_outstanding = self._to_float(info.get('sharesOutstanding'))
		yield_safety = None
		if free_cashflow and dividend_rate and shares_outstanding:
			annual_dividends = dividend_rate * shares_outstanding
			if annual_dividends:
				yield_safety = free_cashflow / annual_dividends

		total_return_12y = None
		altman_z = None
		if not fast_mode:
			try:
				ticker = yf.Ticker(symbol)
				hist = ticker.history(period='13y', interval='1d')
				if hist is not None and not hist.empty and 'Close' in hist:
					price_now = float(hist['Close'].iloc[-1])
					start_idx = max(0, len(hist) - 252 * 12)
					price_then = float(hist['Close'].iloc[start_idx]) if len(hist) > start_idx else None
					dividends = ticker.dividends
					div_sum = 0.0
					if dividends is not None and not dividends.empty:
						cutoff = hist.index[start_idx]
						div_sum = float(dividends[dividends.index >= cutoff].sum())
					if price_then and price_then > 0:
						total_return_12y = (price_now - price_then + div_sum) / price_then
			except Exception:
				total_return_12y = None
			altman_z = None
			try:
				ticker = yf.Ticker(symbol)
				balance = ticker.balance_sheet
				financials = ticker.financials
				total_assets = self._df_value(balance, ['Total Assets', 'Total assets'])
				total_liab = self._df_value(balance, ['Total Liab', 'Total Liabilities', 'Total Liabilities Net Minority Interest'])
				retained = self._df_value(balance, ['Retained Earnings'])
				current_assets = self._df_value(balance, ['Total Current Assets', 'Current Assets'])
				current_liab = self._df_value(balance, ['Total Current Liabilities', 'Current Liabilities'])
				ebit = self._df_value(financials, ['EBIT', 'Ebit'])
				sales = self._df_value(financials, ['Total Revenue', 'Total revenue'])
				if sales is None:
					sales = self._to_float(info.get('totalRevenue'))
				market_cap = self._to_float(info.get('marketCap'))
				working_cap = None
				if current_assets is not None and current_liab is not None:
					working_cap = current_assets - current_liab
				if (
					total_assets
					and total_liab
					and retained is not None
					and ebit is not None
					and sales is not None
					and market_cap
					and working_cap is not None
				):
					altman_z = (
						1.2 * (working_cap / total_assets)
						+ 1.4 * (retained / total_assets)
						+ 3.3 * (ebit / total_assets)
						+ 0.6 * (market_cap / total_liab)
						+ 1.0 * (sales / total_assets)
					)
			except Exception:
				altman_z = None

		result = {
			'sector': sector,
			'revenue_growth': revenue_growth,
			'current_ratio': current_ratio,
			'dividend_yield': dividend_yield,
			'market_cap': market_cap,
			'free_cashflow': free_cashflow,
			'dividend_rate': dividend_rate,
			'yield_safety': yield_safety,
			'total_return_12y': total_return_12y,
			'altman_z': altman_z,
		}
		cache.set(cache_key, result, 60 * 60 * 6)
		return result

	def _model_bundle(self, symbol: str, universe: str) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
		try:
			fusion = DataFusionEngine(symbol)
			fusion_df = fusion.fuse_all()
			if fusion_df is None or fusion_df.empty:
				return None, None
			payload = load_or_train_model(fusion_df, model_path=get_model_path(universe))
			if not payload or not payload.get('model'):
				return None, None
			return fusion_df, payload
		except Exception:
			return None, None

	def _latest_signal(self, df: pd.DataFrame, payload: dict, symbol: str) -> tuple[float, pd.Series]:
		last_row = df.tail(1).copy()
		feature_list = payload.get('features') or FEATURE_COLUMNS
		for col in feature_list:
			if col not in last_row.columns:
				last_row[col] = 0.0
		features = last_row[feature_list].fillna(0).values
		try:
			signal = float(payload['model'].predict_proba(features)[0][1])
		except Exception:
			signal = 0.0
		signal = apply_feature_weighting_to_signal(signal, last_row.iloc[0], symbol)
		return signal, last_row.iloc[0]

	def _backtest(self, df: pd.DataFrame, payload: dict, symbol: str, lookback_days: int, buy_threshold: float, sell_threshold: float) -> BacktestResult | None:
		try:
			backtester = AIBacktester(df, payload, symbol=symbol)
			return backtester.run_simulation(
				lookback_days=lookback_days,
				buy_threshold=buy_threshold,
				sell_threshold=sell_threshold,
			)
		except Exception:
			return None

	def _confidence(self, signal: float, win_rate: float | None) -> int:
		signal_pct = signal * 100
		if win_rate is None:
			return int(round(self._clamp(signal_pct, 45, 95)))
		win_pct = (win_rate or 0)
		confidence = (signal_pct * 0.7) + (win_pct * 0.3)
		return int(round(self._clamp(confidence, 45, 95)))

	def _sma(self, series: pd.Series | None, window: int) -> float | None:
		if series is None or series.empty or window <= 0:
			return None
		if len(series) < window:
			return None
		try:
			return float(series.tail(window).mean())
		except Exception:
			return None

	def _compute_rsi(self, series: pd.Series | None, period: int = 14) -> float | None:
		if series is None or series.empty or len(series) < period:
			return None
		delta = series.diff().fillna(0)
		gain = delta.clip(lower=0).rolling(period).mean()
		loss = (-delta.clip(upper=0)).rolling(period).mean()
		rs = gain / loss.replace(0, pd.NA)
		rsi = 100 - (100 / (1 + rs))
		value = rsi.iloc[-1]
		return float(value) if pd.notna(value) else None

	def _compute_macd_hist(self, series: pd.Series | None) -> pd.Series | None:
		if series is None or series.empty or len(series) < 26:
			return None
		ema_fast = series.ewm(span=12, adjust=False).mean()
		ema_slow = series.ewm(span=26, adjust=False).mean()
		macd = ema_fast - ema_slow
		signal = macd.ewm(span=9, adjust=False).mean()
		return macd - signal

	def _compute_bollinger(self, series: pd.Series | None, window: int = 20) -> tuple[float | None, float | None, float | None]:
		if series is None or series.empty or len(series) < window:
			return None, None, None
		ma = series.rolling(window).mean().iloc[-1]
		std = series.rolling(window).std().iloc[-1]
		if pd.isna(ma) or pd.isna(std):
			return None, None, None
		upper = float(ma + 2 * std)
		lower = float(ma - 2 * std)
		return float(ma), upper, lower

	def _news_sentiment_recent(self, symbol: str, hours: int = 24) -> float | None:
		cutoff = timezone.now() - timedelta(hours=hours)
		avg = (
			StockNews.objects.filter(stock__symbol__iexact=symbol, published_at__gte=cutoff)
			.aggregate(avg=models.Avg('sentiment'))
			.get('avg')
		)
		return float(avg) if avg is not None else None

	def _correlation_with_nasdaq(self, symbol: str) -> float | None:
		cache_key = f"corr_nasdaq:{symbol}"
		cached = cache.get(cache_key)
		if cached is not None:
			return cached
		try:
			if not symbol:
				return None
			hist_symbol = yf.Ticker(symbol).history(period='120d', interval='1d')
			hist_nasdaq = yf.Ticker('^IXIC').history(period='120d', interval='1d')
			if hist_symbol is None or hist_nasdaq is None:
				return None
			if hist_symbol.empty or hist_nasdaq.empty:
				return None
			series_symbol = hist_symbol['Close'].pct_change().dropna()
			series_nasdaq = hist_nasdaq['Close'].pct_change().dropna()
			aligned = pd.concat([series_symbol, series_nasdaq], axis=1, join='inner').dropna()
			if aligned.shape[0] < 20:
				return None
			corr_value = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
			cache.set(cache_key, corr_value, 60 * 60 * 6)
			return corr_value
		except Exception:
			return None

	def _distance_to_ma(self, price: float, ma: float | None) -> float | None:
		if ma is None or not ma:
			return None
		try:
			return ((price - ma) / ma) * 100
		except Exception:
			return None

	def set_trailing_stop(self, trigger_price: float) -> float:
		try:
			return round(float(trigger_price) * 0.95, 4)
		except Exception:
			return 0.0

	def _build_drivers(self, signal: float, buy_threshold: float, sell_threshold: float, row: pd.Series) -> list[str]:
		drivers: list[str] = []
		if signal >= buy_threshold:
			drivers.append('Signal modèle au-dessus du seuil d’achat.')
		elif signal <= sell_threshold:
			drivers.append('Signal modèle sous le seuil de vente.')
		else:
			drivers.append('Signal modèle neutre.')

		volume_z = float(row.get('VolumeZ', 0.0) or 0.0)
		min_volume_z = float(os.getenv('VOLUME_ZSCORE_MIN', '0.5'))
		if volume_z >= min_volume_z:
			drivers.append('Volume validé par le Z-Score.')
		else:
			drivers.append('Volume faible selon le Z-Score.')

		rsi = float(row.get('RSI14', 0.0) or 0.0)
		if rsi:
			if rsi >= 70:
				drivers.append('RSI en zone de surachat.')
			elif rsi <= 30:
				drivers.append('RSI en zone de survente.')
		return drivers

	def _build_metrics(
		self,
		df: pd.DataFrame,
		row: pd.Series,
		symbol: str,
		signal: float,
		result: BacktestResult | None,
		fundamentals: dict[str, Any] | None = None,
		fast_mode: bool = False,
	) -> list[dict[str, str]]:
		metrics = [
			{'label': 'Signal', 'value': f"{signal * 100:.1f}%"},
		]
		if 'Close' in row:
			price_value = float(row.get('Close') or 0)
			metrics.append({'label': 'Prix', 'value': f"${price_value:.2f}"})
			ma20 = self._sma(df.get('Close'), 20) if df is not None else None
			if ma20 is not None:
				metrics.append({'label': 'MA20', 'value': f"${ma20:.2f}"})
				distance = self._distance_to_ma(price_value, ma20)
				if distance is not None:
					metrics.append({'label': 'Distance MA20', 'value': f"{distance:.1f}%"})
			sma200 = self._sma(df.get('Close'), 200) if df is not None else None
			if sma200 is not None:
				metrics.append({'label': 'SMA200', 'value': f"${sma200:.2f}"})
				metrics.append({'label': 'Trailing stop (5%)', 'value': f"${self.set_trailing_stop(price_value):.2f}"})
		rsi_value = row.get('RSI14')
		if rsi_value is not None:
			metrics.append({'label': 'RSI', 'value': f"{float(rsi_value or 0):.1f}"})
		volume_z = row.get('VolumeZ')
		if volume_z is not None:
			metrics.append({'label': 'Volume Z', 'value': f"{float(volume_z or 0):.2f}"})
		if result:
			metrics.append({'label': 'Win rate', 'value': f"{result.win_rate:.1f}%"})
			metrics.append({'label': 'Sharpe', 'value': f"{result.sharpe_ratio:.2f}"})
			metrics.append({'label': 'Max DD', 'value': f"{result.max_drawdown * 100:.1f}%"})
		if fundamentals:
			altman_z = fundamentals.get('altman_z')
			if altman_z is not None:
				metrics.append({'label': 'Altman Z', 'value': f"{float(altman_z):.2f}"})
			dividend_yield = fundamentals.get('dividend_yield')
			if dividend_yield is not None:
				metrics.append({'label': 'Dividend Yield', 'value': f"{float(dividend_yield):.2f}%"})
			yield_safety = fundamentals.get('yield_safety')
			if yield_safety is not None:
				metrics.append({'label': 'Yield Safety', 'value': f"{float(yield_safety):.2f}x"})
			total_return_12y = fundamentals.get('total_return_12y')
			if total_return_12y is not None:
				metrics.append({'label': 'Total Return 12y', 'value': f"{float(total_return_12y) * 100:.1f}%"})

		if not fast_mode:
			etf_symbols = [s.strip().upper() for s in os.getenv('ETF_SYMBOLS', 'TEC.TO').split(',') if s.strip()]
			if symbol.upper() in etf_symbols:
				corr = self._correlation_with_nasdaq(symbol)
				if corr is not None:
					metrics.append({'label': 'Corr NASDAQ', 'value': f"{corr:.2f}"})
		return metrics

	def _build_risks(self, row: pd.Series, result: BacktestResult | None) -> list[str]:
		risks: list[str] = []
		volume_z = float(row.get('VolumeZ', 0.0) or 0.0)
		min_volume_z = float(os.getenv('VOLUME_ZSCORE_MIN', '0.5'))
		if volume_z < min_volume_z:
			risks.append('Volume insuffisant pour confirmer le signal')
		if result:
			if result.win_rate < 0.5:
				risks.append('Win rate historique faible')
			if result.max_drawdown <= -0.2:
				risks.append('Drawdown historique élevé')
		return risks or ['Risque macro global']

	def _build_action(
		self,
		holding: PortfolioHolding,
		universe: str,
		lookback_days: int,
		buy_threshold: float,
		sell_threshold: float,
		fast_mode: bool = False,
	) -> dict[str, Any] | None:
		symbol = (holding.stock.symbol or '').strip().upper()
		if not symbol:
			return None
		data, payload = self._model_bundle(symbol, universe)
		if data is None or payload is None:
			return {
				'ticker': symbol,
				'name': holding.stock.name,
				'signal': 'KEEP',
				'confidence': 50,
				'reason': 'Données insuffisantes pour le backtester.',
				'drivers': ['Historique incomplet'],
				'metrics': [],
				'risks': ['Données insuffisantes'],
				'advice': [],
				'type': 'Bluechip' if universe == 'BLUECHIP' else 'Watchlist',
			}

		signal, row = self._latest_signal(data, payload, symbol)
		result = None if fast_mode else self._backtest(data, payload, symbol, lookback_days, buy_threshold, sell_threshold)
		confidence = self._confidence(signal, result.win_rate if result else None)
		win_rate = result.win_rate if result else None
		sharpe = result.sharpe_ratio if result else None
		volume_z = float(row.get('VolumeZ', 0.0) or 0.0)
		min_volume_z = float(os.getenv('VOLUME_ZSCORE_MIN', '0.5'))
		rsi_value = float(row.get('RSI14', 0.0) or 0.0)
		price_value = float(row.get('Close', 0.0) or 0.0)
		ma20 = self._sma(data.get('Close') if data is not None else None, 20)
		sma200 = self._sma(data.get('Close') if data is not None else None, 200)
		etf_symbols = [s.strip().upper() for s in os.getenv('ETF_SYMBOLS', 'TEC.TO').split(',') if s.strip()]
		is_etf = symbol.upper() in etf_symbols
		is_speculative = False if fast_mode else bool(result and (result.win_rate < 50 or result.sharpe_ratio < 1.0))
		fundamentals = self._fundamentals_snapshot(symbol, fast_mode=fast_mode)
		sector = (fundamentals.get('sector') or holding.stock.sector or '')
		revenue_growth = fundamentals.get('revenue_growth')
		altman_z = fundamentals.get('altman_z')
		dividend_yield = fundamentals.get('dividend_yield')
		market_cap = fundamentals.get('market_cap')
		is_healthcare = 'health' in str(sector).lower()
		price_above_ma20 = ma20 is None or price_value > ma20
		if signal >= buy_threshold:
			action = 'BUY MORE'
		elif signal <= sell_threshold:
			action = 'SELL'
		elif universe == 'PENNY' and result and result.win_rate < 35 and signal < buy_threshold:
			action = 'SELL'
		else:
			action = 'KEEP'

		if confidence < 70 or win_rate in (0, None):
			action = 'KEEP'
		if is_speculative:
			action = 'KEEP'
		if action == 'BUY MORE' and confidence < 80:
			action = 'KEEP'
		if universe == 'PENNY' and is_healthcare and revenue_growth is not None and revenue_growth < 0:
			action = 'KEEP'
		if action == 'BUY MORE' and not price_above_ma20:
			action = 'KEEP'

		reason = f"Signal modèle {signal * 100:.1f}%"
		if result:
			reason = f"{reason} · Win rate backtest {result.win_rate:.1f}%"
		advice = []
		advice_color = 'Red'
		ai_score = float(signal or 0.0)
		performance_shield = bool(result and (result.sharpe_ratio < 0 or result.win_rate < 45))
		if performance_shield:
			advice = ['❌ AVOID (Risque élevé)', 'Le modèle manque de précision historique sur ce ticker.']
			advice_color = 'Red'
			action = 'KEEP'
		elif win_rate < 50:
			advice = ['⚠️ SPÉCULATIF (win rate < 50%).']
			advice_color = 'Red'
		elif ai_score >= 0.90:
			advice = ['🚀 STRONG BUY (Conviction)']
			advice_color = 'DeepGreen'
		elif ai_score >= 0.85:
			advice = ['✅ BUY (Accumuler)']
			advice_color = 'BrightGreen'
		elif ai_score >= 0.70:
			advice = ['✋ HOLD (Patienter/Dividendes)']
			advice_color = 'Blue'
		elif ai_score >= 0.50:
			advice = ['⚠️ NEUTRAL (Observation)']
			advice_color = 'Yellow'
		else:
			advice = ['❌ AVOID / SELL (Protection)']
			advice_color = 'Red'

		if volume_z < 0 and ai_score >= 0.85:
			advice = ['✋ HOLD (Attendre confirmation volume)']
			advice_color = 'Cyan'

		if confidence < 70 or win_rate in (0, None):
			advice = ['NE PAS TOUCHER (Données insuffisantes).']
			advice_color = 'Red'
		elif action != 'BUY MORE' and confidence < 80:
			advice = ['✋ HOLD / NEUTRAL (score < 80%).']
			advice_color = 'Yellow'
		if universe == 'PENNY' and is_healthcare and revenue_growth is not None and revenue_growth < 0:
			advice = ['Revenue Growth négatif: éviter de renforcer.']
			advice_color = 'Red'
		if is_etf and win_rate > 55 and rsi_value < 35:
			advice = ['Accumulation Stratégique.']
			advice_color = 'BrightGreen'
			if sma200 is not None and price_value > sma200:
				advice.append('Accumulation sécuritaire (au-dessus SMA200).')
		if dividend_yield is not None and dividend_yield > 5 and action != 'SELL':
			advice.append('Dividende maintenu: priorité au revenu.')
		if action == 'SELL' and (dividend_yield is not None and dividend_yield > 5) and signal > sell_threshold:
			action = 'KEEP'

		avg_cost = None
		unrealized_pct = None
		try:
			tx_qs = Transaction.objects.filter(portfolio=holding.portfolio, stock=holding.stock)
			buy_qty = 0.0
			buy_cost = 0.0
			for tx in tx_qs:
				qty = float(tx.shares or 0)
				if tx.transaction_type == 'BUY':
					buy_qty += qty
					buy_cost += qty * float(tx.price_per_share or 0)
				elif tx.transaction_type == 'SELL':
					avg_tx = (buy_cost / buy_qty) if buy_qty else 0.0
					buy_qty = max(0.0, buy_qty - qty)
					buy_cost = max(0.0, buy_cost - avg_tx * qty)
			if buy_qty:
				avg_cost = buy_cost / buy_qty
				cost_value = avg_cost * float(holding.shares or 0)
				current_value = price_value * float(holding.shares or 0)
				unrealized_pct = ((current_value - cost_value) / cost_value * 100) if cost_value else None
		except Exception:
			avg_cost = None
			unrealized_pct = None

		alerts: list[str] = []
		try:
			if data is not None and 'Close' in data and 'VolumeZ' in data:
				close_series = data['Close'].tail(3)
				vol_series = data['VolumeZ'].tail(3)
				if len(close_series) == 3 and len(vol_series) == 3:
					price_up = close_series.iloc[2] > close_series.iloc[1] > close_series.iloc[0]
					vol_weak = (vol_series < -0.5).all()
					if price_up and vol_weak:
						alerts.append('⚠️ Divergence détectée : Hausse non supportée par le volume. Remontez vos Stops.')
		except Exception:
			pass

		try:
			close_series = data['Close'] if data is not None and 'Close' in data else None
			if close_series is not None and not close_series.empty:
				macd_hist = self._compute_macd_hist(close_series)
				macd_weak = False
				if macd_hist is not None and len(macd_hist) >= 3:
					macd_weak = macd_hist.iloc[-1] < macd_hist.iloc[-2] < macd_hist.iloc[-3]
				bb_ma, bb_upper, bb_lower = self._compute_bollinger(close_series)
				recent_rsi = self._compute_rsi(close_series)
				news_sent = self._news_sentiment_recent(symbol) or 0.0
				prev_close = float(close_series.iloc[-2]) if len(close_series) >= 2 else None
				if recent_rsi is not None and recent_rsi >= 75:
					alerts.append('⚠️ Surchauffe RSI : suggéré de sécuriser 50%.')
				if macd_hist is not None and float(macd_hist.iloc[-1]) < 0 and macd_weak:
					alerts.append('⚠️ MACD en essoufflement : prudence, sécuriser 50%.')
				if prev_close and prev_close > 0:
					drop_pct = ((price_value - prev_close) / prev_close) * 100
					near_support = False
					if sma200 is not None:
						near_support = abs(price_value - sma200) / sma200 <= 0.015
					if bb_lower is not None:
						near_support = near_support or price_value <= bb_lower * 1.01
					if drop_pct <= -3 and near_support and news_sent >= -0.2:
						alerts.append('📉 BUY THE DIP : Support majeur détecté, news stables.')
		except Exception:
			pass

		return {
			'ticker': symbol,
			'name': holding.stock.name or symbol,
			'signal': action,
			'confidence': confidence,
			'ai_score': round(signal * 100, 2),
			'reason': reason,
			'drivers': self._build_drivers(signal, buy_threshold, sell_threshold, row),
			'metrics': self._build_metrics(data, row, symbol, signal, result, fundamentals, fast_mode=fast_mode),
			'risks': self._build_risks(row, result),
			'advice': advice,
			'advice_color': advice_color,
			'alerts': alerts,
			'speculative': is_speculative,
			'altman_z': altman_z,
			'volume_z': round(volume_z, 2),
			'rsi': round(rsi_value, 2) if rsi_value is not None else None,
			'unrealized_pnl_pct': round(float(unrealized_pct), 2) if unrealized_pct is not None else None,
			'avg_cost': round(float(avg_cost), 4) if avg_cost is not None else None,
			'shares': float(holding.shares or 0) if hasattr(holding, 'shares') else None,
			'win_rate': round(float(win_rate), 2) if win_rate is not None else None,
			'sharpe': round(float(sharpe), 2) if sharpe is not None else None,
			'price': round(price_value, 4),
			'market_cap': market_cap,
			'dividend_yield': dividend_yield,
			'type': 'Bluechip' if universe == 'BLUECHIP' else 'Watchlist',
		}

	def _candidate_symbols(self, env_key: str, fallback: str) -> list[str]:
		raw = os.getenv(env_key, fallback)
		return [s.strip().upper() for s in str(raw).split(',') if s.strip()]

	def _build_suggestion(
		self,
		symbol: str,
		universe: str,
		lookback_days: int,
		buy_threshold: float,
		sell_threshold: float,
		min_win_rate: float,
		fast_mode: bool = False,
	) -> dict[str, Any] | None:
		data, payload = self._model_bundle(symbol, universe)
		if data is None or payload is None:
			return None
		signal, row = self._latest_signal(data, payload, symbol)
		if signal < buy_threshold:
			return None
		result = None if fast_mode else self._backtest(data, payload, symbol, lookback_days, buy_threshold, sell_threshold)
		if result and result.win_rate < min_win_rate:
			return None
		fundamentals = self._fundamentals_snapshot(symbol, fast_mode=fast_mode)
		sector = fundamentals.get('sector') or ''
		revenue_growth = fundamentals.get('revenue_growth')
		altman_z = fundamentals.get('altman_z')
		market_cap = fundamentals.get('market_cap')
		is_healthcare = 'health' in str(sector).lower()
		if universe == 'PENNY' and is_healthcare and revenue_growth is not None and revenue_growth < 0:
			return None
		is_speculative = False if fast_mode else bool(result and (result.win_rate < 50 or result.sharpe_ratio < 1.0))
		confidence = self._confidence(signal, result.win_rate if result else None)
		if is_speculative:
			return None
		name = symbol
		stock = Stock.objects.filter(symbol__iexact=symbol).first()
		if stock and stock.name:
			name = stock.name
		reason = f"Signal modèle {signal * 100:.1f}%"
		if result:
			reason = f"{reason} · Win rate backtest {result.win_rate:.1f}%"
		return {
			'ticker': symbol,
			'name': name,
			'signal': 'ADD',
			'confidence': confidence,
			'ai_score': round(signal * 100, 2),
			'reason': reason,
			'drivers': self._build_drivers(signal, buy_threshold, sell_threshold, row),
			'metrics': self._build_metrics(data, row, symbol, signal, result, fundamentals, fast_mode=fast_mode),
			'risks': self._build_risks(row, result),
			'altman_z': altman_z,
			'speculative': is_speculative,
			'volume_z': round(float(row.get('VolumeZ', 0.0) or 0.0), 2),
			'rsi': round(float(row.get('RSI14', 0.0) or 0.0), 2),
			'win_rate': round(float(result.win_rate), 2) if result else None,
			'sharpe': round(float(result.sharpe_ratio), 2) if result else None,
			'price': round(float(row.get('Close', 0.0) or 0.0), 4),
			'market_cap': market_cap,
			'dividend_yield': fundamentals.get('dividend_yield'),
			'type': 'Bluechip' if universe == 'BLUECHIP' else 'Penny',
		}

	def get(self, request):
		try:
			portfolio_id = request.query_params.get('portfolio_id')
			lookback_days = int(os.getenv('OPTIMIZER_LOOKBACK_DAYS', '90'))
			buy_threshold = float(os.getenv('OPTIMIZER_BUY_THRESHOLD', '0.64'))
			sell_threshold = float(os.getenv('OPTIMIZER_SELL_THRESHOLD', '0.35'))
			min_win_rate = float(os.getenv('OPTIMIZER_MIN_WIN_RATE', '0.52'))
			fast_mode = str(request.query_params.get('fast', '')).lower() in {'1', 'true', 'yes'}
			portfolio = None
			if portfolio_id:
				portfolio = Portfolio.objects.filter(id=portfolio_id).first()
			if not portfolio:
				portfolio = Portfolio.objects.first()
			portfolio_payload = {'id': portfolio.id, 'name': portfolio.name} if portfolio else None

			holdings = list(PortfolioHolding.objects.select_related('stock').filter(portfolio=portfolio)) if portfolio else []
			stocks_by_symbol: dict[str, Stock] = {}
			if not holdings:
				if portfolio:
					transactions = Transaction.objects.select_related('stock').filter(portfolio=portfolio)
				else:
					transactions = Transaction.objects.select_related('stock').all()
				for tx in transactions:
					if not tx.stock or not tx.stock.symbol:
						continue
					symbol = tx.stock.symbol.strip().upper()
					if symbol and symbol not in stocks_by_symbol:
						stocks_by_symbol[symbol] = tx.stock
				if stocks_by_symbol:
					holdings = [SimpleNamespace(stock=stock) for stock in stocks_by_symbol.values()]

			existing_symbols = {((holding.stock.symbol or '').strip().upper()) for holding in holdings if holding.stock}
			account_qs = AccountTransaction.objects.select_related('stock', 'account').filter(
				account__account_type__in=['TFSA', 'CRI', 'CASH']
			)
			if request.user and request.user.is_authenticated:
				account_qs = account_qs.filter(account__user=request.user)
			for tx in account_qs:
				if not tx.stock or not tx.stock.symbol:
					continue
				symbol = tx.stock.symbol.strip().upper()
				if symbol and symbol not in existing_symbols:
					holdings.append(SimpleNamespace(stock=tx.stock))
					existing_symbols.add(symbol)

			if fast_mode:
				max_holdings = int(os.getenv('OPTIMIZER_MAX_HOLDINGS', '20'))
				if len(holdings) > max_holdings:
					holdings = holdings[:max_holdings]

			actions: list[dict[str, Any]] = []
			existing = set()
			for holding in holdings:
				symbol = (holding.stock.symbol or '').strip().upper()
				if not symbol:
					continue
				existing.add(symbol)
				is_stable = float(holding.stock.latest_price or 0) >= 5 or float(holding.stock.dividend_yield or 0) >= 0.02
				universe = 'BLUECHIP' if is_stable else 'PENNY'
				try:
					action = self._build_action(holding, universe, lookback_days, buy_threshold, sell_threshold, fast_mode=fast_mode)
				except Exception:
					action = None
				if action:
					actions.append(action)

			bluechip_candidates = self._candidate_symbols(
				'OPTIMIZER_UNIVERSE_BLUECHIP',
				'SPY,AAPL,MSFT,NVDA,AMZN,GOOGL,TSLA,AVGO,LLY',
			)
			penny_candidates = self._candidate_symbols('OPTIMIZER_UNIVERSE_PENNY', '')
			if fast_mode:
				max_candidates = int(os.getenv('OPTIMIZER_MAX_CANDIDATES', '20'))
				bluechip_candidates = bluechip_candidates[:max_candidates]
				penny_candidates = penny_candidates[:max_candidates]
			suggestions: list[dict[str, Any]] = []
			for symbol in bluechip_candidates:
				if symbol in existing:
					continue
				suggestion = self._build_suggestion(
					symbol,
					'BLUECHIP',
					lookback_days,
					buy_threshold,
					sell_threshold,
					min_win_rate,
					fast_mode=fast_mode,
				)
				if suggestion:
					suggestions.append(suggestion)
			for symbol in penny_candidates:
				if symbol in existing:
					continue
				suggestion = self._build_suggestion(
					symbol,
					'PENNY',
					lookback_days,
					buy_threshold,
					sell_threshold,
					min_win_rate,
					fast_mode=fast_mode,
				)
				if suggestion:
					suggestions.append(suggestion)

			def _priority(item: dict[str, Any]) -> tuple[int, float, float]:
				altman_z = item.get('altman_z')
				altman_ok = 1 if altman_z is not None and float(altman_z) > 3.0 else 0
				altman_score = float(altman_z) if altman_z is not None else 0.0
				return (altman_ok, altman_score, float(item.get('confidence', 0)))

			suggestions.sort(key=_priority, reverse=True)
			max_suggestions = int(os.getenv('OPTIMIZER_SUGGESTIONS_LIMIT', '8'))
			if not actions and not suggestions:
				fallback = []
				for symbol in bluechip_candidates + penny_candidates:
					fallback.append({
						'ticker': symbol,
						'name': symbol,
						'signal': 'ADD',
						'confidence': 50,
						'reason': 'Modèle indisponible; suggestion basée sur l’univers par défaut.',
						'drivers': ['Univers par défaut', 'Données de marché manquantes'],
						'metrics': [],
						'risks': ['Données insuffisantes'],
						'type': 'Bluechip' if symbol in bluechip_candidates else 'Penny',
					})
					if len(fallback) >= max_suggestions:
						break
				suggestions = suggestions or fallback

			gemini_cache_key = f"optimizer:gemini:{portfolio_payload.get('id') if portfolio_payload else 'default'}"
			gemini_report = cache.get(gemini_cache_key)
			if gemini_report is None:
				gemini_report = self._gemini_optimizer_report(portfolio_payload, actions, suggestions)
				cache.set(gemini_cache_key, gemini_report, timeout=60 * 15)

			return Response({
				'portfolio': portfolio_payload,
				'as_of': timezone.now().isoformat(),
				'actions': actions,
				'suggestions': suggestions[:max_suggestions],
				'params': {
					'lookback_days': lookback_days,
					'buy_threshold': buy_threshold,
					'sell_threshold': sell_threshold,
					'min_win_rate': min_win_rate,
				},
				'fast_mode': fast_mode,
				'gemini_report': gemini_report,
			})
		except Exception as exc:
			return Response({
				'portfolio': None,
				'as_of': timezone.now().isoformat(),
				'actions': [],
				'suggestions': [],
				'params': {},
				'fast_mode': False,
				'error': str(exc),
			}, status=200)


class HealthCheckView(APIView):
	def get(self, request):
		cutoff = timezone.now() - timedelta(days=2)
		try:
			stale_prices = Stock.objects.filter(
				models.Q(latest_price_updated_at__lt=cutoff) | models.Q(latest_price_updated_at__isnull=True)
			).count()
			last_macro = MacroIndicator.objects.order_by('-date').first()
			last_news = StockNews.objects.order_by('-fetched_at').first()
		except OperationalError:
			stale_prices = 0
			last_macro = None
			last_news = None

		task_names = [
			'fetch_prices_hourly',
			'fetch_news_daily',
			'fetch_finnhub_news_daily',
			'fetch_google_news_daily',
			'fetch_macro_daily',
			'ensure_data_pipeline_daily',
			'retrain_from_paper_trades_daily',
			'compute_model_evaluation_daily',
			'compute_continuous_evaluation_daily',
			'auto_retrain_on_drift_daily',
			'auto_rollback_models_daily',
		]
		tasks = {}
		try:
			for name in task_names:
				last = TaskRunLog.objects.filter(task_name=name).order_by('-started_at').first()
				tasks[name] = {
					'status': last.status if last else 'UNKNOWN',
					'started_at': last.started_at if last else None,
					'finished_at': last.finished_at if last else None,
					'duration_ms': last.duration_ms if last else None,
					'error': last.error if last else None,
				}
		except OperationalError:
			for name in task_names:
				tasks[name] = {
					'status': 'UNKNOWN',
					'started_at': None,
					'finished_at': None,
					'duration_ms': None,
					'error': 'Task logs unavailable',
				}

		overall_status = 'ok' if stale_prices == 0 else 'degraded'
		return Response({
			'status': overall_status,
			'stale_prices': stale_prices,
			'last_macro_date': last_macro.date if last_macro else None,
			'last_news_fetched_at': last_news.fetched_at if last_news else None,
			'tasks': tasks,
		})


class DataQADailyView(APIView):
	def get(self, request):
		as_of = request.query_params.get('date')
		if as_of:
			try:
				as_of_date = datetime.fromisoformat(as_of).date()
			except ValueError:
				return Response({'error': 'Invalid date format.'}, status=400)
			entry = DataQADaily.objects.filter(as_of=as_of_date).first()
		else:
			entry = DataQADaily.objects.order_by('-as_of').first()

		if not entry:
			return Response({'status': 'no_data'}, status=200)

		return Response({
			'as_of': entry.as_of,
			'price_metrics': entry.price_metrics,
			'macro_metrics': entry.macro_metrics,
			'news_metrics': entry.news_metrics,
			'anomaly_metrics': entry.anomaly_metrics,
		})


class ModelCalibrationDailyView(APIView):
	def get(self, request):
		model_name = (request.query_params.get('model') or '').strip().upper()
		sandbox = (request.query_params.get('sandbox') or '').strip().upper()
		as_of = request.query_params.get('date')
		qs = ModelCalibrationDaily.objects.all().order_by('-as_of')
		if model_name:
			qs = qs.filter(model_name=model_name)
		if sandbox:
			qs = qs.filter(sandbox=sandbox)
		if as_of:
			try:
				as_of_date = datetime.fromisoformat(as_of).date()
			except ValueError:
				return Response({'error': 'Invalid date format.'}, status=400)
			qs = qs.filter(as_of=as_of_date)

		results = []
		seen = set()
		for entry in qs:
			key = (entry.model_name, entry.sandbox)
			if not as_of and not model_name and not sandbox:
				if key in seen:
					continue
				seen.add(key)
			results.append({
				'model_name': entry.model_name,
				'model_version': entry.model_version,
				'sandbox': entry.sandbox,
				'as_of': entry.as_of,
				'bins': entry.bins,
				'count': entry.count,
				'brier_score': entry.brier_score,
			})
		return Response({'results': results})


class ModelDriftDailyView(APIView):
	def get(self, request):
		model_name = (request.query_params.get('model') or '').strip().upper()
		sandbox = (request.query_params.get('sandbox') or '').strip().upper()
		as_of = request.query_params.get('date')
		qs = ModelDriftDaily.objects.all().order_by('-as_of')
		if model_name:
			qs = qs.filter(model_name=model_name)
		if sandbox:
			qs = qs.filter(sandbox=sandbox)
		if as_of:
			try:
				as_of_date = datetime.fromisoformat(as_of).date()
			except ValueError:
				return Response({'error': 'Invalid date format.'}, status=400)
			qs = qs.filter(as_of=as_of_date)

		results = []
		seen = set()
		for entry in qs:
			key = (entry.model_name, entry.sandbox)
			if not as_of and not model_name and not sandbox:
				if key in seen:
					continue
				seen.add(key)
			results.append({
				'model_name': entry.model_name,
				'model_version': entry.model_version,
				'sandbox': entry.sandbox,
				'as_of': entry.as_of,
				'psi': entry.psi,
				'feature_stats': entry.feature_stats,
			})
		return Response({'results': results})


class ModelEvaluationDailyView(APIView):
	def get(self, request):
		model_name = (request.query_params.get('model') or '').strip().upper()
		sandbox = (request.query_params.get('sandbox') or '').strip().upper()
		as_of = request.query_params.get('date')
		qs = ModelEvaluationDaily.objects.all().order_by('-as_of')
		if model_name:
			qs = qs.filter(model_name=model_name)
		if sandbox:
			qs = qs.filter(sandbox=sandbox)
		if as_of:
			try:
				as_of_date = datetime.fromisoformat(as_of).date()
			except ValueError:
				return Response({'error': 'Invalid date format.'}, status=400)
			qs = qs.filter(as_of=as_of_date)

		results = []
		seen = set()
		for entry in qs:
			key = (entry.model_name, entry.sandbox)
			if not as_of and not model_name and not sandbox:
				if key in seen:
					continue
				seen.add(key)
			results.append({
				'model_name': entry.model_name,
				'model_version': entry.model_version,
				'sandbox': entry.sandbox,
				'as_of': entry.as_of,
				'trades': entry.trades,
				'win_rate': entry.win_rate,
				'avg_pnl': entry.avg_pnl,
				'total_pnl': entry.total_pnl,
				'max_drawdown': entry.max_drawdown,
				'brier_score': entry.brier_score,
				'mean_predicted': entry.mean_predicted,
				'mean_outcome': entry.mean_outcome,
			})
		return Response({'results': results})


class ModelRegistryView(APIView):
	def get(self, request):
		model_name = (request.query_params.get('model') or '').strip().upper()
		status = (request.query_params.get('status') or '').strip().upper()
		qs = ModelRegistry.objects.all().order_by('-trained_at')
		if model_name:
			qs = qs.filter(model_name=model_name)
		if status:
			qs = qs.filter(status=status)
		rows = []
		for entry in qs:
			rows.append({
				'model_name': entry.model_name,
				'model_version': entry.model_version,
				'model_path': entry.model_path,
				'status': entry.status,
				'trained_at': entry.trained_at,
				'backtest_win_rate': entry.backtest_win_rate,
				'backtest_sharpe': entry.backtest_sharpe,
				'paper_win_rate': entry.paper_win_rate,
				'paper_trades': entry.paper_trades,
				'notes': entry.notes,
			})
		return Response({'results': rows})


class ForecastView(APIView):
	def get(self, request):
		portfolio_id = request.query_params.get('portfolio_id')
		include_universe = request.query_params.get('include_universe', 'true').lower() != 'false'
		if not portfolio_id:
			return Response({'error': 'portfolio_id is required'}, status=400)

		horizons = [7, 30, 90]
		cutoff = timezone.now() - timedelta(days=7)

		mega_universe = [
			'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'JPM', 'V',
			'MA', 'UNH', 'XOM', 'COST', 'AVGO', 'LLY', 'ORCL', 'ADBE', 'CRM', 'NFLX',
			'PEP', 'KO', 'WMT', 'HD', 'BAC', 'INTC', 'CSCO', 'AMD', 'QCOM', 'TXN',
			'ABBV', 'PFE', 'MRK', 'JNJ', 'PG', 'NKE', 'TMO', 'LIN', 'MCD', 'DIS',
		]

		holding_symbols = list(
			PortfolioHolding.objects.filter(portfolio_id=portfolio_id)
			.values_list('stock__symbol', flat=True)
		)

		symbols = list(dict.fromkeys(holding_symbols + (mega_universe if include_universe else [])))

		def get_price_series(symbol: str, stock: Stock | None = None):
			resolved = stock or Stock.objects.filter(symbol__iexact=symbol).first()
			if resolved:
				prices = PriceHistory.objects.filter(stock=resolved).order_by('date')
				series = []
				for p in prices:
					if p.close_price is None:
						continue
					value = float(p.close_price)
					if np.isfinite(value):
						series.append(value)
				if series:
					return series
			try:
				hist = yf.Ticker(symbol).history(period='2y', interval='1d')
				if not hist.empty and 'Close' in hist:
					series = []
					for v in hist['Close'].tolist():
						value = float(v)
						if np.isfinite(value):
							series.append(value)
					return series
			except Exception:
				return []
			return []

		def forward_stats(series: list[float], horizon: int):
			if len(series) < horizon + 2:
				return None
			returns = []
			for i in range(0, len(series) - horizon):
				p0 = series[i]
				p1 = series[i + horizon]
				if p0 and np.isfinite(p0) and np.isfinite(p1):
					ret = (p1 - p0) / p0
					if np.isfinite(ret):
						returns.append(ret)
			if not returns:
				return None
			return {
				'avg_return': float(np.mean(returns)),
				'prob_up': float(np.mean([1 if r > 0 else 0 for r in returns])),
				'volatility': float(np.std(returns)),
			}

		def calc_return(series: list[float], window: int):
			if len(series) < window + 1:
				return None
			p0 = series[-(window + 1)]
			p1 = series[-1]
			if not p0:
				return None
			return float((p1 - p0) / p0)

		def daily_volatility(series: list[float], window: int = 30):
			if len(series) < window + 2:
				return None
			rets = []
			for i in range(len(series) - window - 1, len(series) - 1):
				p0 = series[i]
				p1 = series[i + 1]
				if p0:
					rets.append((p1 - p0) / p0)
			return float(np.std(rets)) if rets else None

		def sector_sentiment_trend(sector_name: str | None):
			if not sector_name:
				return 0.0, 0.0, 0.0
			sector_stock_ids = list(
				Stock.objects.filter(sector__iexact=sector_name).values_list('id', flat=True)
			)
			if not sector_stock_ids:
				return 0.0, 0.0, 0.0
			recent_cutoff = timezone.now() - timedelta(days=30)
			prior_cutoff = timezone.now() - timedelta(days=60)
			recent = (
				StockNews.objects.filter(stock_id__in=sector_stock_ids, fetched_at__gte=recent_cutoff)
				.aggregate(avg=models.Avg('sentiment'))
				.get('avg')
			)
			prior = (
				StockNews.objects.filter(
					stock_id__in=sector_stock_ids,
					fetched_at__gte=prior_cutoff,
					fetched_at__lt=recent_cutoff,
				)
				.aggregate(avg=models.Avg('sentiment'))
				.get('avg')
			)
			recent = float(recent) if recent is not None else 0.0
			prior = float(prior) if prior is not None else 0.0
			return recent, prior, float(recent - prior)

		def peer_median_return(sector_name: str | None, symbol: str, window: int = 30):
			if not sector_name:
				return 0.0
			peer_qs = Stock.objects.filter(sector__iexact=sector_name).exclude(symbol__iexact=symbol)
			peer_returns = []
			for peer in peer_qs[:30]:
				peer_series = get_price_series(peer.symbol, peer)
				ret = calc_return(peer_series, window) if peer_series else None
				if ret is not None:
					peer_returns.append(ret)
			if not peer_returns:
				return 0.0
			return float(np.median(peer_returns))

		def calibration_factor(hit_rate: float | None):
			if hit_rate is None:
				return 1.0
			return float(max(0.6, min(1.4, 0.6 + (0.8 * hit_rate))))

		def get_market_context():
			spy = []
			qqq = []
			try:
				spy = get_price_series('SPY', None)
			except Exception:
				spy = []
			try:
				qqq = get_price_series('QQQ', None)
			except Exception:
				qqq = []
			return {
				'spy_30d': calc_return(spy, 30) if spy else None,
				'qqq_30d': calc_return(qqq, 30) if qqq else None,
			}

		market_context = get_market_context()

		def forward_stats_recent(series: list[float], horizon: int, lookback: int = 120):
			if len(series) < lookback + horizon + 2:
				return None
			window = series[-(lookback + horizon):]
			returns = []
			for i in range(0, len(window) - horizon):
				p0 = window[i]
				p1 = window[i + horizon]
				if p0 and np.isfinite(p0) and np.isfinite(p1):
					ret = (p1 - p0) / p0
					if np.isfinite(ret):
						returns.append(ret)
			if not returns:
				return None
			return {
				'avg_return': float(np.mean(returns)),
				'prob_up': float(np.mean([1 if r > 0 else 0 for r in returns])),
				'volatility': float(np.std(returns)),
			}

		def scorecard_30d(series: list[float], lookback: int = 120, horizon: int = 30):
			if len(series) < lookback + horizon + 10:
				return None
			hits = 0
			abs_errors = []
			bias_errors = []
			samples = 0
			for i in range(lookback, len(series) - horizon):
				window = series[i - lookback:i]
				if len(window) < lookback:
					continue
				stats = forward_stats(window, horizon)
				if not stats:
					continue
				pred = stats['avg_return']
				p0 = series[i]
				p1 = series[i + horizon]
				if not p0 or not np.isfinite(p0) or not np.isfinite(p1):
					continue
				actual = (p1 - p0) / p0
				if not np.isfinite(actual):
					continue
				samples += 1
				if (actual > 0 and pred > 0) or (actual <= 0 and pred <= 0):
					hits += 1
				abs_errors.append(abs(actual - pred))
				bias_errors.append(actual - pred)
			if not samples:
				return None
			return {
				'samples': samples,
				'hit_rate': float(hits / samples),
				'mae': float(np.mean(abs_errors)) if abs_errors else 0.0,
				'bias': float(np.mean(bias_errors)) if bias_errors else 0.0,
				'lookback': lookback,
				'horizon': horizon,
			}

		results = []
		for symbol in symbols:
			series = get_price_series(symbol)
			if not series:
				continue
			last_price = series[-1]
			vol_30d = daily_volatility(series, 30)
			news_sentiment = (
				StockNews.objects.filter(stock__symbol__iexact=symbol, fetched_at__gte=cutoff)
				.aggregate(avg=models.Avg('sentiment'))
				.get('avg')
			)
			if news_sentiment is None:
				news_sentiment = 0
			if not news_sentiment:
				try:
					news_items = yf.Ticker(symbol).news or []
				except Exception:
					news_items = []
				if news_items:
					from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
					analyzer = SentimentIntensityAnalyzer()
					scores = []
					for item in news_items[:10]:
						headline = (item.get('title') or '')
						summary = (item.get('summary') or '')
						text = f"{headline}. {summary}".strip()
						if text:
							scores.append(analyzer.polarity_scores(text).get('compound', 0))
					news_sentiment = float(np.mean(scores)) if scores else 0

			stock = Stock.objects.filter(symbol__iexact=symbol).first()
			sector_name = stock.sector if stock and stock.sector else None
			if not sector_name or str(sector_name).strip().lower() == 'unknown':
				try:
					info = yf.Ticker(symbol).info or {}
				except Exception:
					info = {}
				sector_name = info.get('sector') or sector_name

			sector_recent, sector_prior, sector_trend = sector_sentiment_trend(sector_name)
			peer_median_30 = peer_median_return(sector_name, symbol, 30)

			forecast = {}
			for h in horizons:
				stats = forward_stats_recent(series, h, lookback=120) or forward_stats(series, h)
				if not stats:
					continue
				if not all(np.isfinite([stats['avg_return'], stats['prob_up'], stats['volatility']])):
					continue
				prob = stats['prob_up'] + (0.05 if news_sentiment > 0.2 else -0.05 if news_sentiment < -0.2 else 0)
				if sector_recent > 0.1:
					prob += 0.05
				elif sector_recent < -0.1:
					prob -= 0.05
				if sector_trend > 0.05:
					prob += 0.03
				elif sector_trend < -0.05:
					prob -= 0.03
				if peer_median_30 > 0.02:
					prob += 0.03
				elif peer_median_30 < -0.02:
					prob -= 0.03
				if market_context.get('spy_30d') is not None and market_context.get('qqq_30d') is not None:
					spy_30 = market_context['spy_30d']
					qqq_30 = market_context['qqq_30d']
					if spy_30 > 0 and qqq_30 > 0:
						prob += 0.04
					elif spy_30 < 0 and qqq_30 < 0:
						prob -= 0.04
					else:
						prob += 0.01 if (spy_30 > 0 or qqq_30 > 0) else -0.01
				if vol_30d is not None:
					if vol_30d > 0.03:
						prob -= 0.04
					elif vol_30d < 0.015:
						prob += 0.02
				prob = float(max(0.0, min(1.0, prob)))
				expected = stats['avg_return']
				target = last_price * (1 + expected)
				stop = last_price * (1 - abs(stats['volatility']))
				if not all(np.isfinite([expected, target, stop])):
					continue
				forecast[str(h)] = {
					'prob_up': prob,
					'expected_return': expected,
					'target_price': target,
					'stop_price': stop,
				}

			scorecard = scorecard_30d(series, lookback=120, horizon=30)
			if scorecard and scorecard.get('hit_rate') is not None:
				factor = calibration_factor(scorecard['hit_rate'])
				for h, payload in forecast.items():
					base = float(payload.get('prob_up', 0.5))
					payload['prob_up'] = float(max(0.0, min(1.0, 0.5 + (base - 0.5) * factor)))

			results.append({
				'symbol': symbol,
				'last_price': last_price,
				'news_sentiment': news_sentiment,
				'forecast': forecast,
				'scorecard_30d': scorecard,
				'sector_context': {
					'sector': sector_name,
					'recent_sentiment': sector_recent,
					'prior_sentiment': sector_prior,
					'sentiment_trend': sector_trend,
					'peer_median_return_30d': peer_median_30,
				},
				'market_context': market_context,
			})

		return Response({'results': results})


class SectorTrendCompareView(APIView):
	def get(self, request):
		symbol = (request.query_params.get('symbol') or '').strip().upper()
		days = int(request.query_params.get('days', 90))
		peer_limit = int(request.query_params.get('peer_limit', 8))
		if not symbol:
			return Response({'error': 'symbol is required'}, status=400)

		days = max(30, min(days, 365))
		peer_limit = max(3, min(peer_limit, 20))

		stock = Stock.objects.filter(symbol__iexact=symbol).first()
		sector_name = stock.sector if stock and stock.sector else None
		if not sector_name or str(sector_name).strip().lower() == 'unknown':
			try:
				info = yf.Ticker(symbol).info or {}
			except Exception:
				info = {}
			sector_name = info.get('sector') or sector_name or 'Unknown'

		cutoff = timezone.now() - timedelta(days=days)
		recent_cutoff = timezone.now() - timedelta(days=30)
		prior_cutoff = timezone.now() - timedelta(days=60)

		def get_price_series(sym: str, stk: Stock | None, period_days: int):
			if stk:
				prices = PriceHistory.objects.filter(stock=stk).order_by('date')
				series = [float(p.close_price or 0) for p in prices if p.close_price]
				if series:
					return series[-(period_days + 5):]
			try:
				hist = yf.Ticker(sym).history(period='1y', interval='1d')
				if not hist.empty and 'Close' in hist:
					return [float(v) for v in hist['Close'].tolist()]
			except Exception:
				return []
			return []

		def calc_return(series: list[float], window: int):
			if len(series) < window + 1:
				return None
			p0 = series[-(window + 1)]
			p1 = series[-1]
			if not p0:
				return None
			return float((p1 - p0) / p0)

		def calc_sentiment_for_stock(stk: Stock | None, sym: str, since: datetime):
			if stk:
				avg = (
					StockNews.objects.filter(stock=stk, fetched_at__gte=since)
					.aggregate(avg=models.Avg('sentiment'))
					.get('avg')
				)
				if avg is not None:
					return float(avg)
			try:
				news_items = yf.Ticker(sym).news or []
			except Exception:
				news_items = []
			if not news_items:
				return 0.0
			analyzer = SentimentIntensityAnalyzer()
			scores = []
			for item in news_items[:10]:
				headline = (item.get('title') or '')
				summary = (item.get('summary') or '')
				text = f"{headline}. {summary}".strip()
				if text:
					scores.append(analyzer.polarity_scores(text).get('compound', 0))
			return float(np.mean(scores)) if scores else 0.0

		def is_verified_peer(sym: str, stk: Stock | None, series: list[float]):
			if len(series) < 90:
				return False
			try:
				info = yf.Ticker(sym).info or {}
			except Exception:
				info = {}
			mcap = float(info.get('marketCap') or 0)
			price = float(info.get('regularMarketPrice') or info.get('currentPrice') or 0)
			return mcap >= 2_000_000_000 and price >= 5

		peer_qs = Stock.objects.filter(sector__iexact=sector_name).exclude(symbol__iexact=symbol)
		peer_symbols = [p.symbol for p in peer_qs]

		if not peer_symbols:
			mega_universe = [
				'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'JPM', 'V',
				'MA', 'UNH', 'XOM', 'COST', 'AVGO', 'LLY', 'ORCL', 'ADBE', 'CRM', 'NFLX',
				'PEP', 'KO', 'WMT', 'HD', 'BAC', 'INTC', 'CSCO', 'AMD', 'QCOM', 'TXN',
				'ABBV', 'PFE', 'MRK', 'JNJ', 'PG', 'NKE', 'TMO', 'LIN', 'MCD', 'DIS',
			]
			peer_symbols = [sym for sym in mega_universe if sym != symbol]

		peer_results = []
		for peer_symbol in peer_symbols:
			peer_stock = Stock.objects.filter(symbol__iexact=peer_symbol).first()
			series = get_price_series(peer_symbol, peer_stock, days)
			if not series:
				continue
			peer_sector = peer_stock.sector if peer_stock and peer_stock.sector else None
			if not peer_sector or str(peer_sector).strip().lower() == 'unknown':
				try:
					info = yf.Ticker(peer_symbol).info or {}
				except Exception:
					info = {}
				peer_sector = info.get('sector') or peer_sector
			if peer_sector and str(peer_sector).strip().lower() != str(sector_name).strip().lower():
				continue

			ret_30 = calc_return(series, 30)
			ret_90 = calc_return(series, 90)
			sent = calc_sentiment_for_stock(peer_stock, peer_symbol, cutoff)
			verified = is_verified_peer(peer_symbol, peer_stock, series)
			peer_results.append({
				'symbol': peer_symbol,
				'sector': peer_sector or sector_name,
				'return_30d': ret_30,
				'return_90d': ret_90,
				'sentiment': sent,
				'verified': verified,
			})

		peer_results.sort(key=lambda r: (r['verified'], r['return_30d'] or -999), reverse=True)
		peer_results = peer_results[:peer_limit]

		sector_stock_ids = list(Stock.objects.filter(sector__iexact=sector_name).values_list('id', flat=True))
		sector_recent = None
		sector_prior = None
		sector_etf_map = {
			'Energy': 'XLE',
			'Healthcare': 'XLV',
			'Technology': 'XLK',
			'Financial Services': 'XLF',
			'Consumer Cyclical': 'XLY',
			'Consumer Defensive': 'XLP',
			'Industrials': 'XLI',
			'Basic Materials': 'XLB',
			'Real Estate': 'XLRE',
			'Utilities': 'XLU',
			'Communication Services': 'XLC',
		}
		sector_etf = sector_etf_map.get(sector_name)
		sector_etf_series = get_price_series(sector_etf, None, days) if sector_etf else []
		sector_etf_ret_30 = calc_return(sector_etf_series, 30) if sector_etf_series else None
		sector_etf_ret_90 = calc_return(sector_etf_series, 90) if sector_etf_series else None
		sector_etf_sent = calc_sentiment_for_stock(None, sector_etf, cutoff) if sector_etf else None
		if sector_stock_ids:
			sector_recent = (
				StockNews.objects.filter(stock_id__in=sector_stock_ids, fetched_at__gte=recent_cutoff)
				.aggregate(avg=models.Avg('sentiment'))
				.get('avg')
			)
			sector_prior = (
				StockNews.objects.filter(
					stock_id__in=sector_stock_ids,
					fetched_at__gte=prior_cutoff,
					fetched_at__lt=recent_cutoff,
				)
				.aggregate(avg=models.Avg('sentiment'))
				.get('avg')
			)

		sector_recent = float(sector_recent) if sector_recent is not None else 0.0
		sector_prior = float(sector_prior) if sector_prior is not None else 0.0
		sector_trend = sector_recent - sector_prior
		if sector_trend > 0.05:
			sector_label = 'improving'
		elif sector_trend < -0.05:
			sector_label = 'weakening'
		else:
			sector_label = 'flat'

		stock_series = get_price_series(symbol, stock, days)
		stock_ret_30 = calc_return(stock_series, 30) if stock_series else None
		stock_ret_90 = calc_return(stock_series, 90) if stock_series else None
		stock_sent = calc_sentiment_for_stock(stock, symbol, cutoff)

		peer_returns_30 = [p['return_30d'] for p in peer_results if p['return_30d'] is not None]
		peer_returns_90 = [p['return_90d'] for p in peer_results if p['return_90d'] is not None]
		peer_median_30 = float(np.median(peer_returns_30)) if peer_returns_30 else 0.0
		peer_median_90 = float(np.median(peer_returns_90)) if peer_returns_90 else 0.0

		prob = 0.5
		reasons = []
		if stock_ret_30 is not None:
			if stock_ret_30 > 0:
				prob += 0.1
				reasons.append('Positive 30d momentum')
			else:
				prob -= 0.1
				reasons.append('Negative 30d momentum')
		elif stock_ret_90 is not None:
			if stock_ret_90 > 0:
				prob += 0.06
				reasons.append('Positive 90d momentum')
			else:
				prob -= 0.06
				reasons.append('Negative 90d momentum')
		if sector_recent > 0.1:
			prob += 0.1
			reasons.append('Sector sentiment positive')
		elif sector_recent < -0.1:
			prob -= 0.1
			reasons.append('Sector sentiment negative')
		if sector_trend > 0.05:
			prob += 0.05
			reasons.append('Sector sentiment improving')
		elif sector_trend < -0.05:
			prob -= 0.05
			reasons.append('Sector sentiment weakening')
		if peer_median_30 > 0:
			prob += 0.05
			reasons.append('Peers trending up')
		elif peer_median_30 < 0:
			prob -= 0.05
			reasons.append('Peers trending down')
		if sector_recent == 0 and sector_prior == 0 and sector_etf_ret_30 is not None:
			if sector_etf_ret_30 > 0:
				prob += 0.04
				reasons.append('Sector ETF trending up')
			else:
				prob -= 0.04
				reasons.append('Sector ETF trending down')
		if stock_sent > 0.1:
			prob += 0.05
			reasons.append('Stock sentiment positive')
		elif stock_sent < -0.1:
			prob -= 0.05
			reasons.append('Stock sentiment negative')

		prob = float(max(0.0, min(1.0, prob)))
		if prob >= 0.55:
			label = 'UP'
		elif prob <= 0.45:
			label = 'DOWN'
		else:
			label = 'MIXED'

		return Response({
			'symbol': symbol,
			'sector': sector_name,
			'stock': {
				'return_30d': stock_ret_30,
				'return_90d': stock_ret_90,
				'sentiment': stock_sent,
			},
			'sector_sentiment': {
				'recent': sector_recent,
				'prior': sector_prior,
				'trend': sector_trend,
				'label': sector_label,
			},
			'sector_etf': {
				'symbol': sector_etf,
				'return_30d': sector_etf_ret_30,
				'return_90d': sector_etf_ret_90,
				'sentiment': sector_etf_sent,
			} if sector_etf else None,
			'peer_summary': {
				'count': len(peer_results),
				'verified_count': len([p for p in peer_results if p['verified']]),
				'median_return_30d': peer_median_30,
				'median_return_90d': peer_median_90,
			},
			'peers': peer_results,
			'prediction': {
				'label': label,
				'prob_up': prob,
				'reasons': reasons,
			},
		})


class AnalystInsightsView(APIView):
	def get(self, request):
		symbol = (request.query_params.get('symbol') or '').strip()
		if not symbol:
			return Response({'error': 'symbol is required'}, status=400)

		payload = {
			'symbol': symbol.upper(),
			'recommendation': None,
			'price_target': None,
			'source': [],
		}

		api_key = os.getenv('FINNHUB_API_KEY')
		if api_key:
			client = finnhub.Client(api_key=api_key)
			try:
				recs = client.recommendation_trends(symbol)
			except Exception:
				recs = []
			if recs:
				def _rec_key(item):
					period = item.get('period')
					try:
						return datetime.strptime(period, '%Y-%m-%d') if period else datetime.min
					except Exception:
						return datetime.min
				latest = max(recs, key=_rec_key)
				payload['recommendation'] = {
					'period': latest.get('period'),
					'strong_buy': latest.get('strongBuy'),
					'buy': latest.get('buy'),
					'hold': latest.get('hold'),
					'sell': latest.get('sell'),
					'strong_sell': latest.get('strongSell'),
				}
				payload['source'].append('finnhub')

			try:
				pt = client.price_target(symbol)
			except Exception:
				pt = None
			if pt:
				payload['price_target'] = {
					'high': pt.get('targetHigh'),
					'low': pt.get('targetLow'),
					'mean': pt.get('targetMean'),
					'median': pt.get('targetMedian'),
					'last_updated': pt.get('lastUpdated'),
				}
				if 'finnhub' not in payload['source']:
					payload['source'].append('finnhub')

		if payload['recommendation'] is None or payload['price_target'] is None:
			try:
				info = yf.Ticker(symbol).info or {}
			except Exception:
				info = {}

			if payload['recommendation'] is None and info:
				payload['recommendation'] = {
					'period': None,
					'strong_buy': None,
					'buy': None,
					'hold': None,
					'sell': None,
					'strong_sell': None,
					'mean': info.get('recommendationMean'),
					'analyst_count': info.get('numberOfAnalystOpinions'),
				}
				payload['source'].append('yahoo')

			if payload['price_target'] is None and info:
				payload['price_target'] = {
					'high': info.get('targetHighPrice'),
					'low': info.get('targetLowPrice'),
					'mean': info.get('targetMeanPrice'),
					'median': info.get('targetMedianPrice'),
					'last_updated': None,
				}
				payload['source'].append('yahoo')

		return Response(payload)


class PortfolioCsvImportView(APIView):
	parser_classes = (MultiPartParser, FormParser)

	def post(self, request):
		upload = request.data.get('file')
		if not upload:
			return Response({'error': 'CSV file is required.'}, status=400)

		portfolio_name = request.data.get('portfolio_name')
		initial_capital = request.data.get('initial_capital')
		initial_capital = float(initial_capital) if initial_capital else None

		raw = upload.read()
		try:
			decoded = raw.decode('utf-8-sig')
		except UnicodeDecodeError:
			decoded = raw.decode('latin-1')
		sample = decoded[:4096]
		if '\t' in sample:
			reader = csv.DictReader(io.StringIO(decoded), delimiter='\t')
		elif ';' in sample and sample.count(';') >= sample.count(','):
			reader = csv.DictReader(io.StringIO(decoded), delimiter=';')
		else:
			try:
				dialect = csv.Sniffer().sniff(sample)
			except csv.Error:
				dialect = csv.excel
			reader = csv.DictReader(io.StringIO(decoded), dialect=dialect)

		# Normalize headers
		if reader.fieldnames:
			normalized = []
			for f in reader.fieldnames:
				key = f.strip().lower()
				key = unicodedata.normalize('NFKD', key).encode('ascii', 'ignore').decode('ascii')
				key = ''.join(ch if ch.isalnum() else '_' for ch in key)
				while '__' in key:
					key = key.replace('__', '_')
				normalized.append(key.strip('_'))
			reader.fieldnames = normalized
		else:
			return Response({'error': 'CSV headers are missing.'}, status=400)

		rows = [
			row for row in reader
			if any(v is not None and str(v).strip() for v in (row or {}).values())
		]
		if not rows:
			return Response({'error': 'CSV has no data rows.'}, status=400)

		name_from_csv = (
			rows[0].get('portfolio_name')
			or rows[0].get('portfolio')
			or rows[0].get('account')
			or rows[0].get('account_name')
			or rows[0].get('numero_de_compte')
			or rows[0].get('numero_compte')
		)
		portfolio_name = portfolio_name or name_from_csv or 'Imported Portfolio'

		portfolio, _ = Portfolio.objects.get_or_create(
			name=portfolio_name,
			defaults={'capital': initial_capital or 0},
		)
		if initial_capital is not None:
			portfolio.capital = initial_capital
			portfolio.save(update_fields=['capital'])

		created_stocks = 0
		created_transactions = 0
		created_holdings = 0
		skipped_rows = 0
		errors = []
		total_value = 0.0

		def get_value(row, keys):
			for k in keys:
				if k in row and row[k] not in (None, ''):
					return row[k]
			return ''

		def parse_float(value):
			if value in (None, ''):
				return 0.0
			cleaned = str(value).replace(',', '').replace('%', '').strip()
			return float(cleaned)

		def parse_date(value):
			if not value:
				return ''
			value = str(value).strip()
			if '/' in value:
				try:
					return datetime.strptime(value, '%m/%d/%Y').date().isoformat()
				except ValueError:
					return value
			return value

		# Sort rows by date if present
		def date_key(r):
			return r.get('date') or ''

		for row in sorted(rows, key=date_key):
			symbol = (get_value(row, ['symbol', 'ticker', 'asset', 'security', 'symbole']) or '').upper().strip()
			if not symbol:
				continue

			stock_defaults = {
				'name': get_value(row, ['name', 'security_name', 'description', 'description_titre']) or symbol,
				'sector': get_value(row, ['sector', 'industry', 'secteur']) or 'Unknown',
				'target_weight': parse_float(get_value(row, ['target_weight', 'weight', 'allocation'])),
				'dividend_yield': parse_float(get_value(row, ['dividend_yield', 'div_yield', 'yield', 'rendement'])),
			}
			stock, created = Stock.objects.get_or_create(symbol=symbol, defaults=stock_defaults)
			if not created:
				for key, value in stock_defaults.items():
					setattr(stock, key, value)
				stock.save()
			else:
				created_stocks += 1

			market_price = parse_float(
				get_value(row, ['prix_du_marche', 'market_price', 'last_price', 'prix'])
			)
			market_value = parse_float(
				get_value(row, ['valeur_marchande', 'market_value', 'valeur_du_marche'])
			)
			if market_price == 0 and market_value > 0:
				quantity_hint = parse_float(get_value(row, ['shares', 'quantity', 'qty', 'units', 'quantite']))
				if quantity_hint > 0:
					market_price = market_value / quantity_hint

			if market_price > 0:
				stock.latest_price = market_price
				stock.latest_price_updated_at = datetime.now()
				stock.save(update_fields=['latest_price', 'latest_price_updated_at'])
				PriceHistory.objects.update_or_create(
					stock=stock,
					date=datetime.now().date(),
					defaults={'close_price': market_price},
				)

			tx_type = (get_value(row, ['transaction_type', 'type', 'action', 'side']) or '').upper()
			shares = get_value(row, ['shares', 'quantity', 'qty', 'units', 'quantite'])
			price = get_value(row, ['price_per_share', 'price', 'avg_price', 'fill_price', 'cout_unitaire_moyen', 'cout_unitaire'])
			date = parse_date(get_value(row, ['date', 'trade_date', 'transaction_date', 'settle_date']))

			is_holdings_row = not tx_type

			if is_holdings_row and shares:
				# Holdings-style CSV row: create/update holding without cash impact
				try:
					holding, _ = PortfolioHolding.objects.get_or_create(
						portfolio=portfolio,
						stock=stock,
						defaults={'shares': 0},
					)
					share_count = parse_float(shares)
					holding.shares = float(holding.shares or 0) + share_count
					holding.save(update_fields=['shares'])
					created_holdings += 1
					if market_value > 0:
						total_value += market_value
					elif market_price > 0:
						total_value += share_count * market_price
				except Exception as exc:
					skipped_rows += 1
					if len(errors) < 20:
						errors.append({'symbol': symbol, 'error': str(exc)})

			if tx_type and shares and price and date:
				try:
					serializer = TransactionSerializer(data={
						'portfolio': portfolio.id,
						'stock': stock.id,
						'shares': parse_float(shares),
						'price_per_share': parse_float(price),
						'date': date,
						'transaction_type': tx_type,
					})
					serializer.is_valid(raise_exception=True)
					serializer.save()
					created_transactions += 1
				except Exception as exc:
					skipped_rows += 1
					if len(errors) < 20:
						errors.append({'symbol': symbol, 'error': str(exc)})

		if total_value > 0:
			portfolio.capital = total_value
			portfolio.save(update_fields=['capital'])

		return Response({
			'portfolio_id': portfolio.id,
			'portfolio_name': portfolio.name,
			'created_stocks': created_stocks,
			'created_transactions': created_transactions,
			'created_holdings': created_holdings,
			'portfolio_capital': float(portfolio.capital or 0),
			'skipped_rows': skipped_rows,
			'errors': errors,
		})
