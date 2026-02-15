import csv
import io
import json
import os
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Any
import unicodedata
from urllib.parse import quote
from urllib.request import Request, urlopen
from django.db import models
from django.db.utils import OperationalError
from django.db.models import Prefetch
from django.utils import timezone
import finnhub
import yfinance as yf
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

from .models import (
	Account,
	AccountTransaction,
	AlertEvent,
	Dividend,
	DripSnapshot,
	Portfolio,
	PortfolioDigest,
	PortfolioHolding,
	PennySignal,
	PennyStockSnapshot,
	PennyStockUniverse,
	PriceHistory,
	Prediction,
	PaperTrade,
	Stock,
	StockNews,
	MacroIndicator,
	DataQADaily,
	ModelCalibrationDaily,
	ModelDriftDaily,
	ModelEvaluationDaily,
	Transaction,
	UserPreference,
	TaskRunLog,
	ModelRegistry,
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
)
from .ai_module import run_predictions
from .tasks import _fetch_yahoo_screener, _fetch_yfinance_screeners
from .ai_scout import build_scout_summary
from .ml_engine.engine.data_fusion import DataFusionEngine
from .ml_engine.backtester import AIBacktester, load_or_train_model, FEATURE_COLUMNS, get_model_path


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
					import yfinance as yf

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
			open_trades = PaperTrade.objects.filter(status='OPEN')
			closed_trades = PaperTrade.objects.filter(status='CLOSED')
			if sandbox:
				open_trades = open_trades.filter(sandbox=sandbox)
				closed_trades = closed_trades.filter(sandbox=sandbox)

			closed_pnl = float(sum([float(t.pnl or 0) for t in closed_trades]))
			open_value = 0.0
			for t in open_trades:
				open_value += float(t.entry_price) * float(t.quantity)

			available = initial_capital + closed_pnl - open_value
			return Response({
				'sandbox': sandbox or 'ALL',
				'initial_capital': initial_capital,
				'available_capital': round(available, 2),
				'open_value': round(open_value, 2),
				'closed_pnl': round(closed_pnl, 2),
				'open_positions': PaperTradeSerializer(open_trades, many=True).data,
				'closed_positions': PaperTradeSerializer(closed_trades[:25], many=True).data,
			})
		except Exception:
			return Response({
				'sandbox': sandbox or 'ALL',
				'initial_capital': initial_capital,
				'available_capital': round(initial_capital, 2),
				'open_value': 0,
				'closed_pnl': 0,
				'open_positions': [],
				'closed_positions': [],
			})


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
			model = joblib.load(model_path)
		except Exception:
			return Response({'error': 'Failed to load model.'}, status=503)

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
		features = np.array([
			float(close.iloc[-1]),
			float(close.rolling(10).mean().iloc[-1]),
			float(close.rolling(20).mean().iloc[-1]),
			float(close.rolling(50).mean().iloc[-1]),
			float(ret.rolling(20).std().iloc[-1]),
			float(volume.pct_change().rolling(10).mean().iloc[-1]),
			float(rsi(close, 14)),
		])

		try:
			prob = float(model.predict_proba([features])[0][1])
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


class PortfolioDashboardView(APIView):
	def _build_confidence_meter(self) -> dict[str, Any] | None:
		symbol = (os.getenv('CONFIDENCE_SYMBOL') or os.getenv('PAPER_WATCHLIST', 'SPY').split(',')[0]).strip().upper()
		if not symbol:
			return None
		try:
			fusion = DataFusionEngine(symbol)
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
			volume_z = float(last_row.iloc[0].get('VolumeZ', 0.0) or 0.0)
			vol_regime = float(last_row.iloc[0].get('vol_regime', 0.0) or 0.0)
			ai_score = round(signal * 100, 2)
			min_score = float(os.getenv('CONFIDENCE_AI_SCORE_MIN', '80'))
			min_volume_z = float(os.getenv('CONFIDENCE_VOLUME_Z_MIN', '0.5'))
			max_vol_regime = float(os.getenv('CONFIDENCE_VOL_REGIME_MAX', '1.6'))
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
				'status': status,
				'label': label,
				'thresholds': {
					'ai_score_min': min_score,
					'volume_z_min': min_volume_z,
					'vol_regime_max': max_vol_regime,
				},
			}
		except Exception:
			return {'symbol': symbol, 'status': 'unavailable'}

	def get(self, request):
		portfolio_id = request.query_params.get('portfolio_id')
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
					'allocation': {
						'stable_pct': 0,
						'risky_pct': 0,
						'stable_value': 0,
						'risky_value': 0,
					},
					'holdings': [],
					'chart': [],
					'confidence_meter': self._build_confidence_meter(),
				}, status=200)

			position_map = {}
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
			total_value = 0.0
			stable_value = 0.0
			risky_value = 0.0
			change_1d = 0.0
			change_7d = 0.0

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
					'category': 'Stable' if is_stable else 'Risky',
				})

			allocation_pct = (stable_value / total_value * 100) if total_value else 0
			change_1d_pct = (change_1d / (total_value - change_1d) * 100) if total_value else 0
			change_7d_pct = (change_7d / (total_value - change_7d) * 100) if total_value else 0

			return Response({
				'portfolio': None,
				'total_balance': round(total_value, 2),
				'change_24h': round(change_1d, 2),
				'change_24h_pct': round(change_1d_pct, 2),
				'change_7d': round(change_7d, 2),
				'change_7d_pct': round(change_7d_pct, 2),
				'allocation': {
					'stable_pct': round(allocation_pct, 2),
					'risky_pct': round(100 - allocation_pct, 2),
					'stable_value': round(stable_value, 2),
					'risky_value': round(risky_value, 2),
				},
				'holdings': items,
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
			total_value += value
			cost_value = avg_cost * float(holding.shares or 0)
			unrealized = value - cost_value
			unrealized_pct = (unrealized / cost_value * 100) if cost_value else 0

			prev_1d = PriceHistory.objects.filter(stock=stock).order_by('-date')[1:2].first()
			if prev_1d:
				change_1d += (price - float(prev_1d.close_price)) * float(holding.shares or 0)

			prev_7d = PriceHistory.objects.filter(stock=stock).order_by('-date')[7:8].first()
			if prev_7d:
				change_7d += (price - float(prev_7d.close_price)) * float(holding.shares or 0)

			is_stable = effective_price >= 5 or float(stock.dividend_yield or 0) >= 0.02
			if is_stable:
				stable_value += value
			else:
				risky_value += value

			items.append({
				'ticker': stock.symbol,
				'name': stock.name,
				'sector': stock.sector,
				'dividend_yield': float(stock.dividend_yield or 0),
				'shares': float(holding.shares or 0),
				'price': effective_price,
				'value': value,
				'avg_cost': round(avg_cost, 4),
				'cost_value': round(cost_value, 2),
				'unrealized_pnl': round(unrealized, 2),
				'unrealized_pnl_pct': round(unrealized_pct, 2),
				'category': 'Stable' if is_stable else 'Risky',
			})

		allocation_pct = (stable_value / total_value * 100) if total_value else 0
		change_1d_pct = (change_1d / (total_value - change_1d) * 100) if total_value else 0
		change_7d_pct = (change_7d / (total_value - change_7d) * 100) if total_value else 0

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

		return Response({
			'portfolio': {'id': portfolio.id, 'name': portfolio.name},
			'total_balance': round(total_value, 2),
			'change_24h': round(change_1d, 2),
			'change_24h_pct': round(change_1d_pct, 2),
			'change_7d': round(change_7d, 2),
			'change_7d_pct': round(change_7d_pct, 2),
			'allocation': {
				'stable_pct': round(allocation_pct, 2),
				'risky_pct': round(100 - allocation_pct, 2),
				'stable_value': round(stable_value, 2),
				'risky_value': round(risky_value, 2),
			},
			'holdings': items,
			'chart': chart,
			'confidence_meter': self._build_confidence_meter(),
		})


class AccountDashboardView(APIView):
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

		for account in accounts:
			positions = []
			total_value = 0.0
			total_cost = 0.0
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
				unrealized_pct = ((current_value - cost_value) / cost_value * 100) if cost_value else None
				total_value += current_value
				total_cost += cost_value

				weekly_price = self._price_at_or_before(stock, weekly_date)
				monthly_price = self._price_at_or_before(stock, monthly_date)
				annual_price = self._price_at_or_before(stock, annual_date)

				weekly_return = ((current - weekly_price) / weekly_price * 100) if weekly_price else None
				monthly_return = ((current - monthly_price) / monthly_price * 100) if monthly_price else None
				annual_return = ((current - annual_price) / annual_price * 100) if annual_price else None

				latest_two = list(PriceHistory.objects.filter(stock=stock).order_by('-date')[:2])
				prev_close = float(latest_two[1].close_price) if len(latest_two) > 1 else None
				day_change_pct = ((current - prev_close) / prev_close * 100) if prev_close else None

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
					'unrealized_pnl_pct': round(unrealized_pct, 2) if unrealized_pct is not None else None,
					'weekly_return_pct': round(weekly_return, 2) if weekly_return is not None else None,
					'monthly_return_pct': round(monthly_return, 2) if monthly_return is not None else None,
					'annual_return_pct': round(annual_return, 2) if annual_return is not None else None,
					'day_change_pct': round(day_change_pct, 2) if day_change_pct is not None else None,
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
			data = yf.Ticker(symbol).history(period='1y', interval='1d', timeout=10)
			spy = yf.Ticker('SPY').history(period='1y', interval='1d', timeout=10)
		except Exception:
			return Response({'error': 'Failed to load price data.'}, status=503)

		if data is None or data.empty or 'Close' not in data or len(data) < 200:
			return Response({'error': 'Insufficient price history.'}, status=400)
		if spy is None or spy.empty or 'Close' not in spy or len(spy) < 200:
			return Response({'error': 'Insufficient SPY history.'}, status=400)

		close = data['Close']
		volume = data['Volume'] if 'Volume' in data else pd.Series([0] * len(close), index=close.index)
		ret = close.pct_change().dropna()

		spy_close = spy['Close']
		spy_ret = spy_close.pct_change().dropna()
		aligned = pd.concat([ret, spy_ret], axis=1, join='inner').dropna()
		aligned.columns = ['stock', 'spy']
		if len(aligned) < 60:
			return Response({'error': 'Insufficient aligned history.'}, status=400)

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
			return Response({'error': 'Prediction failed.'}, status=500)

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
			return Response({'error': 'No data available for backtest.'}, status=400)

		payload = load_or_train_model(data, model_path=get_model_path(universe))
		backtester = AIBacktester(data, payload, symbol=symbol)
		result = backtester.run_simulation(lookback_days=days)

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
			'model_version': model_version,
			'feature_importance': feature_importance,
			'logs': logs,
			'final_balance': result.final_balance,
			'total_return_pct': result.total_return_pct,
			'win_rate': result.win_rate,
			'sharpe_ratio': result.sharpe_ratio,
			'max_drawdown': result.max_drawdown,
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
