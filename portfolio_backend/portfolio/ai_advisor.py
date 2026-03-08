import json
import os
from django.utils import timezone
from dataclasses import dataclass
from typing import Any, Iterable

import requests
import pandas as pd

from .alpaca_data import get_intraday_context
from .crypto_processor import _btc_panic
from .models import Portfolio, PortfolioHolding, Stock, StockNews, AccountTransaction
from .ml_engine.engine.data_fusion import DataFusionEngine
from .ml_engine.backtester import (
	apply_feature_weighting_to_signal,
	get_model_path,
	load_or_train_model,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class AdvisorContext:
	symbol: str | None
	price: float | None
	rsi14: float | None
	patterns: list[str]
	rvol: float | None
	ml_score: float | None
	btc_panic: float | None
	news_sentiment: float | None
	portfolio_snapshot: dict[str, Any]


class DeepSeekAdvisor:
	def __init__(self) -> None:
		self.base_url = (os.getenv('OLLAMA_BASE_URL') or '').strip().rstrip('/')
		self.chat_base_url = (os.getenv('OLLAMA_CHAT_BASE_URL') or self.base_url).strip().rstrip('/')
		self.chat_mode = str(os.getenv('OLLAMA_CHAT_MODE', '')).strip().lower() in {'1', 'true', 'yes', 'y'}
		self.model = (os.getenv('OLLAMA_MODEL') or 'deepseek-r1:8b').strip()
		self.timeout = int(os.getenv('OLLAMA_TIMEOUT', '90'))
		self.analyzer = SentimentIntensityAnalyzer()

	def _portfolio_snapshot(self) -> dict[str, Any]:
		portfolio = Portfolio.objects.first()
		if not portfolio:
			portfolio_info = None
		else:
			portfolio_info = {'id': portfolio.id, 'name': portfolio.name}
		holdings = []
		if portfolio:
			holdings = (
				PortfolioHolding.objects.select_related('stock')
				.filter(portfolio=portfolio)
				[:20]
			)
		items = []
		if holdings:
			for holding in holdings:
				stock = holding.stock
				items.append({
					'symbol': (stock.symbol or '').strip().upper(),
					'name': stock.name or (stock.symbol or ''),
					'shares': float(holding.shares or 0),
					'latest_price': float(stock.latest_price or 0),
				})
		else:
			positions: dict[int, dict[str, Any]] = {}
			account_qs = AccountTransaction.objects.select_related('stock', 'account').filter(
				account__account_type__in=['TFSA', 'CRI', 'CASH']
			)
			for tx in account_qs:
				if not tx.stock or not tx.stock.symbol:
					continue
				if tx.type == 'DIVIDEND':
					continue
				sign = 1 if tx.type == 'BUY' else -1
				entry = positions.setdefault(
					tx.stock_id,
					{'stock': tx.stock, 'shares': 0.0},
				)
				entry['shares'] += float(tx.quantity or 0) * sign
			for entry in positions.values():
				stock = entry.get('stock')
				shares = float(entry.get('shares') or 0)
				if not stock or shares <= 0:
					continue
				items.append({
					'symbol': (stock.symbol or '').strip().upper(),
					'name': stock.name or (stock.symbol or ''),
					'shares': shares,
					'latest_price': float(stock.latest_price or 0),
				})
		return {
			'portfolio': portfolio_info,
			'holdings': items,
		}

	def _news_sentiment(self, symbol: str) -> float | None:
		if not symbol:
			return None
		stock = Stock.objects.filter(symbol__iexact=symbol).first()
		qs = StockNews.objects.filter(stock=stock) if stock else StockNews.objects.filter(stock__symbol__iexact=symbol)
		latest = list(qs.order_by('-published_at')[:6])
		if not latest:
			return None
		scores = []
		for item in latest:
			headline = (item.headline or '').strip()
			if not headline:
				continue
			scores.append(self.analyzer.polarity_scores(headline)['compound'])
		if not scores:
			return None
		return float(sum(scores) / len(scores))

	def _ml_score(self, symbol: str, universe: str) -> float | None:
		try:
			engine = DataFusionEngine(symbol)
			frame = engine.fuse_all()
			if frame is None or frame.empty:
				return None
			payload = load_or_train_model(frame, model_path=get_model_path(universe))
			if not payload or not payload.get('model'):
				return None
			last_row = frame.tail(1).copy()
			feature_list = payload.get('features') or []
			for col in feature_list:
				if col not in last_row.columns:
					last_row[col] = 0.0
			features = last_row[feature_list].fillna(0).values
			try:
				signal = float(payload['model'].predict_proba(features)[0][1])
			except Exception:
				signal = float(payload['model'].predict(features)[0]) if hasattr(payload['model'], 'predict') else 0.0
			signal = apply_feature_weighting_to_signal(signal, last_row.iloc[0], symbol)
			return round(signal * 100, 2)
		except Exception:
			return None

	def _build_context(self, symbol: str | None, user_query: str) -> AdvisorContext:
		context = get_intraday_context(symbol) if symbol else None
		price = None
		rsi14 = None
		patterns = []
		rvol = None
		if context:
			bars = context.get('bars')
			if isinstance(bars, pd.DataFrame) and not bars.empty:
				try:
					price = float(bars.iloc[-1]['close'])
				except Exception:
					price = None
			rsi14 = context.get('rsi14')
			try:
				rsi14 = float(rsi14) if rsi14 is not None else None
			except Exception:
				rsi14 = None
			patterns = list(context.get('patterns') or [])
			try:
				rvol = float(context.get('rvol')) if context.get('rvol') is not None else None
			except Exception:
				rvol = None

		universe = 'PENNY' if price is not None and price < 7 else 'BLUECHIP'
		ml_score = self._ml_score(symbol, universe) if symbol else None
		btc_panic = None
		try:
			btc_panic = float(_btc_panic()) if symbol and symbol.upper().startswith('HIVE') else None
		except Exception:
			btc_panic = None

		news_sentiment = self._news_sentiment(symbol) if symbol else None
		portfolio_snapshot = self._portfolio_snapshot() if 'TFSA' in user_query.upper() or 'PORTFOLIO' in user_query.upper() else {}

		return AdvisorContext(
			symbol=symbol,
			price=price,
			rsi14=rsi14,
			patterns=patterns,
			rvol=rvol,
			ml_score=ml_score,
			btc_panic=btc_panic,
			news_sentiment=news_sentiment,
			portfolio_snapshot=portfolio_snapshot,
		)

	def _build_prompt(self, ctx: AdvisorContext, user_query: str) -> str:
		symbol = ctx.symbol or 'PORTFOLIO'
		btc_note = ''
		if symbol.upper().startswith('HIVE'):
			btc_note = f"\nBTC Panic: {ctx.btc_panic if ctx.btc_panic is not None else 'N/A'}"
		portfolio_note = ''
		if ctx.portfolio_snapshot:
			portfolio_note = f"\nPortfolio: {json.dumps(ctx.portfolio_snapshot, ensure_ascii=False)}"
		return (
			"Tu es un analyste technique senior et quant. Réponds uniquement en français."
			" Réponds sans exposer ton raisonnement interne.\n"
			f"Données techniques : {ctx.patterns}, RSI: {ctx.rsi14}, Prix: {ctx.price}.\n"
			f"Score ML : {ctx.ml_score}.\n"
			f"Sentiment news: {ctx.news_sentiment}.\n"
			f"Ticker: {symbol}."
			f"{btc_note}"
			f"{portfolio_note}\n"
			"Consigne : Fournis un diagnostic puis un plan d'action avec Entrée, Stop-loss, TP1 (200$), TP2 (500$). "
			"Si tu vois un Bearish Engulfing sur NVDA avec un score ML faible, suggère une vente à {price} avec rachat à 175$. "
			"Si HIVE.TO est analysé, mentionne l'impact de la force du BTC.\n"
			f"Question utilisateur: {user_query}\n"
		)

	def _lite_summary(self, ctx: AdvisorContext, user_query: str) -> str:
		lines = ["Synthèse rapide (mode lite):"]
		if ctx.symbol:
			lines.append(f"Ticker: {ctx.symbol}")
		if ctx.price is not None:
			lines.append(f"Prix: {ctx.price}")
		if ctx.rsi14 is not None:
			lines.append(f"RSI14: {ctx.rsi14}")
		if ctx.patterns:
			lines.append(f"Patterns: {', '.join(ctx.patterns)}")
		if ctx.ml_score is not None:
			lines.append(f"Score ML: {ctx.ml_score}")
		lines.append("Plan d'action: WAIT si signal incertain, BUY si RSI < 30 et patterns haussiers, SELL si RSI > 70 et patterns baissiers.")
		return "\n".join(lines)

	def stream_answer(self, symbol: str | None, user_query: str) -> Iterable[str]:
		ctx = self._build_context(symbol, user_query)
		if not self.base_url and not self.chat_base_url:
			yield json.dumps({'text': self._lite_summary(ctx, user_query), 'done': True})
			return
		prompt = self._build_prompt(ctx, user_query)
		in_think = False
		try:
			if self.chat_mode or '/v1' in self.chat_base_url:
				now = timezone.now().strftime('%d/%m/%Y %H:%M')
				payload = {
					'model': self.model,
					'messages': [
						{
							'role': 'system',
							'content': (
								f"Tu es Danas. Nous sommes le {now}. "
								"Ignore ta date interne et utilise uniquement les données live fournies. "
								"Contexte Canada: TFSA = CELI (Compte d'Épargne Libre d'Impôt). "
								"Réponds uniquement en français. Ne révèle pas ton raisonnement interne."
							),
						},
						{'role': 'user', 'content': prompt},
					],
					'stream': True,
				}
				response = requests.post(
					f"{self.chat_base_url}/chat/completions",
					json=payload,
					stream=True,
					timeout=self.timeout,
				)
				response.encoding = 'utf-8'
				response.raise_for_status()
				for line in response.iter_lines(decode_unicode=True):
					if not line:
						continue
					if line.startswith('data:'):
						data = line.replace('data:', '').strip()
						if data == '[DONE]':
							yield json.dumps({'text': '', 'done': True})
							return
						try:
							chunk = json.loads(data)
						except Exception:
							chunk = {}
						delta = ((chunk.get('choices') or [{}])[0].get('delta') or {})
						text = delta.get('content') or ''
						if not text:
							continue
						if '<think>' in text:
							in_think = True
							text = text.split('<think>', 1)[-1]
						if in_think and '</think>' in text:
							text = text.split('</think>', 1)[-1]
							in_think = False
						if in_think:
							continue
						if text:
							yield json.dumps({'text': text, 'done': False})
				yield json.dumps({'text': '', 'done': True})
				return
			payload = {
				'model': self.model,
				'prompt': prompt,
				'stream': True,
			}
			response = requests.post(
				f"{self.base_url}/api/generate",
				json=payload,
				stream=True,
				timeout=self.timeout,
			)
			response.encoding = 'utf-8'
			response.raise_for_status()
			for line in response.iter_lines():
				if not line:
					continue
				chunk = json.loads(line)
				yield json.dumps({'text': chunk.get('response', ''), 'done': bool(chunk.get('done'))})
				if chunk.get('done'):
					return
		except Exception:
			yield json.dumps({'text': self._lite_summary(ctx, user_query), 'done': True})
