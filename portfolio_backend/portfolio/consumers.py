import asyncio
from datetime import datetime

from channels.generic.websocket import AsyncJsonWebsocketConsumer
from asgiref.sync import sync_to_async

from django.core.cache import cache

from .models import PaperTrade, Stock
from .alpaca_data import get_latest_trade_price
from . import market_data as yf


@sync_to_async
def _build_payload() -> dict:
	open_trades = list(PaperTrade.objects.filter(status='OPEN').order_by('-entry_date')[:50])
	tickers = list({t.ticker for t in open_trades if t.ticker})
	stocks = Stock.objects.filter(symbol__in=tickers)
	price_map = {s.symbol: float(s.latest_price or 0) for s in stocks}

	def _live_price(symbol: str) -> float | None:
		symbol = (symbol or '').strip().upper()
		if not symbol:
			return None
		cache_key = f"live_price:{symbol}"
		cached = cache.get(cache_key)
		if cached is not None:
			return float(cached)
		price = None
		try:
			price = get_latest_trade_price(symbol)
		except Exception:
			price = None
		if price is None:
			try:
				hist = yf.Ticker(symbol).history(period='1d', interval='1m', timeout=10)
				close = hist['Close'].iloc[-1] if hist is not None and not hist.empty and 'Close' in hist else None
				price = float(close) if close is not None else None
			except Exception:
				price = None
		if price is not None:
			cache.set(cache_key, float(price), timeout=10)
		return float(price) if price is not None else None

	for symbol in tickers:
		live = _live_price(symbol)
		if live is not None:
			price_map[symbol] = float(live)

	positions = []
	for trade in open_trades:
		price = price_map.get(trade.ticker)
		entry_price = float(trade.entry_price)
		qty = float(trade.quantity)
		current_value = (price or entry_price) * qty
		entry_value = entry_price * qty
		unrealized = current_value - entry_value
		positions.append({
			'ticker': trade.ticker,
			'sandbox': trade.sandbox,
			'entry_price': entry_price,
			'quantity': trade.quantity,
			'current_price': price,
			'current_value': round(current_value, 2),
			'unrealized_pnl': round(unrealized, 2),
		})

	return {
		'timestamp': datetime.utcnow().isoformat() + 'Z',
		'prices': price_map,
		'positions': positions,
	}


class LiveUpdatesConsumer(AsyncJsonWebsocketConsumer):
	async def connect(self):
		await self.accept()
		self._task = asyncio.create_task(self._send_loop())

	async def disconnect(self, close_code):
		if getattr(self, '_task', None):
			self._task.cancel()

	async def _send_loop(self):
		while True:
			payload = await _build_payload()
			await self.send_json(payload)
			await asyncio.sleep(10)
