import asyncio
from datetime import datetime

from channels.generic.websocket import AsyncJsonWebsocketConsumer
from asgiref.sync import sync_to_async

from .models import PaperTrade, Stock


@sync_to_async
def _build_payload() -> dict:
	open_trades = list(PaperTrade.objects.filter(status='OPEN').order_by('-entry_date')[:50])
	tickers = list({t.ticker for t in open_trades if t.ticker})
	stocks = Stock.objects.filter(symbol__in=tickers)
	price_map = {s.symbol: float(s.latest_price or 0) for s in stocks}

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
			await asyncio.sleep(5)
