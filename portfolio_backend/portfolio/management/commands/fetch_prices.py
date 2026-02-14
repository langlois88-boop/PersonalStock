import yfinance as yf
from django.core.management.base import BaseCommand
from django.utils import timezone

from portfolio.models import PriceHistory, Stock


class Command(BaseCommand):
    help = 'Fetch latest close price for each Stock and print it'

    def handle(self, *args, **options):
        for stock in Stock.objects.all():
            ticker = yf.Ticker(stock.symbol)
            try:
                data = ticker.history(period='1mo', interval='1d', timeout=10)
            except Exception as exc:
                self.stdout.write(f"{stock.symbol}: error {exc}")
                continue
            if data is None or data.empty or 'Close' not in data:
                self.stdout.write(f"{stock.symbol}: no data")
                continue
            last_row = data.iloc[-1]
            price = float(last_row['Close'])
            day_low = float(last_row['Low']) if 'Low' in data else None
            day_high = float(last_row['High']) if 'High' in data else None

            info = {}
            try:
                info = ticker.info or {}
            except Exception:
                info = {}

            sector = (info.get('sector') or '').strip()
            div_yield = info.get('dividendYield')
            if div_yield is None:
                div_yield = info.get('trailingAnnualDividendYield')
            div_yield = float(div_yield) if div_yield is not None else None

            stock.latest_price = price
            stock.day_low = day_low
            stock.day_high = day_high
            if (not stock.sector) or stock.sector.lower() == 'unknown':
                if sector:
                    stock.sector = sector
            if not stock.dividend_yield or float(stock.dividend_yield or 0) == 0:
                if div_yield is not None:
                    stock.dividend_yield = div_yield
            stock.latest_price_updated_at = timezone.now()
            stock.save(update_fields=['latest_price', 'day_low', 'day_high', 'sector', 'dividend_yield', 'latest_price_updated_at'])

            for dt, row in data.iterrows():
                PriceHistory.objects.update_or_create(
                    stock=stock,
                    date=dt.date(),
                    defaults={'close_price': float(row['Close'])},
                )

            self.stdout.write(f"{stock.symbol}: {price}")
