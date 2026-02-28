import yfinance as yf
import pandas as pd
from django.utils import timezone
from portfolio.models import Stock, PriceHistory


def main() -> None:
    symbols = ["ATD.TO", "TEC.TO", "RY", "RY.TO"]

    for symbol in symbols:
        stock = Stock.objects.filter(symbol__iexact=symbol).first()
        if not stock:
            print(f"{symbol}: stock not found, skipping")
            continue
        try:
            hist = yf.Ticker(stock.symbol).history(period="5y", interval="1d", auto_adjust=False)
        except Exception as exc:
            print(f"{stock.symbol}: error {exc}")
            continue
        if hist is None or hist.empty:
            print(f"{stock.symbol}: no data")
            continue
        if isinstance(hist.columns, pd.MultiIndex):
            level0 = hist.columns.get_level_values(0)
            level1 = hist.columns.get_level_values(1)
            if "Close" in level0 or "Adj Close" in level0:
                hist.columns = level0
            elif "Close" in level1 or "Adj Close" in level1:
                hist.columns = level1
            else:
                hist.columns = [col[0] for col in hist.columns]
        if "Close" not in hist.columns and "Adj Close" in hist.columns:
            hist = hist.rename(columns={"Adj Close": "Close"})
        if "Close" not in hist.columns:
            print(f"{stock.symbol}: missing Close")
            continue

        last_row = hist.iloc[-1]
        try:
            stock.latest_price = float(last_row["Close"])
        except Exception:
            pass
        try:
            stock.day_low = float(last_row["Low"]) if "Low" in hist.columns else None
            stock.day_high = float(last_row["High"]) if "High" in hist.columns else None
        except Exception:
            pass
        stock.latest_price_updated_at = timezone.now()
        stock.save(update_fields=["latest_price", "day_low", "day_high", "latest_price_updated_at"])

        created = 0
        updated = 0
        for dt, row in hist.iterrows():
            close_price = float(row["Close"]) if pd.notna(row["Close"]) else None
            if close_price is None:
                continue
            _, was_created = PriceHistory.objects.update_or_create(
                stock=stock,
                date=dt.date(),
                defaults={"close_price": close_price},
            )
            if was_created:
                created += 1
            else:
                updated += 1

        print(f"{stock.symbol}: rows={len(hist)} created={created} updated={updated}")


if __name__ == "__main__":
    main()
