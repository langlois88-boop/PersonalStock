from __future__ import annotations

import pandas as pd
from . import market_data as yf
from prophet import Prophet


def run_predictions(symbol: str) -> tuple[float, str]:
    data = yf.download(symbol, period="1y", interval="1d", progress=False)
    if data.empty:
        return 0.0, "HOLD"

    df = data.reset_index()[["Date", "Close"]]
    df.columns = ["ds", "y"]

    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    predicted_price = float(forecast["yhat"].iloc[-1])

    last_close = float(df["y"].iloc[-1])
    if predicted_price > last_close * 1.02:
        recommendation = "BUY"
    elif predicted_price < last_close * 0.98:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    return predicted_price, recommendation
