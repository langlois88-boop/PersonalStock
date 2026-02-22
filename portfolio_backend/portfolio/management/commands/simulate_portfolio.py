from __future__ import annotations

import pandas as pd
from ... import market_data as yf
from django.core.management.base import BaseCommand

from portfolio.simulator.rl_env import (
    simulate_drip,
    simulate_rebalance,
    simulate_strategy,
)


class Command(BaseCommand):
    help = "Run a simple portfolio simulation using historical prices"

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbols",
            type=str,
            required=True,
            help="Comma-separated symbols (e.g. AAPL,MSFT,TSLA)",
        )
        parser.add_argument(
            "--period",
            type=str,
            default="1y",
            help="Historical period to download (default: 1y)",
        )
        parser.add_argument(
            "--strategy",
            type=str,
            default="hold",
            choices=["hold", "drip", "rebalance"],
            help="Simulation strategy: hold, drip, rebalance",
        )
        parser.add_argument(
            "--annual-yield",
            type=float,
            default=0.03,
            help="Annual dividend yield for DRIP strategy (default: 0.03)",
        )
        parser.add_argument(
            "--rebalance-days",
            type=int,
            default=21,
            help="Rebalance frequency in trading days (default: 21)",
        )

    def handle(self, *args, **options):
        symbols = [s.strip() for s in options["symbols"].split(",") if s.strip()]
        period = options["period"]

        if not symbols:
            self.stderr.write("No symbols provided")
            return

        data = yf.download(symbols, period=period, interval="1d", progress=False)
        if data.empty:
            self.stderr.write("No data returned")
            return

        close = data["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame()

        strategy = options["strategy"]
        if strategy == "drip":
            result = simulate_drip(close, annual_yield=options["annual_yield"])
        elif strategy == "rebalance":
            result = simulate_rebalance(close, rebalance_days=options["rebalance_days"])
        else:
            result = simulate_strategy(close)
        values = result["values"]
        if values:
            self.stdout.write(
                f"Simulated {len(values)} days. Final value: ${values[-1]:.2f}"
            )
        else:
            self.stdout.write("No values produced")
