from __future__ import annotations

from datetime import date
import os

import io
import pandas as pd
import numpy as np
import requests
from ... import market_data as yf
from django.core.management.base import BaseCommand

from portfolio.models import MacroIndicator


class Command(BaseCommand):
    help = "Fetch macro indicators (SP500, VIX, 10Y, CPI, Oil) and store them daily."

    def add_arguments(self, parser):
        parser.add_argument('--start', type=str, default='2025-01-01')

    def handle(self, *args, **options):
        start = options['start']

        def fetch_fred_series(series_id: str, column_name: str) -> pd.DataFrame:
            api_key = os.getenv('FRED_API_KEY')
            if api_key:
                url = (
                    'https://api.stlouisfed.org/fred/series/observations'
                    f'?series_id={series_id}&api_key={api_key}&file_type=json&observation_start={start}'
                )
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                payload = resp.json() or {}
                obs = payload.get('observations', [])
                df = pd.DataFrame(obs)
                if df.empty:
                    return pd.DataFrame(columns=[column_name])
                df['date'] = pd.to_datetime(df['date'])
                df[column_name] = pd.to_numeric(df['value'].replace('.', np.nan), errors='coerce')
                df = df.set_index('date')[[column_name]]
                return df

            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            date_col = 'DATE' if 'DATE' in df.columns else 'observation_date'
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            df = df.rename(columns={series_id: column_name})
            df[column_name] = pd.to_numeric(df[column_name].replace('.', np.nan), errors='coerce')
            return df

        fred_frames = [
            fetch_fred_series('GS10', 'interest_rate_10y'),
            fetch_fred_series('CPIAUCSL', 'inflation_rate'),
            fetch_fred_series('VIXCLS', 'vix_index'),
            fetch_fred_series('DCOILWTICO', 'oil_price'),
        ]
        fred = pd.concat(fred_frames, axis=1)

        spy = yf.download('SPY', start=start, interval='1d', progress=False)
        if spy is None or spy.empty or 'Close' not in spy:
            self.stdout.write(self.style.ERROR('SPY data unavailable.'))
            return
        close_col = spy['Close']
        if isinstance(close_col, pd.DataFrame):
            close_col = close_col.iloc[:, 0]
        spy_close = close_col.rename('sp500_close')

        df = pd.concat([spy_close, fred], axis=1).sort_index()
        df = df.ffill()

        created = 0
        updated = 0

        for idx, row in df.iterrows():
            if pd.isna(row.get('sp500_close')):
                continue
            day = idx.date() if hasattr(idx, 'date') else date.fromisoformat(str(idx))

            defaults = {
                'sp500_close': float(row.get('sp500_close')),
                'vix_index': float(row.get('vix_index')) if not pd.isna(row.get('vix_index')) else 0.0,
                'interest_rate_10y': float(row.get('interest_rate_10y')) if not pd.isna(row.get('interest_rate_10y')) else 0.0,
                'inflation_rate': float(row.get('inflation_rate')) if not pd.isna(row.get('inflation_rate')) else 0.0,
                'oil_price': float(row.get('oil_price')) if not pd.isna(row.get('oil_price')) else None,
            }

            obj, was_created = MacroIndicator.objects.update_or_create(
                date=day,
                defaults=defaults,
            )
            if was_created:
                created += 1
            else:
                updated += 1

        self.stdout.write(self.style.SUCCESS(f"Macro indicators synced. created={created} updated={updated}"))
