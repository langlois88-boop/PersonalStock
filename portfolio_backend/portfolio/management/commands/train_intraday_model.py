from __future__ import annotations

from django.core.management.base import BaseCommand

from ...ml_engine.intraday_training import (
    build_dataset,
    save_intraday_model,
    train_voting_ensemble,
    train_xgboost_model,
)


class Command(BaseCommand):
    help = "Train intraday 1m model(s) using Alpaca data."

    def add_arguments(self, parser):
        parser.add_argument('--symbols', type=str, default='ONCY', help='Comma-separated symbols')
        parser.add_argument('--bluechips', type=str, default='AAPL,NVDA,QQQ', help='Comma-separated bluechip symbols')
        parser.add_argument('--pennies', type=str, default='ONCY', help='Comma-separated penny symbols')
        parser.add_argument('--days', type=int, default=365)
        parser.add_argument('--mode', type=str, default='xgb', choices=['xgb', 'ensemble'])

    def handle(self, *args, **options):
        days = int(options['days'])
        if options['mode'] == 'ensemble':
            bluechips = [s.strip().upper() for s in options['bluechips'].split(',') if s.strip()]
            pennies = [s.strip().upper() for s in options['pennies'].split(',') if s.strip()]
            result = train_voting_ensemble(bluechips, pennies, days=days)
            path = save_intraday_model(result, 'ensemble')
            self.stdout.write(self.style.SUCCESS(f'Ensemble model saved to {path}'))
            return

        symbols = [s.strip().upper() for s in options['symbols'].split(',') if s.strip()]
        dataset, labels, feature_cols = build_dataset(symbols, days=days)
        result = train_xgboost_model(dataset, labels, feature_cols)
        path = save_intraday_model(result, 'xgb')
        self.stdout.write(self.style.SUCCESS(f'Model saved to {path}'))
        if result.importances:
            self.stdout.write('Top features:')
            for name, score in result.importances[:10]:
                self.stdout.write(f'- {name}: {score:.4f}')
