from __future__ import annotations

from django.core.management.base import BaseCommand

from ...ml_engine.recommender_training import save_recommender_model, train_recommender_model


class Command(BaseCommand):
    help = "Train and calibrate the recommender model."

    def handle(self, *args, **options):
        payload = train_recommender_model()
        path = save_recommender_model(payload)
        self.stdout.write(self.style.SUCCESS(f'Recommender model saved to {path}'))
