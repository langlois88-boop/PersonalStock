"""Regression test (2026-08-20) : CACHES n'était pas défini du tout avant
ce fix -- Django retombait sur LocMemCache (mémoire du process, jamais
partagé entre workers, jamais persisté -- perdu à chaque redémarrage/
déploiement de conteneur, très fréquent sur ce projet). Confirmé en
direct en testant portfolio/services/penny_risk_screen.py : un cache.set()
dans un process invisible dans un second process séparé.

Ce test verrouille juste le BACKEND configuré -- pas un test d'intégration
Redis complet (les autres suites qui utilisent cache.set/get partout
dans ce projet couvrent déjà le comportement réel)."""

from __future__ import annotations

from django.conf import settings
from django.test import TestCase


class CacheBackendIsRedisNotLocMemTests(TestCase):
    def test_default_cache_backend_is_redis(self):
        backend = settings.CACHES['default']['BACKEND']
        self.assertIn(
            'redis', backend.lower(),
            f"CACHES['default']['BACKEND']={backend} -- si ce n'est plus Redis, "
            "le cache redevient invisible entre process/workers et est perdu à "
            "chaque redémarrage de conteneur (voir le commentaire dans settings.py).",
        )

    def test_cache_uses_a_separate_redis_db_from_celery_broker(self):
        location = settings.CACHES['default']['LOCATION']
        self.assertFalse(
            location.rstrip('/').endswith('/0'),
            "Le cache Django ne devrait pas partager la DB Redis 0 avec le "
            "broker Celery et la couche Channels.",
        )
