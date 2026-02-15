#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[rebuild] Pull latest code"
git pull

echo "[rebuild] Build and restart containers"
docker compose build --no-cache

docker compose up -d

echo "[rebuild] Update prices"
docker compose exec -T backend python manage.py fetch_prices

echo "[rebuild] Done"
