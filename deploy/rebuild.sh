#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "${SKIP_PULL:-0}" != "1" ]]; then
	echo "[rebuild] Pull latest code"
	git pull
else
	echo "[rebuild] Skipping git pull"
fi

echo "[rebuild] Build and restart containers"
docker compose build --no-cache

docker compose up -d

echo "[rebuild] Update prices"
echo "[rebuild] Run migrations"
docker compose exec -T backend python manage.py migrate
docker compose exec -T backend python manage.py fetch_prices

echo "[rebuild] Done"
