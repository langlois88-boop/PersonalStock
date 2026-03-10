#!/bin/bash
set -euo pipefail

cd /mnt/AppStorage/apps/PersonalStock

export PYTHONPATH=/app
export DJANGO_SETTINGS_MODULE=portfolio_backend.settings

log_dir="/mnt/AppStorage/apps/PersonalStock/logs"
mkdir -p "$log_dir"

{
  echo "[$(date)] Scanner start..."
  docker compose exec -T backend python manage.py ml_scanner

  container_id=$(docker compose ps -q backend)
  if [ -n "$container_id" ]; then
    docker cp "$container_id:/app/logs/market_scanner.log" "$log_dir/market_scanner.log" || true
  fi
  echo "[$(date)] Scanner done."
} >> "$log_dir/cron_scanner.log" 2>&1
