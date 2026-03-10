#!/bin/bash
set -euo pipefail

cd /mnt/AppStorage/apps/PersonalStock

export PYTHONPATH=/app
export DJANGO_SETTINGS_MODULE=portfolio_backend.settings

export MIN_CV_MEAN_STABLE=0.51
export MIN_CV_MEAN_PENNY=0.51
export MIN_WF_F1_PENNY=0.39
export STABLE_FEATURE_SET=core
export PENNY_FEATURE_SET=core
export FEATURE_IMPORTANCE_MIN_STABLE=0.04
export BACKTEST_LOG_PATH=/app/logs/backtest_history.csv

log_dir="/mnt/AppStorage/apps/PersonalStock/logs"
mkdir -p "$log_dir"

{
  echo "[$(date)] Training STABLE..."
  docker compose exec -T backend python -m portfolio.ml_engine.pipelines.stable_pipeline

  echo "[$(date)] Training PENNY..."
  docker compose exec -T backend python -m portfolio.ml_engine.pipelines.penny_pipeline

  echo "[$(date)] Backtest..."
  docker compose exec -T backend python -m portfolio.ml_engine.backtest_engine

  container_id=$(docker compose ps -q backend)
  if [ -n "$container_id" ] && [ -f "$log_dir/backtest_history.csv" ]; then
    rm -f "$log_dir/backtest_history.csv"
  fi
  if [ -n "$container_id" ]; then
    docker cp "$container_id:/app/logs/backtest_history.csv" "$log_dir/backtest_history.csv" || true
    docker cp "$container_id:/app/logs/market_scanner.log" "$log_dir/market_scanner.log" || true
  fi

  echo "[$(date)] Pipeline completed."
} >> "$log_dir/weekly_run.log" 2>&1
