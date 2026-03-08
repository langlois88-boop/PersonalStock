from __future__ import annotations

import json
import os
import requests

BASE_URL = os.getenv("BACKTEST_BASE_URL", "http://127.0.0.1:8000/api/backtester/?symbol=SPY&days=365")
BASELINE_JSON = os.getenv("BACKTEST_BASELINE_JSON", "")

def _load_baseline() -> dict:
    if BASELINE_JSON:
        try:
            return json.loads(BASELINE_JSON)
        except Exception:
            return {}
    return {}


def main() -> None:
    resp = requests.get(BASE_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    new_metrics = {
        "Return": data.get("total_return_pct"),
        "MaxDD": (data.get("max_drawdown") or 0) * 100,
        "Sharpe": data.get("sharpe_ratio"),
    }

    baseline = _load_baseline()
    results = {
        "baseline": baseline,
        "candidate": new_metrics,
    }

    print(json.dumps(results, indent=2))
    if baseline.get("Return") is not None and new_metrics["Return"] is not None:
        improvement = new_metrics["Return"] - baseline["Return"]
        print(f"Return improvement: {improvement:.2f}%")


if __name__ == "__main__":
    main()
