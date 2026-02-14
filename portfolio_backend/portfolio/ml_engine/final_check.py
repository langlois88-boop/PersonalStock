from __future__ import annotations

import json
import requests

BASE_URL = "http://127.0.0.1:8000/api/backtester/?symbol=SPY&days=365"

OLD = {"Return": -20.78, "MaxDD": -22.27, "Sharpe": -2.44}


def main() -> None:
    resp = requests.get(BASE_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    new_metrics = {
        "Return": data.get("total_return_pct"),
        "MaxDD": (data.get("max_drawdown") or 0) * 100,
        "Sharpe": data.get("sharpe_ratio"),
    }

    results = {
        "Ancien Modèle": OLD,
        "Nouveau Modèle (Optimisé)": new_metrics,
    }

    print(json.dumps(results, indent=2))
    if new_metrics["Return"] is not None:
        improvement = new_metrics["Return"] - OLD["Return"]
        print(f"Amélioration du rendement : {improvement:.2f}%")


if __name__ == "__main__":
    main()
