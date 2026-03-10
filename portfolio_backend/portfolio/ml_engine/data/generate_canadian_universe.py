from __future__ import annotations

"""Generate canadian_universe.csv from database symbols."""

import os
from pathlib import Path

import pandas as pd
import requests


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "portfolio_backend.settings")
    import django

    django.setup()

    from portfolio.models import Stock, PennyStockUniverse

    symbols: list[tuple[str, float]] = []
    fallback: list[str] = []

    for stock in Stock.objects.all().only("symbol"):
        symbol = (stock.symbol or "").strip().upper()
        if not symbol:
            continue
        if symbol.endswith(".TO") or symbol.endswith(".V"):
            symbols.append((symbol, 0.0))
        fallback.append(symbol)

    for penny in PennyStockUniverse.objects.all().only("symbol", "volume"):
        symbol = (penny.symbol or "").strip().upper()
        if not symbol:
            continue
        volume = float(penny.volume or 0.0)
        if symbol.endswith(".TO") or symbol.endswith(".V"):
            symbols.append((symbol, volume))
        fallback.append(symbol)

    unique: dict[str, float] = {}
    for symbol, volume in symbols:
        if not symbol:
            continue
        existing = unique.get(symbol)
        if existing is None or volume > existing:
            unique[symbol] = volume

    ranked = sorted(unique.items(), key=lambda item: item[1], reverse=True)
    picked = [sym for sym, _ in ranked]

    limit = int(os.getenv("CANADIAN_UNIVERSE_LIMIT", "300"))
    if len(picked) < limit:
        seen = set(picked)
        for symbol in fallback:
            if symbol and symbol not in seen:
                picked.append(symbol)
                seen.add(symbol)
            if len(picked) >= limit:
                break

    if len(picked) < limit and os.getenv("CANADIAN_UNIVERSE_FETCH_WEB", "0") == "1":
        extra = _fetch_from_wikipedia(limit - len(picked))
        for symbol in extra:
            if symbol not in picked:
                picked.append(symbol)
            if len(picked) >= limit:
                break

    picked = picked[:limit]

    output_path = Path(os.getenv("CANADIAN_UNIVERSE_PATH", "/app/portfolio/ml_engine/data/canadian_universe.csv"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"symbol": picked})
    df.to_csv(output_path, index=False)
    print(f"Saved {len(picked)} symbols to {output_path}")


def _fetch_from_wikipedia(max_items: int) -> list[str]:
    urls = [
        "https://en.wikipedia.org/wiki/S%26P/TSX_60",
        "https://en.wikipedia.org/wiki/S%26P/TSX_Venture_50",
        "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index",
    ]
    symbols: list[str] = []
    for url in urls:
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            tables = pd.read_html(response.text)
        except Exception:
            continue

        for table in tables:
            columns = [str(col).lower() for col in table.columns]
            if not any("symbol" in col or "ticker" in col for col in columns):
                continue
            symbol_col = None
            for col in table.columns:
                if "symbol" in str(col).lower() or "ticker" in str(col).lower():
                    symbol_col = col
                    break
            if symbol_col is None:
                continue
            raw = table[symbol_col].astype(str).tolist()
            for item in raw:
                sym = item.strip().upper().replace(".", "-")
                if not sym or sym == "NAN":
                    continue
                if "TSX VENTURE" in url or "VENTURE" in url:
                    if not sym.endswith(".V"):
                        sym = f"{sym}.V"
                else:
                    if not sym.endswith(".TO"):
                        sym = f"{sym}.TO"
                symbols.append(sym)

    unique: list[str] = []
    seen = set()
    for sym in symbols:
        if sym not in seen:
            unique.append(sym)
            seen.add(sym)
        if len(unique) >= max_items:
            break
    return unique


if __name__ == "__main__":
    main()
