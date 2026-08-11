"""
Client Alpaca DÉDIÉ à FUNDAMENTAL_LAB (module `analysis`), séparé du client
partagé de `portfolio/alpaca_data.py` utilisé par WATCHLIST/AI_BLUECHIP/
AI_PENNY.

Pourquoi un fichier séparé plutôt que paramétrer les fonctions existantes
(voir docs/TECH_DEBT_NOTES.md, item 11 / diagnostic B1, 2026-08-11) :
- Les fonctions de `alpaca_data.py` sont utilisées par le pipeline de
  trading réel (3 sandboxes actifs) ; les modifier pour accepter un second
  jeu de clés introduit un risque de régression sur du code qui gère déjà
  du capital réel, pour un bénéfice de duplication de code mineur.
- `broker='ALPACA_LAB'` (nouvelle valeur sur PaperTrade.BROKER_CHOICES) +
  ce fichier séparé garantissent qu'aucune requête existante non scopée par
  sandbox (ex. sync_alpaca_paper_trades, _journal_paper_trades) ne peut
  jamais toucher au compte FUNDAMENTAL_LAB, même par accident futur.

Convention de nommage DÉLIBÉRÉE : chaque fonction ici porte un nom explicite
("_lab_"/"lab_") pour qu'un futur copier-coller ne réintroduise jamais
silencieusement une confusion entre les deux comptes -- ne jamais renommer
ces fonctions pour "matcher" celles de alpaca_data.py.

Env vars dédiées (jamais commitées, voir deploy/.env sur le NAS, confirmé
dans .gitignore) : ALPACA_API_KEY_FUNDAMENTAL_LAB /
ALPACA_SECRET_KEY_FUNDAMENTAL_LAB. Compte paper vérifié en direct le
2026-08-11 (account_number=PA34ZP3JNN2T, $50 000 de capital de départ),
totalement séparé du compte partagé ALPACA_API_KEY/ALPACA_SECRET_KEY.

Import du SDK dupliqué volontairement (pas réimporté depuis alpaca_data.py)
pour qu'un changement du bloc d'import protégé de ce dernier ne puisse
jamais casser ce module, et vice versa -- indépendance complète.
"""
from __future__ import annotations

import os
from typing import Any

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except Exception:  # pragma: no cover
    TradingClient = None
    MarketOrderRequest = None
    OrderSide = None
    TimeInForce = None


def _alpaca_lab_trading_client() -> "TradingClient | None":
    if TradingClient is None:
        return None
    api_key = os.getenv('ALPACA_API_KEY_FUNDAMENTAL_LAB')
    api_secret = os.getenv('ALPACA_SECRET_KEY_FUNDAMENTAL_LAB')
    if not api_key or not api_secret:
        return None
    # Toujours paper trading pour ce sandbox -- aucune variante live prévue.
    return TradingClient(api_key, api_secret, paper=True)


def get_lab_account() -> Any | None:
    client = _alpaca_lab_trading_client()
    if client is None:
        return None
    try:
        return client.get_account()
    except Exception:
        return None


def get_lab_open_positions() -> list[Any]:
    client = _alpaca_lab_trading_client()
    if client is None:
        return []
    try:
        return client.get_all_positions() or []
    except Exception:
        return []


def submit_lab_market_order(symbol: str, qty: int, side: str) -> Any | None:
    """
    Place un ordre marché réel sur le compte paper dédié FUNDAMENTAL_LAB.
    Retourne None si le client n'est pas configuré ou si l'ordre échoue --
    l'appelant (analysis/tasks.py) traite ça comme un échec explicite,
    jamais un fallback silencieux vers une position simulée.
    """
    client = _alpaca_lab_trading_client()
    if client is None or MarketOrderRequest is None or OrderSide is None or TimeInForce is None:
        return None
    symbol = (symbol or '').strip().upper()
    if not symbol or qty <= 0:
        return None
    try:
        side_enum = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side_enum,
            time_in_force=TimeInForce.DAY,
        )
        return client.submit_order(request)
    except Exception:
        return None


def get_lab_order_by_id(order_id: str) -> Any | None:
    client = _alpaca_lab_trading_client()
    if client is None:
        return None
    try:
        return client.get_order_by_id(order_id)
    except Exception:
        return None


def close_lab_position(symbol: str) -> Any | None:
    client = _alpaca_lab_trading_client()
    if client is None:
        return None
    try:
        return client.close_position(symbol)
    except Exception:
        return None
