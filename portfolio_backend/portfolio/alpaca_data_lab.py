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

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except Exception:  # pragma: no cover
    TradingClient = None
    MarketOrderRequest = None
    StopOrderRequest = None
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
    except Exception as exc:
        logger.warning("submit_lab_market_order: échec pour %s qty=%s side=%s -- %r", symbol, qty, side, exc)
        return None


def submit_lab_stop_order(symbol: str, qty: int, stop_price: float) -> Any | None:
    """
    Place un vrai ordre stop-loss (SELL, GTC) chez Alpaca sur le compte
    paper dédié FUNDAMENTAL_LAB (2026-08-25).

    Pourquoi GTC (Good-Til-Canceled) et pas DAY comme submit_lab_market_order :
    un ordre DAY expirerait chaque soir sans jamais avoir protégé la position
    le lendemain -- il faut qu'il reste actif en continu tant qu'on ne l'a
    pas annulé nous-même (cancel_lab_order), typiquement à la fermeture réelle
    de la position (take_profit/max_hold_time/stop_loss rempli).

    Root cause du besoin (voir analysis/tasks.py::monitor_lab_positions) :
    un stop-loss vérifié seulement toutes les 10 min pendant les heures de
    marché peut laisser une perte dépasser largement le seuil visé avant
    réaction (SLQT -37,57% le 2026-08-25 contre -5% visé). Un ordre stop
    réel est surveillé en continu par Alpaca -- ça ne règle pas le risque de
    gap hors heures de marché (aucun ordre ne s'exécute marché fermé, même
    chez un vrai courtier), mais élimine le lag de polling intraday.
    """
    client = _alpaca_lab_trading_client()
    if client is None or StopOrderRequest is None or OrderSide is None or TimeInForce is None:
        return None
    symbol = (symbol or '').strip().upper()
    if not symbol or qty <= 0 or stop_price is None or stop_price <= 0:
        return None
    try:
        request = StopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            stop_price=round(float(stop_price), 2),
        )
        return client.submit_order(request)
    except Exception as exc:
        # 2026-09-01 : root cause encore inconnue d'un échec récurrent de
        # placement pour des penny stocks sous 1$ (SOAR/CLGN/CHOW) -- avant
        # ce log, `except Exception: return None` avalait le message réel
        # sans aucune trace, y compris dans les logs du conteneur. L'appelant
        # (monitor_lab_positions) journalise déjà l'échec dans SystemLog (visible
        # au dashboard) ; ce logger.warning donne le détail technique associé
        # (raison exacte du rejet Alpaca) dans les logs du conteneur pour
        # diagnostiquer la PROCHAINE occurrence au lieu de deviner.
        logger.warning(
            "submit_lab_stop_order: échec pour %s qty=%s stop_price=%s -- %r",
            symbol, qty, stop_price, exc,
        )
        return None


def get_lab_order_by_id(order_id: str) -> Any | None:
    client = _alpaca_lab_trading_client()
    if client is None:
        return None
    try:
        return client.get_order_by_id(order_id)
    except Exception as exc:
        logger.warning("get_lab_order_by_id: échec pour order_id=%s -- %r", order_id, exc)
        return None


def cancel_lab_order(order_id: str) -> bool:
    """
    Annule un ordre en attente (typiquement le stop order protecteur placé
    par submit_lab_stop_order) sur le compte paper dédié FUNDAMENTAL_LAB.

    Appelé juste avant de soumettre un ordre de vente manuel (take_profit /
    max_hold_time / stop_loss avec ratio de split ajusté) pour ne jamais
    laisser deux ordres de vente actifs simultanément sur la même position
    -- ça causerait soit un rejet Alpaca (qty insuffisante après la première
    vente), soit une double-vente si les deux venaient à se remplir.

    Retourne True si l'annulation a réussi, False sinon (échec réseau/API,
    OU ordre déjà dans un état terminal -- rempli/annulé/expiré -- que le
    SDK signale par une exception ; les deux cas sont indistingables ici
    sans inspecter le message d'erreur). L'appelant (monitor_lab_positions)
    vérifie déjà le statut réel de l'ordre stop AVANT d'appeler ceci, donc
    un False signifie concrètement "vérifier manuellement côté dashboard
    Alpaca avant de soumettre l'ordre de vente suivant", jamais un succès
    silencieux supposé.
    """
    if not order_id:
        return True
    client = _alpaca_lab_trading_client()
    if client is None:
        return False
    try:
        client.cancel_order_by_id(order_id)
        return True
    except Exception as exc:
        logger.warning("cancel_lab_order: échec pour order_id=%s -- %r", order_id, exc)
        return False


def close_lab_position(symbol: str) -> Any | None:
    client = _alpaca_lab_trading_client()
    if client is None:
        return None
    try:
        return client.close_position(symbol)
    except Exception:
        return None
