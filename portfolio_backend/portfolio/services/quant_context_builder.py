from __future__ import annotations

"""
quant_context_builder.py  — FIX #11
=====================================
Enrichit AskTheQuantStreamView avec le contexte portfolio réel.

Problème actuel : le stream DeepSeek ne reçoit pas les positions actuelles,
P&L ni stats sandbox → réponses génériques sans rapport avec la réalité.

Ce module construit un contexte structuré que DeepSeekAdvisor reçoit
avant de répondre, rendant les réponses pertinentes et personnalisées.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import Any

logger = logging.getLogger(__name__)


def build_quant_context(user=None, question: str = "") -> dict[str, Any]:
    """
    Construit le contexte complet pour le quant stream.

    Args:
        user     : instance User Django (peut être None si pas d'auth)
        question : la question posée (pour adapter le contexte)

    Returns:
        dict avec tout le contexte pertinent

    Usage dans views.py → AskTheQuantStreamView.post() :

        from .quant_context_builder import build_quant_context, context_to_prompt

        context = build_quant_context(user=request.user, question=question)
        context_block = context_to_prompt(context, question)

        # Passer à DeepSeekAdvisor :
        advisor = DeepSeekAdvisor()
        for chunk in advisor.stream_answer(symbol, question, context=context_block):
            yield chunk
    """
    context: dict[str, Any] = {}

    try:
        context["portfolio"] = _portfolio_snapshot(user)
    except Exception as exc:
        logger.warning("portfolio context failed: %s", exc)
        context["portfolio"] = {}

    try:
        context["sandboxes"] = _sandbox_stats()
    except Exception as exc:
        logger.warning("sandbox context failed: %s", exc)
        context["sandboxes"] = {}

    try:
        context["active_signals"] = _active_signals()
    except Exception as exc:
        context["active_signals"] = []

    try:
        context["paper_performance"] = _paper_performance()
    except Exception as exc:
        context["paper_performance"] = {}

    try:
        context["market_regime"] = _market_regime()
    except Exception as exc:
        context["market_regime"] = {}

    try:
        context["model_health"] = _model_health()
    except Exception as exc:
        context["model_health"] = {}

    return context


# ─── Blocs de contexte ───────────────────────────────────────────────────────

def _portfolio_snapshot(user=None) -> dict[str, Any]:
    """Positions actuelles, valeur totale, P&L."""
    try:
        from portfolio.models import Portfolio, PortfolioHolding
        portfolio = Portfolio.objects.filter().order_by("-id").first()
        if not portfolio:
            return {}

        holdings = PortfolioHolding.objects.filter(portfolio=portfolio).select_related("stock")
        positions = []
        total_value = float(portfolio.capital or 0)

        for h in holdings:
            symbol = h.stock.symbol if h.stock else "?"
            shares = float(h.shares or 0)
            avg_cost = float(h.average_cost or 0)

            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                price_data = ticker.history(period="2d")
                current_price = float(price_data["Close"].iloc[-1]) if not price_data.empty else avg_cost
            except Exception:
                current_price = avg_cost

            market_val = shares * current_price
            cost_basis = shares * avg_cost
            pnl_pct = ((current_price - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0
            pnl_abs = market_val - cost_basis
            total_value += market_val

            positions.append({
                "symbol": symbol,
                "shares": round(shares, 2),
                "avg_cost": round(avg_cost, 2),
                "current_price": round(current_price, 2),
                "market_value": round(market_val, 2),
                "pnl_pct": round(pnl_pct, 2),
                "pnl_abs": round(pnl_abs, 2),
                "sector": h.stock.sector if h.stock else None,
            })

        positions.sort(key=lambda x: abs(x["pnl_abs"]), reverse=True)
        winners = [p for p in positions if p["pnl_pct"] > 0]
        losers = [p for p in positions if p["pnl_pct"] < 0]

        return {
            "total_value": round(total_value, 2),
            "cash": round(float(portfolio.capital or 0), 2),
            "n_positions": len(positions),
            "positions": positions[:10],  # top 10 pour pas surcharger le prompt
            "top_winner": positions[0]["symbol"] if winners else None,
            "top_loser": positions[-1]["symbol"] if losers else None,
            "total_pnl_pct": round(
                sum(p["pnl_pct"] * p["market_value"] for p in positions) / max(total_value, 1), 2
            ),
        }
    except Exception as exc:
        logger.debug("_portfolio_snapshot error: %s", exc)
        return {}


def _sandbox_stats() -> dict[str, Any]:
    """Stats des sandboxes paper trading."""
    try:
        from django.core.cache import cache
        from portfolio.models import PaperTrade, SandboxWatchlist
        from django.db import models as dj_models

        result = {}
        for sandbox in ["WATCHLIST", "AI_BLUECHIP", "AI_PENNY"]:
            trades = PaperTrade.objects.filter(sandbox=sandbox, status="CLOSED").order_by("-entry_date")[:100]
            if not trades.exists():
                continue

            pnl_list = [float(t.pnl or 0) for t in trades]
            wins = [p for p in pnl_list if p > 0]
            win_rate = len(wins) / len(pnl_list) * 100 if pnl_list else 0
            total_pnl = sum(pnl_list)

            # Symboles actifs dans le watchlist sandbox
            watchlist = SandboxWatchlist.objects.filter(sandbox=sandbox).first()
            symbols = (watchlist.symbols or [])[:5] if watchlist else []

            result[sandbox] = {
                "win_rate": round(win_rate, 1),
                "total_trades": len(pnl_list),
                "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(total_pnl / max(len(pnl_list), 1), 2),
                "active_symbols": symbols,
            }

        return result
    except Exception as exc:
        logger.debug("_sandbox_stats error: %s", exc)
        return {}


def _active_signals() -> list[dict[str, Any]]:
    """Signaux actifs en cours."""
    try:
        from portfolio.models import ActiveSignal
        from django.utils import timezone

        signals = ActiveSignal.objects.filter(
            created_at__gte=timezone.now() - timedelta(days=2)
        ).order_by("-confidence")[:5]

        return [
            {
                "ticker": s.ticker,
                "pattern": s.pattern or "",
                "entry": float(s.entry_price or 0),
                "target": float(s.target_price or 0),
                "stop": float(s.stop_loss or 0),
                "confidence": float(s.confidence or 0),
            }
            for s in signals
        ]
    except Exception:
        return []


def _paper_performance() -> dict[str, Any]:
    """Performance récente des paper trades."""
    try:
        from portfolio.models import PaperTrade
        from django.utils import timezone

        recent = PaperTrade.objects.filter(
            status="CLOSED",
            entry_date__gte=timezone.now() - timedelta(days=30),
        )
        if not recent.exists():
            return {}

        pnl_list = [float(t.pnl or 0) for t in recent]
        wins = sum(1 for p in pnl_list if p > 0)
        total = len(pnl_list)

        return {
            "last_30d_trades": total,
            "last_30d_win_rate": round(wins / max(total, 1) * 100, 1),
            "last_30d_pnl": round(sum(pnl_list), 2),
            "best_trade": round(max(pnl_list, default=0), 2),
            "worst_trade": round(min(pnl_list, default=0), 2),
        }
    except Exception:
        return {}


def _market_regime() -> dict[str, Any]:
    """Contexte macro actuel (VIX, SPY, TSX)."""
    try:
        from django.core.cache import cache
        cached = cache.get("market_regime_context")
        if cached:
            return cached

        import yfinance as yf
        result = {}
        for sym, key in [("SPY", "spy"), ("^VIX", "vix"), ("^GSPTSE", "tsx")]:
            try:
                hist = yf.Ticker(sym).history(period="5d")
                if hist.empty or "Close" not in hist:
                    continue
                last = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else last
                chg = (last - prev) / prev * 100 if prev else 0
                result[key] = {"price": round(last, 2), "change_pct": round(chg, 2)}
            except Exception:
                continue

        vix_level = result.get("vix", {}).get("price", 20)
        spy_chg = result.get("spy", {}).get("change_pct", 0)

        if vix_level > 30:
            regime = "FEAR"
        elif vix_level > 20:
            regime = "CAUTION"
        elif spy_chg > 0.5:
            regime = "BULL"
        elif spy_chg < -0.5:
            regime = "BEAR"
        else:
            regime = "NEUTRAL"

        result["regime"] = regime
        cache.set("market_regime_context", result, 300)
        return result
    except Exception:
        return {}


def _model_health() -> dict[str, Any]:
    """Santé des modèles ML (win rate, dernière évaluation)."""
    try:
        from portfolio.models import ModelEvaluationDaily
        from django.utils import timezone
        cutoff = timezone.now().date() - timedelta(days=7)

        result = {}
        for model_name in ["BLUECHIP", "PENNY"]:
            evals = ModelEvaluationDaily.objects.filter(
                model_name=model_name,
                as_of__gte=cutoff,
            ).order_by("-as_of")
            if not evals.exists():
                continue
            latest = evals.first()
            result[model_name] = {
                "win_rate": round(float(latest.win_rate or 0), 1) if latest else None,
                "brier_score": round(float(latest.brier_score or 0), 3) if latest else None,
                "as_of": str(latest.as_of) if latest else None,
            }
        return result
    except Exception:
        return {}


# ─── Génération du bloc de contexte pour le prompt ───────────────────────────

def context_to_prompt(context: dict[str, Any], question: str = "") -> str:
    """
    Convertit le contexte en bloc texte structuré pour le prompt DeepSeek.

    Le bloc est optimisé pour être concis mais complet :
    - Positions en P&L
    - Stats sandbox
    - Signaux actifs
    - Régime marché
    - Santé modèles
    """
    lines = ["## CONTEXTE PORTFOLIO RÉEL\n"]

    # Portfolio
    pf = context.get("portfolio") or {}
    if pf:
        lines.append(f"**Valeur totale:** ${pf.get('total_value', 0):,.2f} | Cash: ${pf.get('cash', 0):,.2f}")
        lines.append(f"**Positions ({pf.get('n_positions', 0)}):**")
        for pos in (pf.get("positions") or [])[:6]:
            pnl_sign = "+" if pos["pnl_pct"] >= 0 else ""
            lines.append(
                f"  • {pos['symbol']}: {pos['shares']} actions @ ${pos['avg_cost']:.2f} | "
                f"Actuel ${pos['current_price']:.2f} | P&L: {pnl_sign}{pos['pnl_pct']:.1f}% "
                f"(${pnl_sign}{pos['pnl_abs']:,.0f})"
            )
        lines.append("")

    # Sandboxes
    sandboxes = context.get("sandboxes") or {}
    if sandboxes:
        lines.append("**Sandboxes paper trading:**")
        for sb, stats in sandboxes.items():
            lines.append(
                f"  • {sb}: {stats.get('total_trades', 0)} trades | "
                f"Win rate {stats.get('win_rate', 0):.1f}% | "
                f"P&L total ${stats.get('total_pnl', 0):+,.2f}"
            )
            if stats.get("active_symbols"):
                lines.append(f"    Watchlist: {', '.join(stats['active_symbols'])}")
        lines.append("")

    # Signaux actifs
    signals = context.get("active_signals") or []
    if signals:
        lines.append("**Signaux actifs:**")
        for sig in signals[:3]:
            lines.append(
                f"  • {sig['ticker']}: entry ${sig['entry']:.2f} | "
                f"target ${sig['target']:.2f} | stop ${sig['stop']:.2f} | "
                f"confiance {sig['confidence']:.0f}%"
            )
        lines.append("")

    # Performance récente
    perf = context.get("paper_performance") or {}
    if perf:
        lines.append(
            f"**Performance 30j:** {perf.get('last_30d_trades', 0)} trades | "
            f"{perf.get('last_30d_win_rate', 0):.1f}% win rate | "
            f"P&L ${perf.get('last_30d_pnl', 0):+,.2f}"
        )
        lines.append("")

    # Régime marché
    regime = context.get("market_regime") or {}
    if regime:
        lines.append(f"**Régime marché:** {regime.get('regime', 'N/A')}")
        spy = regime.get("spy") or {}
        vix = regime.get("vix") or {}
        tsx = regime.get("tsx") or {}
        if spy:
            lines.append(f"  SPY: ${spy.get('price', 0):.2f} ({spy.get('change_pct', 0):+.2f}%)")
        if vix:
            lines.append(f"  VIX: {vix.get('price', 0):.1f}")
        if tsx:
            lines.append(f"  TSX: ${tsx.get('price', 0):,.0f} ({tsx.get('change_pct', 0):+.2f}%)")
        lines.append("")

    # Santé modèles
    models = context.get("model_health") or {}
    if models:
        lines.append("**Modèles ML:**")
        for m, stats in models.items():
            lines.append(
                f"  • {m}: win rate {stats.get('win_rate', 0):.1f}% | "
                f"Brier {stats.get('brier_score', 0):.3f} | "
                f"Évalué le {stats.get('as_of', 'N/A')}"
            )
        lines.append("")

    lines.append("---")
    lines.append(f"**Question:** {question}\n")

    return "\n".join(lines)


# ─── Patch pour AskTheQuantStreamView dans views.py ──────────────────────────
#
# Dans portfolio/views.py → AskTheQuantStreamView.post() :
#
# AVANT :
#   def post(self, request):
#       question = request.data.get("question", "")
#       symbol = request.data.get("symbol", "GENERAL")
#       advisor = DeepSeekAdvisor()
#       def stream():
#           for chunk in advisor.stream_answer(symbol, question):
#               yield f"data: {chunk}\n\n"
#       return StreamingHttpResponse(stream(), content_type="text/event-stream")
#
# APRÈS :
#   from .quant_context_builder import build_quant_context, context_to_prompt
#
#   def post(self, request):
#       question = request.data.get("question", "")
#       symbol = request.data.get("symbol", "GENERAL")
#
#       # Construire le contexte portfolio
#       include_context = request.data.get("include_context", True)
#       context_block = ""
#       if include_context:
#           try:
#               ctx = build_quant_context(user=request.user, question=question)
#               context_block = context_to_prompt(ctx, question)
#           except Exception:
#               context_block = ""
#
#       # Question enrichie avec le contexte
#       enriched_question = f"{context_block}\n{question}" if context_block else question
#
#       advisor = DeepSeekAdvisor()
#       def stream():
#           for chunk in advisor.stream_answer(symbol, enriched_question):
#               yield f"data: {chunk}\n\n"
#
#       return StreamingHttpResponse(stream(), content_type="text/event-stream")
