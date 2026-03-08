from __future__ import annotations

import json

from portfolio.tasks import (
    _normalize_yf_symbol,
    _intraday_context_for_timeframe,
    _has_hammer_pattern,
    _rsi_divergence,
    _send_telegram_alert,
)
from portfolio.ai_advisor import DeepSeekAdvisor


def dry_run_test(symbol: str = "SHOP.TO") -> None:
    symbol = _normalize_yf_symbol(symbol)
    ctx_5m = _intraday_context_for_timeframe(symbol, minutes=390, timeframe=5, rvol_window=20)
    if not ctx_5m:
        print("❌ Impossible de récupérer le contexte 5m.")
        return

    bars = ctx_5m.get('bars')
    if bars is None or bars.empty:
        print("❌ Aucune barre intraday.")
        return

    price = float(bars.iloc[-1]['close'])
    rvol = float(ctx_5m.get('rvol') or 0)
    day_change = float(ctx_5m.get('day_change_pct') or 0)
    is_hammer = _has_hammer_pattern(ctx_5m)
    has_div = _rsi_divergence(ctx_5m)

    advisor = DeepSeekAdvisor()
    prompt = (
        f"Analyse {symbol}. Prix: {price:.2f}. RVOL: {rvol:.2f}. "
        f"Hammer: {is_hammer}. RSI divergence: {has_div}. "
        "Donne un verdict court pour un trade Wealthsimple."
    )
    chunks = []
    for item in advisor.stream_answer(symbol, prompt):
        try:
            payload = json.loads(item)
        except Exception:
            payload = {'text': str(item)}
        chunks.append(payload.get('text') or '')
        if payload.get('done'):
            break
    ai_text = "".join(chunks).strip()

    ws_symbol = symbol.split('.')[0]
    ws_link = f"https://my.wealthsimple.com/app/invest/search?query={ws_symbol}"
    message = (
        f"🇨🇦 *TEST SIGNAL : {symbol}*\n"
        "━━━━━━━━━━━━━━━\n"
        f"💰 *Prix:* {price:.2f}$ ({day_change:.2f}%)\n"
        f"📊 *RVOL:* {rvol:.2f}\n"
        f"🛠️ *Tech:* {'Hammer ✅' if is_hammer else 'Normal'} | RSI Div {'✅' if has_div else '—'}\n"
        "━━━━━━━━━━━━━━━\n"
        f"🧠 *ANALYSE DANAS :*\n_{ai_text[:400]}_\n\n"
        f"🔗 [Ouvrir Wealthsimple]({ws_link})"
    )

    _send_telegram_alert(message, allow_during_blackout=True, category='report')
    print("✅ Test envoyé sur Telegram.")


if __name__ == '__main__':
    dry_run_test("SHOP.TO")
