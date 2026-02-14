import { useEffect, useState } from 'react';
import api from '../api/api';

function LivePaperTrading() {
  const [summary, setSummary] = useState(null);

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const res = await api.get('paper-trades/summary/');
        if (!active) return;
        setSummary(res.data);
      } catch (err) {
        if (!active) return;
        setSummary(null);
      }
    };

    load();
    const interval = setInterval(load, 60000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  if (!summary) {
    return (
      <div className="text-slate-400">Paper trading data unavailable.</div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <p className="text-xs text-slate-400 uppercase tracking-[0.2em]">Capital initial</p>
          <p className="text-2xl text-white font-semibold">${summary.initial_capital}</p>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <p className="text-xs text-slate-400 uppercase tracking-[0.2em]">Capital disponible</p>
          <p className="text-2xl text-white font-semibold">${summary.available_capital}</p>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <p className="text-xs text-slate-400 uppercase tracking-[0.2em]">P&L réalisé</p>
          <p className={`text-2xl font-semibold ${summary.closed_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            ${summary.closed_pnl}
          </p>
        </div>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
        <h3 className="text-white font-semibold mb-4">Positions ouvertes</h3>
        {summary.open_positions?.length ? (
          <div className="space-y-3">
            {summary.open_positions.map((trade) => (
              <div key={`${trade.ticker}-${trade.entry_date}`} className="flex items-center justify-between text-sm text-slate-300">
                <span className="text-white font-semibold">{trade.ticker}</span>
                <span>Entrée ${trade.entry_price}</span>
                <span>Stop ${trade.stop_loss}</span>
                <span>Qté {trade.quantity}</span>
                <span className="text-slate-400">{trade.notes || '—'}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-slate-400 text-sm">Aucune position ouverte.</p>
        )}
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
        <h3 className="text-white font-semibold mb-4">Positions clôturées (25 dernières)</h3>
        {summary.closed_positions?.length ? (
          <div className="space-y-3">
            {summary.closed_positions.map((trade) => (
              <div key={`${trade.ticker}-${trade.exit_date}`} className="flex items-center justify-between text-sm text-slate-300">
                <span className="text-white font-semibold">{trade.ticker}</span>
                <span>Entrée ${trade.entry_price}</span>
                <span>Sortie ${trade.exit_price ?? '—'}</span>
                <span className={Number(trade.pnl) >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                  P&L ${trade.pnl}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-slate-400 text-sm">Aucune position clôturée.</p>
        )}
      </div>
    </div>
  );
}

export default LivePaperTrading;
