import { useEffect, useState } from 'react';
import { cachedGet } from '../api/cachedApi';

function LivePaperTrading() {
  const [summary, setSummary] = useState(null);
  const [selectedTrade, setSelectedTrade] = useState(null);
  const [loading, setLoading] = useState(true);
  const [livePositions, setLivePositions] = useState({});

  const formatExplanation = (explanations = []) => {
    if (!Array.isArray(explanations) || explanations.length === 0) return '—';
    return explanations
      .map((item) => `${item.feature}: ${Number(item.contribution || 0).toFixed(1)}%`)
      .join(' · ');
  };

  const formatValue = (value) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
    return Number(value).toFixed(4);
  };

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const data = await cachedGet('paper-trades/summary/', {}, 30000);
        if (!active) return;
        setSummary(data);
      } catch (err) {
        if (!active) return;
        setSummary(null);
      } finally {
        if (active) setLoading(false);
      }
    };

    load();
    const interval = setInterval(load, 60000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    const wsEnabled = (process.env.REACT_APP_WS_UPDATES || '').toLowerCase() === 'true';
    if (!wsEnabled) return undefined;

    let socket;
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${protocol}://${window.location.hostname}:8001/ws/updates/`;

    try {
      socket = new WebSocket(wsUrl);
      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          const positions = payload?.positions || [];
          const map = {};
          positions.forEach((pos) => {
            map[pos.ticker] = pos;
          });
          setLivePositions(map);
        } catch (err) {
          // ignore parse errors
        }
      };
    } catch (err) {
      // ignore socket errors
    }

    return () => {
      if (socket) socket.close();
    };
  }, []);

  if (loading) {
    return (
      <div className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[1, 2, 3].map((item) => (
            <div key={item} className="h-24 rounded-2xl border border-slate-800 bg-slate-900/60 animate-pulse" />
          ))}
        </div>
        <div className="h-40 rounded-2xl border border-slate-800 bg-slate-900/60 animate-pulse" />
        <div className="h-40 rounded-2xl border border-slate-800 bg-slate-900/60 animate-pulse" />
      </div>
    );
  }

  if (!summary) {
    return (
      <div className="text-slate-400">Paper trading data unavailable.</div>
    );
  }

  return (
    <>
      <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <p className="text-xs text-slate-400 uppercase tracking-[0.2em]">Capital initial</p>
          <p className="text-2xl text-white font-semibold">${summary.initial_capital}</p>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <p className="text-xs text-slate-400 uppercase tracking-[0.2em]">Capital disponible</p>
          <p className="text-2xl text-white font-semibold">${summary.available_capital}</p>
          <p className="mt-1 text-xs text-rose-400">Total Risk: ${summary.total_risk ?? 0}</p>
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
              <div key={`${trade.ticker}-${trade.entry_date}`} className="space-y-2 rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                <div className="flex flex-wrap items-center justify-between gap-2 text-sm text-slate-300">
                  <span className="text-white font-semibold">{trade.ticker}</span>
                  <span>Entrée ${trade.entry_price}</span>
                  <span>Actuel ${livePositions[trade.ticker]?.current_price ?? '—'}</span>
                  <span>Stop ${trade.stop_loss}</span>
                  <span>Qté {trade.quantity}</span>
                  <span className={Number(livePositions[trade.ticker]?.unrealized_pnl) >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                    U-P&L {livePositions[trade.ticker]?.unrealized_pnl ?? '—'}
                  </span>
                  <button
                    type="button"
                    onClick={() => setSelectedTrade(trade)}
                    className="text-xs text-indigo-300 hover:text-indigo-200"
                  >
                    Détails
                  </button>
                </div>
                <div className="text-xs text-slate-400">
                  Explications: {formatExplanation(trade.entry_explanations)}
                </div>
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
              <div key={`${trade.ticker}-${trade.exit_date}`} className="space-y-2 rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                <div className="flex flex-wrap items-center justify-between gap-2 text-sm text-slate-300">
                  <span className="text-white font-semibold">{trade.ticker}</span>
                  <span>Entrée ${trade.entry_price}</span>
                  <span>Sortie ${trade.exit_price ?? '—'}</span>
                  <span className={Number(trade.pnl) >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                    P&L ${trade.pnl}
                  </span>
                  <button
                    type="button"
                    onClick={() => setSelectedTrade(trade)}
                    className="text-xs text-indigo-300 hover:text-indigo-200"
                  >
                    Détails
                  </button>
                </div>
                <div className="text-xs text-slate-400">
                  Explications: {formatExplanation(trade.entry_explanations)}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-slate-400 text-sm">Aucune position clôturée.</p>
        )}
      </div>
      </div>
      {selectedTrade ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <div className="w-full max-w-2xl rounded-2xl border border-slate-800 bg-slate-900 p-6">
            <div className="flex items-center justify-between">
              <h3 className="text-white font-semibold">Détails du trade · {selectedTrade.ticker}</h3>
              <button
                type="button"
                onClick={() => setSelectedTrade(null)}
                className="text-slate-400 hover:text-slate-200"
              >
                Fermer
              </button>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3 text-sm text-slate-300">
              <div>Signal: {formatValue(selectedTrade.entry_signal)}</div>
              <div>Sandbox: {selectedTrade.sandbox}</div>
              <div>Modèle: {selectedTrade.model_name}</div>
              <div>Version: {selectedTrade.model_version || '—'}</div>
              <div>Entrée: ${selectedTrade.entry_price}</div>
              <div>Stop: ${selectedTrade.stop_loss}</div>
              <div>Qté: {selectedTrade.quantity}</div>
              <div>Notes: {selectedTrade.notes || '—'}</div>
            </div>

            <div className="mt-5">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Top explications</p>
              <div className="mt-2 space-y-2">
                {(selectedTrade.entry_explanations || []).length ? (
                  selectedTrade.entry_explanations.map((item) => (
                    <div key={item.feature} className="flex items-center justify-between text-sm text-slate-200">
                      <span>{item.feature}</span>
                      <span>{Number(item.contribution || 0).toFixed(2)}%</span>
                    </div>
                  ))
                ) : (
                  <p className="text-sm text-slate-400">Aucune explication disponible.</p>
                )}
              </div>
            </div>

            <div className="mt-5">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Features (snapshot)</p>
              <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-slate-300">
                {selectedTrade.entry_features ? (
                  Object.entries(selectedTrade.entry_features).map(([key, value]) => (
                    <div key={key} className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950/60 px-2 py-1">
                      <span>{key}</span>
                      <span>{formatValue(value)}</span>
                    </div>
                  ))
                ) : (
                  <p className="text-sm text-slate-400">Aucun snapshot disponible.</p>
                )}
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}

export default LivePaperTrading;
