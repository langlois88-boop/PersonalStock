import { useEffect, useState } from 'react';
import { cachedGet } from '../api/cachedApi';
import { Term } from './ui/Tooltip';
import { GLOSSARY } from '../glossary';
import { getFeatureExplanation } from '../featureGlossary';

function LivePaperTrading() {
  const [summary, setSummary] = useState(null);
  const [selectedTrade, setSelectedTrade] = useState(null);
  const [loading, setLoading] = useState(true);
  const [livePositions, setLivePositions] = useState({});
  const [brokerFilter, setBrokerFilter] = useState('ALL');
  const [sandboxFilter, setSandboxFilter] = useState('ALL');

  // Was a plain string ("MA20: 99.4% · vol_regime: 0.5% · ...") -- raw
  // ML feature codes with zero explanation in the compact card view (the
  // detail modal already used getFeatureExplanation/Term for this, see
  // lines below; this summary line didn't). Same visible text, now each
  // feature gets the same hoverable "(i)" plain-language explanation
  // instead of requiring a click into "Détails" to find out what it means.
  const formatExplanation = (explanations = []) => {
    if (!Array.isArray(explanations) || explanations.length === 0) return '—';
    return explanations.map((item, idx) => (
      <span key={`${item.feature}-${idx}`}>
        {idx > 0 && ' · '}
        <Term text={getFeatureExplanation(item.feature)}>
          <span>{item.feature}: {Number(item.contribution || 0).toFixed(1)}%</span>
        </Term>
      </span>
    ));
  };

  // Le backend écrit la raison de sortie (stop-loss touché, signal IA,
  // sortie temps, circuit breaker...) directement dans le champ `notes`,
  // séparée par ' | '. On la nettoie pour l'afficher clairement.
  const formatExitReason = (notes) => {
    if (!notes) return null;
    const cleaned = String(notes)
      .split('|')
      .map((s) => s.trim())
      .filter(Boolean)
      .join(' · ');
    return cleaned || null;
  };

  const formatValue = (value) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
    return Number(value).toFixed(4);
  };

  const formatMoney = (value) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
    return `$${Number(value).toFixed(2)}`;
  };

  // Coût total = ce qui a été payé à l'achat (entrée × quantité).
  const tradeCost = (trade) => Number(trade.entry_price || 0) * Number(trade.quantity || 0);
  // Valeur de revente = ce que la vente a rapporté (sortie × quantité).
  const tradeResaleValue = (trade) =>
    trade.exit_price != null ? Number(trade.exit_price) * Number(trade.quantity || 0) : null;

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const params = {};
        if (brokerFilter !== 'ALL') params.broker = brokerFilter;
        if (sandboxFilter !== 'ALL') params.sandbox = sandboxFilter;
        const data = await cachedGet('paper-trades/summary/', params, 30000);
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
  }, [brokerFilter, sandboxFilter]);

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

  const unrealizedPnl = (summary.open_positions || []).reduce(
    (sum, pos) => sum + (Number.isFinite(pos.unrealized_pnl) ? pos.unrealized_pnl : 0),
    0
  );
  const totalValue = Number(summary.initial_capital || 0) + Number(summary.closed_pnl || 0) + unrealizedPnl;
  const totalGainPct = summary.initial_capital
    ? ((totalValue - summary.initial_capital) / summary.initial_capital) * 100
    : 0;

  return (
    <>
      <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
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
          <p className="text-xs text-slate-400 uppercase tracking-[0.2em]">Total investi (positions ouvertes)</p>
          <p className="text-2xl text-white font-semibold">${Number(summary.open_value ?? 0).toFixed(2)}</p>
          <p className="mt-1 text-xs text-slate-500">Coût d'entrée des positions ouvertes, pas leur valeur au marché.</p>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <p className="text-xs text-slate-400 uppercase tracking-[0.2em]"><Term text={GLOSSARY.pnlRealized}>P&L réalisé</Term></p>
          <p className={`text-2xl font-semibold ${summary.closed_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            ${summary.closed_pnl}
          </p>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <p className="text-xs text-slate-400 uppercase tracking-[0.2em]">Valeur totale (avec profit)</p>
          <p className={`text-2xl font-semibold ${totalValue >= summary.initial_capital ? 'text-emerald-400' : 'text-red-400'}`}>
            ${totalValue.toFixed(2)}
          </p>
          <p className={`mt-1 text-xs ${totalGainPct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {totalGainPct >= 0 ? '+' : ''}{totalGainPct.toFixed(2)}% depuis le départ
          </p>
        </div>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-white font-semibold">Positions ouvertes</h3>
          <div className="flex items-center gap-2">
            <select
              value={sandboxFilter}
              onChange={(event) => setSandboxFilter(event.target.value)}
              className="bg-slate-950/60 border border-slate-800 text-slate-200 text-xs rounded-lg px-2 py-1"
            >
              <option value="ALL">Sandbox: Toutes</option>
              <option value="WATCHLIST">WATCHLIST</option>
              <option value="AI_BLUECHIP">AI_BLUECHIP</option>
              <option value="AI_PENNY">AI_PENNY</option>
              <option value="AI_CRYPTO">AI_CRYPTO</option>
            </select>
            <select
              value={brokerFilter}
              onChange={(event) => setBrokerFilter(event.target.value)}
              className="bg-slate-950/60 border border-slate-800 text-slate-200 text-xs rounded-lg px-2 py-1"
            >
              <option value="ALL">Broker: All</option>
              <option value="SIM">Broker: SIM</option>
              <option value="ALPACA">Broker: Alpaca</option>
            </select>
          </div>
        </div>
        {summary.open_positions?.length ? (
          <div className="space-y-3">
            {summary.open_positions.map((trade) => (
              <div key={`${trade.ticker}-${trade.entry_date}`} className="space-y-2 rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                <div className="flex flex-wrap items-center justify-between gap-2 text-sm text-slate-300">
                  <span className="text-white font-semibold">{trade.ticker}</span>
                  <span>Entrée ${trade.entry_price}</span>
                  <span>
                    Actuel ${livePositions[trade.ticker]?.current_price ?? trade.current_price ?? '—'}
                  </span>
                  <span>Stop ${trade.stop_loss}</span>
                  <span>Qté {trade.quantity}</span>
                  <span
                    className={Number(
                      livePositions[trade.ticker]?.unrealized_pnl ?? trade.unrealized_pnl
                    ) >= 0 ? 'text-emerald-400' : 'text-red-400'}
                  >
                    U-P&L {livePositions[trade.ticker]?.unrealized_pnl ?? trade.unrealized_pnl ?? '—'}
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
                  Coût total {formatMoney(tradeCost(trade))} ({trade.quantity} × ${trade.entry_price})
                  {' · '}
                  Valeur de revente {formatMoney(tradeResaleValue(trade))}
                </div>
                <div className="text-xs text-slate-400">
                  Explications: {formatExplanation(trade.entry_explanations)}
                </div>
                {formatExitReason(trade.notes) && (
                  <div className="text-xs text-amber-300/90">
                    Pourquoi vendu : {formatExitReason(trade.notes)}
                  </div>
                )}
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
              <div>Coût total : {formatMoney(tradeCost(selectedTrade))} ({selectedTrade.quantity} × ${selectedTrade.entry_price})</div>
              <div>Valeur de revente : {formatMoney(tradeResaleValue(selectedTrade))}</div>
              {selectedTrade.status === 'CLOSED' && (
                <div>
                  P&L : <span className={Number(selectedTrade.pnl || 0) >= 0 ? 'text-emerald-400 font-semibold' : 'text-red-400 font-semibold'}>
                    {formatMoney(selectedTrade.pnl)}
                  </span>
                </div>
              )}
              {selectedTrade.status !== 'CLOSED' && (selectedTrade.unrealized_pnl !== undefined) && (
                <div>
                  U-P&L : <span className={Number(selectedTrade.unrealized_pnl || 0) >= 0 ? 'text-emerald-400 font-semibold' : 'text-red-400 font-semibold'}>
                    {formatMoney(selectedTrade.unrealized_pnl)}
                  </span>
                </div>
              )}
              <div className="col-span-2">Pourquoi vendu : {formatExitReason(selectedTrade.notes) || (selectedTrade.status === 'CLOSED' ? 'Aucune information enregistrée.' : '—')}</div>
            </div>

            <div className="mt-5">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400"><Term text={GLOSSARY.topExplanations}>Top explications</Term></p>
              <div className="mt-2 space-y-2">
                {(selectedTrade.entry_explanations || []).length ? (
                  selectedTrade.entry_explanations.map((item) => {
                    const contribution = Number(item.contribution || 0);
                    return (
                      <div key={item.feature} className="flex items-center justify-between text-sm text-slate-200">
                        <Term text={getFeatureExplanation(item.feature)}><span>{item.feature}</span></Term>
                        <span className={contribution >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                          {contribution >= 0 ? '+' : ''}{contribution.toFixed(2)}%
                        </span>
                      </div>
                    );
                  })
                ) : (
                  <p className="text-sm text-slate-400">Aucune explication disponible.</p>
                )}
              </div>
            </div>

            <div className="mt-5">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400"><Term text={GLOSSARY.featuresSnapshot}>Features (snapshot)</Term></p>
              <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-slate-300">
                {selectedTrade.entry_features ? (
                  Object.entries(selectedTrade.entry_features).map(([key, value]) => (
                    <div key={key} className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950/60 px-2 py-1">
                      <Term text={getFeatureExplanation(key)}><span>{key}</span></Term>
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
