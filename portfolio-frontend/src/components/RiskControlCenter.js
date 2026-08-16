import { useEffect, useMemo, useState } from 'react';
import { Gauge, ShieldCheck, TrendingDown } from 'lucide-react';
import api from '../api/api';
import { Term } from './ui/Tooltip';
import { GLOSSARY } from '../glossary';

const sandboxLabels = {
  WATCHLIST: 'Watchlist',
  AI_BLUECHIP: 'AI Bluechip',
  AI_PENNY: 'AI Penny',
};

function RiskControlCenter() {
  // Was 4 editable sliders saved only to localStorage -- nothing in the
  // backend ever read them, so changing them changed exactly nothing
  // about real trading behavior (confirmed live 2026-08-16, reported by
  // Eric). Replaced with a read-only snapshot of the real per-sandbox
  // parameters, fetched from RiskSnapshotView -- Eric explicitly chose
  // this over building a real write path, to keep zero risk on the live
  // trading path for now.
  const [snapshot, setSnapshot] = useState(null);
  const [snapshotError, setSnapshotError] = useState(null);
  const [summary, setSummary] = useState(null);
  const [recentTrades, setRecentTrades] = useState([]);

  useEffect(() => {
    api
      .get('risk-snapshot/')
      .then((res) => setSnapshot(res.data))
      .catch(() => setSnapshotError('Impossible de charger les paramètres réels.'));

    api
      .get('paper-trades/summary/')
      .then((res) => setSummary(res.data))
      .catch(() => setSummary(null));

    api
      .get('paper-trades/', { params: { page_size: 10 } })
      .then((res) => {
        const data = res.data?.results || res.data || [];
        setRecentTrades(Array.isArray(data) ? data.slice(0, 10) : []);
      })
      .catch(() => setRecentTrades([]));
  }, []);

  const exposurePct = useMemo(() => {
    if (!summary) return 0;
    const openValue = Number(summary.open_value || 0);
    const available = Number(summary.available_capital || 0);
    const total = openValue + available;
    if (!total) return 0;
    return Math.min(100, Math.round((openValue / total) * 100));
  }, [summary]);

  const riskStatus = exposurePct > 70 ? 'Élevé' : exposurePct > 40 ? 'Modéré' : 'Contrôlé';
  const drawdownPct = snapshot?.daily_drawdown_circuit_breaker_pct;

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <p className="text-xs uppercase tracking-[0.3em] text-slate-500">Risk Manager</p>
        <h2 className="text-2xl font-semibold text-white">Risk Control Center</h2>
        <p className="text-sm text-slate-400">Surveille l'exposition en temps réel et affiche les vrais paramètres de risque actifs.</p>
      </header>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-2xl bg-slate-900/70 p-5 border border-slate-800">
          <div className="flex items-center gap-2 text-slate-200">
            <Gauge size={18} />
            <span className="text-sm font-semibold">Risque global</span>
          </div>
          <p className="mt-3 text-3xl font-semibold text-white">{riskStatus}</p>
          <p className="text-xs text-slate-400">Exposition actuelle : {exposurePct}%</p>
        </div>
        <div className="rounded-2xl bg-slate-900/70 p-5 border border-slate-800">
          <div className="flex items-center gap-2 text-slate-200">
            <ShieldCheck size={18} />
            <span className="text-sm font-semibold">Cash disponible</span>
          </div>
          <p className="mt-3 text-3xl font-semibold text-emerald-300">${summary?.available_capital || 0}</p>
          <p className="text-xs text-slate-400">Capital engagé : ${summary?.open_value || 0}</p>
        </div>
        <div className="rounded-2xl bg-slate-900/70 p-5 border border-slate-800">
          <div className="flex items-center gap-2 text-slate-200">
            <TrendingDown size={18} />
            <Term text={GLOSSARY.dailyStopLoss}><span className="text-sm font-semibold">Daily stop-loss (réel)</span></Term>
          </div>
          <p className="mt-3 text-3xl font-semibold text-red-300">
            {drawdownPct != null ? `${drawdownPct.toFixed(1)}%` : '—'}
          </p>
          <p className="text-xs text-slate-400">Circuit breaker réel (DAILY_EQUITY_DRAWDOWN_PCT) — arrête les nouvelles entrées du sandbox pour la journée.</p>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="rounded-2xl bg-slate-900/70 p-5 border border-slate-800 space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-slate-200">Paramètres réels actifs (lecture seule)</h3>
            <p className="text-xs text-slate-500 mt-1">
              Ces valeurs sont lues en direct depuis le serveur — ce ne sont plus des curseurs modifiables.
              Changer l'exécution réelle demande une décision dédiée (ça touche les ordres réels), pas un réglage
              rapide ici.
            </p>
          </div>
          {snapshotError ? (
            <p className="text-sm text-rose-300">{snapshotError}</p>
          ) : !snapshot ? (
            <div className="h-32 rounded-xl bg-slate-900/60 animate-pulse" />
          ) : (
            <>
              {snapshot.sandboxes.some((s) => s.exploration_override_active) && (
                <p className="text-xs text-amber-300/90 bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2">
                  Phase d'exploration active — les seuils normaux par sandbox ci-dessous sont actuellement
                  écrasés par un seuil unique commun.
                </p>
              )}
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm text-slate-200">
                  <thead className="text-slate-400 text-xs">
                    <tr className="border-b border-slate-800">
                      <th className="text-left py-2 pr-3">Sandbox</th>
                      <th className="text-left py-2 pr-3">Seuil d'achat</th>
                      <th className="text-left py-2">Seuil de vente</th>
                    </tr>
                  </thead>
                  <tbody>
                    {snapshot.sandboxes.map((row) => (
                      <tr key={row.sandbox} className="border-b border-slate-800/60">
                        <td className="py-2 pr-3 font-semibold">{sandboxLabels[row.sandbox] || row.sandbox}</td>
                        <td className="py-2 pr-3">{row.buy_threshold.toFixed(2)}</td>
                        <td className="py-2">{row.sell_threshold.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2">
                  <p className="text-slate-500 uppercase tracking-[0.15em]">Mise min</p>
                  <p className="text-slate-200 font-semibold mt-1">${snapshot.dynamic_position_min}</p>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2">
                  <p className="text-slate-500 uppercase tracking-[0.15em]">Mise max</p>
                  <p className="text-slate-200 font-semibold mt-1">${snapshot.dynamic_position_max}</p>
                </div>
              </div>
            </>
          )}
        </section>

        <section className="rounded-2xl bg-slate-900/70 p-5 border border-slate-800 space-y-4">
          <h3 className="text-sm font-semibold text-slate-200">
            <Term text={`${GLOSSARY.confidence} • ${GLOSSARY.pnlRealized}`}>Historique confidence vs P&L</Term>
          </h3>
          {recentTrades.length === 0 ? (
            <p className="text-sm text-slate-400">Pas encore de trades à afficher.</p>
          ) : (
            <div className="space-y-3">
              {recentTrades.map((trade) => (
                <div
                  key={trade.id || `${trade.ticker}-${trade.entry_date}`}
                  className="flex items-center justify-between rounded-xl bg-slate-950/70 px-4 py-3 text-sm"
                >
                  <div>
                    <p className="text-slate-200 font-semibold">{trade.ticker}</p>
                    <p className="text-xs text-slate-500">Confidence: {(Number(trade.entry_signal || 0) * 100).toFixed(1)}%</p>
                  </div>
                  <span className={`font-semibold ${Number(trade.pnl || 0) >= 0 ? 'text-emerald-300' : 'text-red-300'}`}>
                    {Number(trade.pnl || 0).toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

export default RiskControlCenter;
