import { motion } from 'framer-motion';
import { useCallback, useEffect, useMemo, useState } from 'react';
import AIRecommendations from './AIRecommendations';
import api from '../api/api';

function OptimizerPage() {
    const normalizeScore = (value) => {
      const raw = Number(value || 0);
      if (!Number.isFinite(raw)) return 0;
      return raw <= 1 ? raw * 100 : raw;
    };
    const scoreColor = (value) => {
      const score = normalizeScore(value);
      if (score >= 85) return 'text-emerald-400';
      if (score >= 70) return 'text-sky-300';
      return 'text-amber-300';
    };
    const signalLabel = (item) => {
      const score = normalizeScore(item.ai_score ?? item.confidence);
      const volumeZ = Number(item.volume_z ?? 0);
      const winRateRaw = Number(item.win_rate ?? 0);
      const winRate = winRateRaw <= 1 ? winRateRaw * 100 : winRateRaw;
      if (winRate && winRate < 50) return { text: 'STATISTIQUEMENT FAIBLE', className: 'bg-slate-700 text-slate-200 border border-slate-500/40' };
      if (score >= 85 && volumeZ > 0.5) return { text: 'STRONG BUY', className: 'bg-emerald-500/20 text-emerald-200 border border-emerald-500/40' };
      if (score >= 70) return { text: 'HOLD / KEEP', className: 'bg-sky-500/20 text-sky-200 border border-sky-500/40' };
      return { text: 'WAITING', className: 'bg-amber-500/20 text-amber-200 border border-amber-500/40' };
    };
  const [actions, setActions] = useState([]);
  const [suggestions, setSuggestions] = useState([]);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState('core');

  const loadOptimizer = useCallback(async () => {
    let isMounted = true;
    setIsRefreshing(true);

    const applyPayload = (payload) => {
      if (!isMounted) return;
      setActions(Array.isArray(payload?.actions) ? payload.actions : []);
      setSuggestions(Array.isArray(payload?.suggestions) ? payload.suggestions : []);
    };

    try {
      const res = await api.get('optimizer/');
      const payload = res?.data || {};
      if (Array.isArray(payload.actions)) {
        applyPayload(payload);
        return;
      }
    } catch (err) {
      // fall through to direct fetch
    }

    try {
      const fallbackUrl = `${window.location.protocol}//${window.location.hostname}:8001/api/optimizer/`;
      const fallbackRes = await fetch(fallbackUrl);
      const fallbackPayload = await fallbackRes.json();
      applyPayload(fallbackPayload);
    } catch (err) {
      applyPayload({ actions: [], suggestions: [] });
    } finally {
      if (isMounted) setIsRefreshing(false);
    }

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    loadOptimizer();
  }, [loadOptimizer]);

  const coreActions = useMemo(
    () =>
      actions.filter((item) => {
        const price = Number(item.price ?? 0);
        const marketCap = Number(item.market_cap ?? 0);
        return price > 5 && marketCap > 2_000_000_000;
      }),
    [actions],
  );
  const moonshotActions = useMemo(
    () =>
      actions.filter((item) => {
        const price = Number(item.price ?? 0);
        const marketCap = Number(item.market_cap ?? 0);
        return price <= 5 || marketCap <= 2_000_000_000;
      }),
    [actions],
  );
  const hasActions = useMemo(() => actions.length > 0, [actions.length]);
  const visibleActions = activeTab === 'core' ? coreActions : moonshotActions;

  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
      <div className="xl:col-span-2 space-y-4">
        {!hasActions ? (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 text-slate-400">
            Aucune position détectée. Ajoute des transactions pour obtenir des recommandations.
          </div>
        ) : null}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setActiveTab('core')}
            className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-colors ${
              activeTab === 'core'
                ? 'bg-indigo-500/20 text-indigo-100 border-indigo-400/40'
                : 'bg-slate-800 text-slate-300 border-slate-700'
            }`}
          >
            Core Portfolio ({coreActions.length})
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('moonshots')}
            className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-colors ${
              activeTab === 'moonshots'
                ? 'bg-rose-500/20 text-rose-100 border-rose-400/40'
                : 'bg-slate-800 text-slate-300 border-slate-700'
            }`}
          >
            Moonshots ({moonshotActions.length})
          </button>
        </div>
        {visibleActions.map((item, idx) => (
          <motion.div
            key={item.ticker}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: idx * 0.05 }}
            className="bg-slate-900 border border-slate-800 rounded-2xl p-5 relative group/action"
          >
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <p className="text-white text-lg font-semibold">{item.ticker}</p>
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-700 text-slate-300 border border-slate-600">
                    {item.type}
                  </span>
                </div>
                <p className="text-xs text-slate-400">{item.name}</p>
                <p className="text-xs text-slate-400">{item.reason}</p>
              </div>
              {(() => {
                const label = signalLabel(item);
                return (
                  <span className={`text-xs px-3 py-1 rounded-full ${label.className}`}>
                    {label.text}
                  </span>
                );
              })()}
            </div>
            {item.speculative ? (
              <div className="mt-2 inline-flex items-center gap-2 text-xs px-2 py-1 rounded-full border border-rose-500/40 bg-rose-500/10 text-rose-200">
                ⚠️ SPÉCULATIF
              </div>
            ) : null}
            {(() => {
              const winRateRaw = Number(item.win_rate ?? 0);
              const winRate = winRateRaw <= 1 ? winRateRaw * 100 : winRateRaw;
              return winRate && winRate < 50 ? (
                <div className="mt-2 inline-flex items-center gap-2 text-xs px-2 py-1 rounded-full border border-amber-500/40 bg-amber-500/10 text-amber-200">
                  ⚠️ STATISTIQUEMENT FAIBLE
                </div>
              ) : null;
            })()}
            <div className="mt-3 grid grid-cols-2 gap-2 text-[11px] text-slate-300">
              <div>Prix: {item.price ?? '—'}</div>
              <div>Volume Z: {item.volume_z ?? '—'}</div>
              <div>Sharpe: {item.sharpe ?? '—'}</div>
              <div>Win rate: {item.win_rate ?? '—'}</div>
            </div>
            <div className="absolute z-20 hidden group-hover/action:block left-4 top-full mt-3 w-[22rem] bg-slate-950 border border-slate-800 rounded-xl p-4 text-sm text-slate-200 shadow-xl">
              <p className="text-slate-100 font-semibold text-base">{item.ticker} · Détails IA</p>
              <p className="text-slate-300 mt-1">{item.reason}</p>
              <div className="mt-3 space-y-2">
                {item.advice?.length ? (
                  <div>
                    <p className="text-[11px] uppercase tracking-widest text-slate-500">Conseils</p>
                    <ul className="mt-1 list-disc list-inside text-sm text-amber-200">
                      {item.advice.map((note) => (
                        <li key={note}>{note}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                {item.alerts?.length ? (
                  <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-2 text-amber-200 text-xs">
                    {item.alerts.map((note) => (
                      <p key={`${item.ticker}-${note}`}>{note}</p>
                    ))}
                  </div>
                ) : null}
                <div>
                  <p className="text-[11px] uppercase tracking-widest text-slate-500">Pourquoi</p>
                  <ul className="mt-1 list-disc list-inside text-sm">
                    {item.drivers?.map((note) => (
                      <li key={note}>{note}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <p className="text-[11px] uppercase tracking-widest text-slate-500">Signaux clés</p>
                  <ul className="mt-1 space-y-1">
                    {item.metrics?.map((metric) => (
                      <li key={`${item.ticker}-${metric.label}`}>
                        <span className="text-slate-400">{metric.label}:</span> {metric.value}
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <p className="text-[11px] uppercase tracking-widest text-slate-500">Risques</p>
                  <ul className="mt-1 list-disc list-inside space-y-1 text-rose-300">
                    {item.risks?.map((risk) => (
                      <li key={`${item.ticker}-${risk}`}>{risk}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
            <div className="mt-4">
              <div className="flex justify-between text-xs text-slate-400">
                <span>Confiance IA</span>
                <span className={scoreColor(item.ai_score ?? item.confidence)}>
                  {normalizeScore(item.ai_score ?? item.confidence).toFixed(0)}%
                </span>
              </div>
              <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden mt-2">
                <div
                  className={`h-full ${scoreColor(item.ai_score ?? item.confidence).replace('text-', 'bg-')}`}
                  style={{ width: `${normalizeScore(item.ai_score ?? item.confidence)}%` }}
                ></div>
              </div>
            </div>
          </motion.div>
        ))}
        <button
          className="w-full py-3 bg-indigo-600 text-white rounded-xl disabled:opacity-60"
          type="button"
          onClick={loadOptimizer}
          disabled={isRefreshing}
        >
          {isRefreshing ? 'Actualisation…' : 'Rebalance'}
        </button>
      </div>
      <AIRecommendations
        items={suggestions}
        title="Ajouts recommandés"
        emptyMessage="Aucune suggestion pour le moment."
        onRefresh={loadOptimizer}
        isRefreshing={isRefreshing}
      />
    </div>
  );
}

export default OptimizerPage;
