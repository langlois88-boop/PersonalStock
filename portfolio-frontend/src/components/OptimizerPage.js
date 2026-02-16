import { motion } from 'framer-motion';
import { useEffect, useMemo, useState } from 'react';
import AIRecommendations from './AIRecommendations';
import api from '../api/api';

function OptimizerPage() {
  const [actions, setActions] = useState([]);
  const [suggestions, setSuggestions] = useState([]);

  useEffect(() => {
    let isMounted = true;

    const loadOptimizer = async () => {
      try {
        const res = await api.get('optimizer/');
        if (!isMounted) return;
        const payload = res.data || {};
        setActions(Array.isArray(payload.actions) ? payload.actions : []);
        setSuggestions(Array.isArray(payload.suggestions) ? payload.suggestions : []);
      } catch (err) {
        if (!isMounted) return;
        setActions([]);
        setSuggestions([]);
      }
    };

    loadOptimizer();

    return () => {
      isMounted = false;
    };
  }, []);

  const hasActions = useMemo(() => actions.length > 0, [actions.length]);

  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
      <div className="xl:col-span-2 space-y-4">
        {!hasActions ? (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 text-slate-400">
            Aucune position détectée. Ajoute des transactions pour obtenir des recommandations.
          </div>
        ) : null}
        {actions.map((item, idx) => (
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
              <span
                className={`text-xs px-3 py-1 rounded-full ${
                  item.signal === 'SELL'
                    ? 'bg-rose-500/10 text-rose-400 border border-rose-500/30'
                    : item.signal === 'BUY MORE'
                      ? 'bg-indigo-500/10 text-indigo-300 border border-indigo-500/30'
                      : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                }`}
              >
                {item.signal}
              </span>
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
                <span>{item.confidence}%</span>
              </div>
              <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden mt-2">
                <div
                  className="h-full bg-indigo-500"
                  style={{ width: `${item.confidence}%` }}
                ></div>
              </div>
            </div>
          </motion.div>
        ))}
        <button className="w-full py-3 bg-indigo-600 text-white rounded-xl">Rebalance</button>
      </div>
      <AIRecommendations
        items={suggestions}
        title="Ajouts recommandés"
        emptyMessage="Aucune suggestion pour le moment."
      />
    </div>
  );
}

export default OptimizerPage;
