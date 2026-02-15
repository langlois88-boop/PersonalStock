import { motion } from 'framer-motion';
import { useEffect, useMemo, useState } from 'react';
import AIRecommendations from './AIRecommendations';
import api from '../api/api';

function OptimizerPage() {
  const [actions, setActions] = useState([]);
  const [suggestions, setSuggestions] = useState([]);

  const buildAction = (holding) => {
    const pnlPct = Number(holding.unrealized_pnl_pct || 0);
    const pnl = Number(holding.unrealized_pnl || 0);
    let signal = 'KEEP';
    if (pnlPct <= -7) signal = 'SELL';
    if (pnlPct >= 3) signal = 'BUY MORE';

    const dividendYield = Number(holding.dividend_yield || 0);
    const sector = holding.sector || 'Non défini';
    const dividendPct = dividendYield > 1 ? dividendYield : dividendYield * 100;
    const fundamentals = `Secteur ${sector} · Dividende ${dividendPct.toFixed(2)}%`;

    const confidence = Math.min(95, Math.max(55, Math.round(50 + Math.abs(pnlPct) * 3)));
    const reason = `P/L ${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)} (${pnlPct.toFixed(2)}%).`;

    return {
      ticker: holding.ticker,
      signal,
      confidence,
      reason,
      aiNotes: [
        `Coût moyen ${Number(holding.avg_cost || 0).toFixed(2)}.`,
        `Valeur book ${Number(holding.cost_value || 0).toFixed(2)}.`,
        `Valeur marché ${Number(holding.value || 0).toFixed(2)}.`,
      ],
      details: {
        fundamentals,
        fundamentalsSource: 'Source: profil du stock.',
        rsi: `Prix actuel ${Number(holding.price || 0).toFixed(2)}.`,
      },
      name: holding.name || holding.ticker,
    };
  };

  useEffect(() => {
    let isMounted = true;

    const loadActions = async () => {
      try {
        const res = await api.get('dashboard/portfolio/');
        if (!isMounted) return;
        const items = res.data?.holdings || [];
        setActions(items.map(buildAction));
      } catch (err) {
        if (!isMounted) return;
        setActions([]);
      }
    };

    const loadSuggestions = async () => {
      try {
        const [bluechip, penny] = await Promise.all([
          api.get('bluechip-hunter/', { params: { limit: 6 } }),
          api.get('penny-scout/', { params: { limit: 6 } }),
        ]);

        if (!isMounted) return;
        const existing = new Set((actions || []).map((item) => item.ticker));

        const mapped = [];
        const bluechipItems = Array.isArray(bluechip.data) ? bluechip.data : [];
        bluechipItems.forEach((item) => {
          if (!item?.ticker || existing.has(item.ticker)) return;
          mapped.push({
            ticker: item.ticker,
            name: item.name || item.sector || 'Bluechip',
            signal: 'ADD',
            confidence: Math.round(Number(item.ai_score || 70)),
            reason: `AI score ${Number(item.ai_score || 70).toFixed(0)}.`,
            type: 'Bluechip',
            drivers: ['Qualité défensive', 'Liquidité élevée', 'Rendement stable'],
            metrics: [
              { label: 'Prix', value: `$${Number(item.latest_price || 0).toFixed(2)}` },
              { label: 'AI score', value: Number(item.ai_score || 70).toFixed(0) },
            ],
            risks: ['Risque macro global'],
          });
        });

        const pennyItems = Array.isArray(penny.data) ? penny.data : [];
        pennyItems.forEach((item) => {
          if (!item?.ticker || existing.has(item.ticker)) return;
          mapped.push({
            ticker: item.ticker,
            name: item.sector || 'Penny',
            signal: 'ADD',
            confidence: Math.round(Number(item.ai_score?.score || item.ai_score || 60)),
            reason: 'Momentum spéculatif détecté.',
            type: 'Penny',
            drivers: ['Momentum court terme', 'Volume en hausse', 'Catalyseur news'],
            metrics: [
              { label: 'Prix', value: `$${Number(item.latest_price || 0).toFixed(2)}` },
              { label: 'AI score', value: Number(item.ai_score?.score || item.ai_score || 60).toFixed(0) },
            ],
            risks: ['Volatilité élevée'],
          });
        });

        setSuggestions(mapped.slice(0, 6));
      } catch (err) {
        if (!isMounted) return;
        setSuggestions([]);
      }
    };

    loadActions();
    loadSuggestions();

    return () => {
      isMounted = false;
    };
  }, [actions.length]);

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
            className="bg-slate-900 border border-slate-800 rounded-2xl p-5 relative"
          >
            <div className="flex items-start justify-between">
              <div>
                <div className="relative inline-block group/ticker">
                  <p className="text-white text-lg font-semibold">{item.ticker}</p>
                  <div className="absolute z-20 hidden group-hover/ticker:block left-0 top-full mt-2 w-80 bg-slate-950 border border-slate-800 rounded-xl p-4 text-sm text-slate-200 shadow-xl">
                    <p className="text-slate-100 font-semibold text-base">{item.ticker} · Décision IA</p>
                    <p className="text-slate-300 mt-1">{item.reason}</p>
                    <div className="mt-3 space-y-2">
                      <div>
                        <p className="text-slate-400">Pourquoi</p>
                        <ul className="text-slate-200 text-sm list-disc list-inside">
                          {item.aiNotes.map((note) => (
                            <li key={note}>{note}</li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <p className="text-slate-400">Fondamentaux</p>
                        <p className="text-slate-200">{item.details.fundamentals}</p>
                        <p className="text-slate-500 text-xs">Source: {item.details.fundamentalsSource}</p>
                      </div>
                      <div>
                        <p className="text-slate-400">RSI</p>
                        <p className="text-slate-200">{item.details.rsi}</p>
                        <p className="text-slate-500 text-xs">Seuils: <span className="text-slate-400">&gt;70 suracheté</span> · <span className="text-slate-400">&lt;30 survendu</span></p>
                      </div>
                    </div>
                  </div>
                </div>
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
