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
    const drivers = [];
    if (pnlPct >= 3) drivers.push('Momentum positif sur la position.');
    if (pnlPct <= -7) drivers.push('Perte au-delà du seuil de tolérance.');
    if (pnlPct > -7 && pnlPct < 3) drivers.push('Performance proche du coût moyen.');
    drivers.push(holding.category === 'Stable' ? 'Profil défensif/stable.' : 'Profil plus volatil.');

    const volumeZ = Number(holding.volume_z);
    const hasVolumeZ = Number.isFinite(volumeZ);
    const volumeZMin = 0.5;
    if (hasVolumeZ) {
      drivers.push(volumeZ >= volumeZMin ? 'Volume Z-Score valide.' : 'Volume Z-Score sous le seuil.');
    }

    const relStrength = holding.relative_strength || {};
    const rsOutperform = relStrength.outperform;
    if (rsOutperform === true) {
      drivers.push('Surperforme le secteur (30j).');
    } else if (rsOutperform === false) {
      drivers.push('Sous-performe le secteur (30j).');
    }

    const metrics = [
      { label: 'Prix', value: `$${Number(holding.price || 0).toFixed(2)}` },
      { label: 'Coût moyen', value: `$${Number(holding.avg_cost || 0).toFixed(2)}` },
      { label: 'P/L', value: `${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)} (${pnlPct.toFixed(2)}%)` },
    ];
    if (hasVolumeZ) {
      metrics.push({
        label: 'Volume Z',
        value: `${volumeZ.toFixed(2)} (cible ≥ ${volumeZMin})`,
      });
    }
    if (relStrength?.stock_return_30d !== null && relStrength?.stock_return_30d !== undefined) {
      metrics.push({
        label: 'Perf 30j',
        value: `${Number(relStrength.stock_return_30d).toFixed(2)}%`,
      });
    }
    if (relStrength?.sector_median_30d !== null && relStrength?.sector_median_30d !== undefined) {
      metrics.push({
        label: 'Secteur 30j',
        value: `${Number(relStrength.sector_median_30d).toFixed(2)}%`,
      });
    }

    const risks = [];
    if (holding.category !== 'Stable' || Math.abs(pnlPct) >= 10) {
      risks.push('Volatilité élevée');
    }
    risks.push('Risque macro global');
    if (holding.earnings_blacklisted) {
      risks.push(`Earnings cette semaine (${holding.earnings_date || 'date inconnue'})`);
    }
    if (hasVolumeZ && volumeZ < volumeZMin) {
      risks.push('Volume insuffisant pour confirmer le signal');
    }

    const advice = [];
    if (signal === 'BUY MORE') {
      advice.push('Acheter au maximum 50% de la position initiale.');
      advice.push('Déplacer le Trailing Stop-Loss de toute la position à 10.50$.');
    }

    return {
      ticker: holding.ticker,
      signal,
      confidence,
      reason,
      drivers,
      metrics,
      risks,
      advice,
      details: {
        fundamentals,
        fundamentalsSource: 'Source: profil du stock.',
        rsi: `Prix actuel ${Number(holding.price || 0).toFixed(2)}.`,
      },
      name: holding.name || holding.ticker,
      type: holding.category === 'Stable' ? 'Bluechip' : 'Watchlist',
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
