import { useEffect, useMemo, useState } from 'react';
import { Area, AreaChart, CartesianGrid, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { motion } from 'framer-motion';
import api from '../api/api';

const emptyState = {
  total_balance: 0,
  change_24h: 0,
  change_24h_pct: 0,
  change_7d: 0,
  change_7d_pct: 0,
  allocation: { stable_pct: 0, risky_pct: 0 },
  holdings: [],
  chart: [],
};

const StatCard = ({ title, value, subtitle, accent }) => (
  <motion.div
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.4 }}
    className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 backdrop-blur-md"
  >
    <p className="text-xs uppercase tracking-[0.2em] text-slate-500">{title}</p>
    <p className={`text-2xl font-semibold ${accent}`}>{value}</p>
    <p className="text-xs text-slate-400">{subtitle}</p>
  </motion.div>
);

function GlobalPortfolio() {
  const [data, setData] = useState(emptyState);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    setError('');
    api
      .get('dashboard/portfolio/')
      .then((res) => setData(res.data))
      .catch(() => {
        setError("Impossible de charger le portefeuille.");
        setData(emptyState);
      })
      .finally(() => setLoading(false));
  }, []);

  const gaugeData = useMemo(
    () => [
      { name: 'Stable', value: data.allocation?.stable_pct || 0, fill: '#6366f1' },
      { name: 'Risky', value: data.allocation?.risky_pct || 0, fill: '#f43f5e' },
    ],
    [data.allocation]
  );

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        <StatCard
          title="Solde Total"
          value={`$${data.total_balance?.toLocaleString('fr-CA')}`}
          subtitle="Valeur actuelle en CAD"
          accent="text-white"
        />
        <StatCard
          title="Changement 24h"
          value={`${data.change_24h >= 0 ? '+' : ''}${data.change_24h}`}
          subtitle={`${data.change_24h_pct}% aujourd'hui"`}
          accent={data.change_24h >= 0 ? 'text-emerald-400' : 'text-rose-400'}
        />
        <StatCard
          title="Performance Hebdo"
          value={`${data.change_7d >= 0 ? '+' : ''}${data.change_7d}`}
          subtitle={`${data.change_7d_pct}% sur 7 jours"`}
          accent={data.change_7d >= 0 ? 'text-emerald-400' : 'text-rose-400'}
        />
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 backdrop-blur-md flex flex-col gap-2">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Allocation 60/40</p>
          <div className="flex items-center gap-4">
            <div className="h-20 w-20">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={gaugeData} dataKey="value" innerRadius={28} outerRadius={36} stroke="none" />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div>
              <p className="text-white text-lg font-semibold">{data.allocation?.stable_pct || 0}% Stable</p>
              <p className="text-xs text-slate-400">{data.allocation?.risky_pct || 0}% Risky</p>
            </div>
          </div>
        </div>
      </div>

      {error && <p className="text-sm text-rose-400">{error}</p>}

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2 bg-slate-900/80 border border-slate-800 rounded-2xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Performance 1 an</h3>
          <div className="h-72">
            {loading ? (
              <p className="text-sm text-slate-400">Chargement…</p>
            ) : data.chart && data.chart.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data.chart}>
                  <defs>
                    <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#6366f1" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="date" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1f2937' }} />
                  <Area type="monotone" dataKey="value" stroke="#6366f1" fillOpacity={1} fill="url(#colorValue)" />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-sm text-slate-400">Aucune donnée de performance.</p>
            )}
          </div>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Positions clés</h3>
          <div className="space-y-3">
            {loading ? (
              <p className="text-sm text-slate-400">Chargement…</p>
            ) : data.holdings && data.holdings.length > 0 ? (
              data.holdings.map((row) => (
                <div key={row.ticker} className="flex items-center justify-between bg-slate-950/40 p-3 rounded-xl">
                  <div>
                    <p className="text-white font-semibold">
                      {row.ticker}{' '}
                      <span
                        className={`text-[10px] px-2 py-0.5 rounded-full ${
                          row.category === 'Stable'
                            ? 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/30'
                            : 'bg-rose-500/10 text-rose-300 border border-rose-500/30'
                        }`}
                      >
                        {row.category}
                      </span>
                    </p>
                    <p className="text-xs text-slate-400">{row.name}</p>
                    <p className="text-xs text-slate-500">{Number(row.shares || 0).toFixed(2)} shares</p>
                  </div>
                  <div className="text-right">
                    <p className="text-white">${Number(row.price || 0).toFixed(2)}</p>
                    <p className="text-xs text-slate-400">Value: ${Number(row.value || 0).toFixed(2)}</p>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-sm text-slate-400">Aucune position enregistrée.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default GlobalPortfolio;
