import { useEffect, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import api from '../api/api';

const features = [
  { name: 'Volume', value: 40 },
  { name: 'Macro', value: 30 },
  { name: 'Sentiment', value: 20 },
  { name: 'RSI', value: 10 },
];

const logs = [
  'Backtester connected to Data Fusion Engine.',
  'Macro refresh completed.',
  'Scout nightly run finished.',
];

const formatPct = (value, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return `${Number(value).toFixed(digits)}%`;
};

const formatNumber = (value, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toFixed(digits);
};

const formatMoney = (value) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(Number(value));
};

function AnalyticsLabPage() {
  const [backtest, setBacktest] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;

    const loadBacktest = async () => {
      try {
        const res = await api.get('backtester/', { params: { symbol: 'SPY', days: 365 } });
        if (!isMounted) return;
        setBacktest(res.data);
      } catch (err) {
        if (!isMounted) return;
        setBacktest(null);
      } finally {
        if (isMounted) setLoading(false);
      }
    };

    loadBacktest();
    return () => {
      isMounted = false;
    };
  }, []);

  const chartData = useMemo(() => {
    if (!backtest?.dates?.length) return [];
    const base = backtest.dates.map((date, index) => ({
      date,
      buyHold: backtest.buy_hold_curve?.[index] ?? null,
      strategy: backtest.equity_curve?.[index] ?? null,
      projection: null,
    }));

    const curve = backtest.equity_curve || [];
    if (curve.length < 2) return base;

    const years = Math.max(1, curve.length / 252);
    const lastValue = curve[curve.length - 1];
    const cagr = lastValue > 0 ? Math.pow(lastValue, 1 / years) - 1 : 0;

    const projectionPoints = Array.from({ length: 60 }, (_, i) => {
      const month = i + 1;
      return {
        date: `+${month}m`,
        buyHold: null,
        strategy: null,
        projection: lastValue * Math.pow(1 + cagr, month / 12),
      };
    });

    return [...base, ...projectionPoints];
  }, [backtest]);

  const reportMetrics = [
    { label: 'Win Rate', value: formatPct(backtest?.win_rate, 2) },
    { label: 'Total Return', value: formatPct(backtest?.total_return_pct, 2) },
    { label: 'Sharpe Ratio', value: formatNumber(backtest?.sharpe_ratio, 3) },
    { label: 'Max Drawdown', value: formatPct((backtest?.max_drawdown ?? 0) * 100, 2) },
    { label: 'Final Balance', value: formatMoney(backtest?.final_balance) },
  ];

  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
      <div className="xl:col-span-2 space-y-6">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div className="flex items-center justify-between mb-4">
            <p className="text-white font-semibold">Backtest Comparator</p>
            <span className="text-xs text-slate-400">{backtest?.symbol || 'SPY'} · 12m</span>
          </div>
          <div className="h-72">
            {loading ? (
              <p className="text-slate-400 text-sm">Loading backtest…</p>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 10, right: 24, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="date" hide />
                  <YAxis stroke="#64748b" />
                  <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1f2937' }} />
                  <Legend />
                  <Line type="monotone" dataKey="buyHold" name="Buy & Hold" stroke="#38bdf8" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="strategy" name="Strategy" stroke="#6366f1" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="projection" name="Projection 5y" stroke="#22c55e" strokeWidth={2} strokeDasharray="6 6" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {reportMetrics.map((metric) => (
            <div key={metric.label} className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
              <p className="text-xs text-slate-400 uppercase tracking-[0.2em]">{metric.label}</p>
              <p className="text-2xl text-white font-semibold">{metric.value}</p>
            </div>
          ))}
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 h-72">
          <div className="mb-4">
            <p className="text-white font-semibold">Feature Importance</p>
            <p className="text-xs text-slate-400 mt-1">
              Hiérarchie de décision: Volume → Macro → Sentiment → RSI. L’IA confirme le flux avant le timing.
            </p>
          </div>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={features} layout="vertical">
              <XAxis type="number" stroke="#64748b" />
              <YAxis dataKey="name" type="category" stroke="#64748b" />
              <Bar dataKey="value" fill="#6366f1" radius={[8, 8, 8, 8]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
        <p className="text-white font-semibold mb-4">Model Logs</p>
        <div className="space-y-3 text-xs text-slate-400 font-mono">
          {logs.map((log) => (
            <div key={log} className="bg-slate-950/60 p-3 rounded-lg">{log}</div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default AnalyticsLabPage;
