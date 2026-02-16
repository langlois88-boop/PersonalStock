import { useEffect, useMemo, useState } from 'react';
import { Area, AreaChart, CartesianGrid, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { motion } from 'framer-motion';
import api from '../api/api';
import UnifiedAlerts from './UnifiedAlerts';

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
  const [livePrices, setLivePrices] = useState({});
  const [accountData, setAccountData] = useState({ accounts: [], top_movers: {} });
  const [accountLoading, setAccountLoading] = useState(true);
  const [selectedAccountId, setSelectedAccountId] = useState('ALL');
  const [sortConfig, setSortConfig] = useState({ key: 'ticker', direction: 'asc' });
  const [news, setNews] = useState({ holdings: [], sectors_news: [], sentiment: { positive: [], negative: [] } });
  const [newsLoading, setNewsLoading] = useState(false);
  const [newsSymbol, setNewsSymbol] = useState('ALL');
  const [newsVisible, setNewsVisible] = useState({ holdings: 6, sectors: 6, positive: 6, negative: 6 });

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

  useEffect(() => {
    setAccountLoading(true);
    const params = selectedAccountId !== 'ALL' ? { account_id: selectedAccountId } : {};
    api
      .get('dashboard/accounts/', { params })
      .then((res) => setAccountData(res.data || { accounts: [], top_movers: {} }))
      .catch(() => setAccountData({ accounts: [], top_movers: {} }))
      .finally(() => setAccountLoading(false));
  }, [selectedAccountId]);

  useEffect(() => {
    let socket;
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${protocol}://${window.location.hostname}:8001/ws/updates/`;

    try {
      socket = new WebSocket(wsUrl);
      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (payload?.prices) {
            setLivePrices(payload.prices);
          }
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

  useEffect(() => {
    setNewsLoading(true);
    const params = newsSymbol !== 'ALL'
      ? { symbol: newsSymbol, limit: 12, sector_limit: 12, sentiment_limit: 12 }
      : { limit: 12, sector_limit: 12, sentiment_limit: 12 };
    api
      .get('dashboard/news/', { params })
      .then((res) => setNews(res.data || { holdings: [], sectors_news: [], sentiment: { positive: [], negative: [] } }))
      .catch(() => setNews({ holdings: [], sectors_news: [], sentiment: { positive: [], negative: [] } }))
      .finally(() => setNewsLoading(false));
  }, [newsSymbol, data.holdings?.length]);

  useEffect(() => {
    setNewsVisible({ holdings: 6, sectors: 6, positive: 6, negative: 6 });
  }, [newsSymbol]);

  const gaugeData = useMemo(
    () => [
      { name: 'Stable', value: data.allocation?.stable_pct || 0, fill: '#6366f1' },
      { name: 'Risky', value: data.allocation?.risky_pct || 0, fill: '#f43f5e' },
    ],
    [data.allocation]
  );

  const formatMoney = (value) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
    return `$${Number(value).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  const formatPct = (value) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
    return `${Number(value).toFixed(2)}%`;
  };

  const formatDate = (value) => {
    if (!value) return '—';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return '—';
    return date.toLocaleDateString('fr-CA', { month: 'short', day: 'numeric' });
  };

  const sentimentBadge = (value) => {
    if (value === null || value === undefined) return 'text-slate-400';
    if (value >= 0.35) return 'text-emerald-300';
    if (value <= -0.35) return 'text-rose-300';
    return 'text-slate-400';
  };

  const toggleSort = (key) => {
    setSortConfig((prev) => {
      if (prev.key === key) {
        return { key, direction: prev.direction === 'asc' ? 'desc' : 'asc' };
      }
      return { key, direction: 'asc' };
    });
  };

  const sortIndicator = (key) => (sortConfig.key === key ? (sortConfig.direction === 'asc' ? '▲' : '▼') : '');

  const getSortValue = (pos, key) => {
    if (key === 'ticker') return (pos.ticker || '').toUpperCase();
    return Number(pos?.[key] ?? 0);
  };

  const sortPositions = (positions) => {
    const list = Array.isArray(positions) ? [...positions] : [];
    list.sort((a, b) => {
      const aVal = getSortValue(a, sortConfig.key);
      const bVal = getSortValue(b, sortConfig.key);
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortConfig.direction === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }
      return sortConfig.direction === 'asc' ? aVal - bVal : bVal - aVal;
    });
    return list;
  };

  const currentValueClass = (pos) => {
    const cost = Number(pos.cost_value || 0);
    const current = Number(pos.current_value || 0);
    if (!cost) return 'text-slate-300';
    const ratio = (current - cost) / cost;
    if (ratio >= 0.08) return 'text-emerald-300';
    if (ratio >= 0.02) return 'text-emerald-400';
    if (ratio <= -0.08) return 'text-rose-300';
    if (ratio <= -0.02) return 'text-rose-400';
    return 'text-slate-300';
  };

  const renderNewsItem = (item) => (
    <a
      key={`${item.url}-${item.ticker}`}
      href={item.url}
      target="_blank"
      rel="noreferrer"
      className="block rounded-xl border border-slate-800 bg-slate-950/60 p-3 hover:border-indigo-500/40 transition"
    >
      <div className="flex items-center justify-between gap-2">
        <p className="text-sm text-white font-semibold truncate">{item.headline}</p>
        <span className={`text-xs ${sentimentBadge(item.sentiment)}`}>
          {item.sentiment === null || item.sentiment === undefined ? '—' : item.sentiment.toFixed(2)}
        </span>
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-slate-400">
        <span className="text-slate-300">{item.ticker}</span>
        {item.sector ? <span>· {item.sector}</span> : null}
        {item.source ? <span>· {item.source}</span> : null}
        <span>· {formatDate(item.published_at)}</span>
      </div>
    </a>
  );

  const confidence = data.confidence_meter || {};
  const confidenceStatus = confidence.status || 'unavailable';
  const confidenceStyles = {
    green: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-200',
    orange: 'border-amber-500/40 bg-amber-500/10 text-amber-200',
    red: 'border-rose-500/40 bg-rose-500/10 text-rose-200',
    neutral: 'border-slate-700 bg-slate-950/60 text-slate-300',
    unavailable: 'border-slate-800 bg-slate-950/60 text-slate-400',
  };
  const confidenceLabel = confidence.label || 'Indisponible';
  const confidenceSymbol = confidence.symbol || '—';
  const confidenceAiScore = confidence.ai_score ?? null;
  const confidenceVolumeZ = confidence.volume_z ?? null;
  const confidenceVolRegime = confidence.vol_regime ?? null;

  const exportAccountsCsv = () => {
    const rows = [];
    const header = [
      'account_name',
      'account_type',
      'ticker',
      'avg_cost',
      'shares',
      'cost_value',
      'current_price',
      'current_value',
      'weekly_return_pct',
      'monthly_return_pct',
      'annual_return_pct',
    ];
    rows.push(header);

    (accountData.accounts || []).forEach((account) => {
      (account.positions || []).forEach((pos) => {
        rows.push([
          account.account_name,
          account.account_type,
          pos.ticker,
          pos.avg_cost,
          pos.shares,
          pos.cost_value,
          pos.current_price,
          pos.current_value,
          pos.weekly_return_pct,
          pos.monthly_return_pct,
          pos.annual_return_pct,
        ]);
      });
    });

    if (rows.length <= 1) return;
    const csv = rows.map((row) => row.map((cell) => {
      const value = cell === null || cell === undefined ? '' : String(cell);
      return value.includes(',') || value.includes('"') || value.includes('\n')
        ? `"${value.replace(/"/g, '""')}"`
        : value;
    }).join(',')).join('\n');

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.setAttribute('download', 'account_positions.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      <UnifiedAlerts />
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

      <div className={`border rounded-2xl p-4 backdrop-blur-md ${confidenceStyles[confidenceStatus] || confidenceStyles.unavailable}`}>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <p className="text-xs uppercase tracking-[0.2em]">Confidence Meter</p>
            <p className="text-lg font-semibold">{confidenceLabel}</p>
            <p className="text-xs opacity-80">Symbole: {confidenceSymbol}</p>
          </div>
          <div className="text-right text-xs space-y-1">
            <p>IA Score: {confidenceAiScore === null ? '—' : `${confidenceAiScore.toFixed(2)}%`}</p>
            <p>Volume Z: {confidenceVolumeZ === null ? '—' : confidenceVolumeZ.toFixed(2)}</p>
            <p>Vol Regime: {confidenceVolRegime === null ? '—' : confidenceVolRegime.toFixed(2)}</p>
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
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">Positions clés</h3>
            <span className="text-xs text-emerald-300">Live</span>
          </div>
          <div className="space-y-3">
            {loading ? (
              <p className="text-sm text-slate-400">Chargement…</p>
            ) : data.holdings && data.holdings.length > 0 ? (
              data.holdings.map((row) => {
                const livePrice = livePrices?.[row.ticker];
                const price = Number(livePrice ?? row.price ?? 0);
                const shares = Number(row.shares || 0);
                const value = price * shares;
                const costValue = Number(row.cost_value || 0);
                const unrealized = value - costValue;
                const unrealizedPct = costValue ? (unrealized / costValue) * 100 : 0;
                return (
                  <div key={row.ticker} className="flex items-center justify-between bg-slate-950/40 p-3 rounded-xl">
                    <div>
                      <p className="text-white font-semibold">
                        <a
                          href={`https://finance.yahoo.com/quote/${row.ticker}`}
                          target="_blank"
                          rel="noreferrer"
                          className="hover:text-indigo-300"
                        >
                          {row.ticker}
                        </a>{' '}
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
                      <p className="text-xs text-slate-500">{shares.toFixed(2)} shares</p>
                      <p className="text-xs text-slate-500">
                        Cost: ${Number(row.cost_value || 0).toFixed(2)} @ ${Number(row.avg_cost || 0).toFixed(2)}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-white">${price.toFixed(2)}</p>
                      <p className="text-xs text-slate-400">Value: ${value.toFixed(2)}</p>
                      <p className={`text-xs ${unrealized >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>
                        P/L: {unrealized >= 0 ? '+' : ''}{unrealized.toFixed(2)} ({unrealizedPct.toFixed(2)}%)
                      </p>
                    </div>
                  </div>
                );
              })
            ) : (
              <p className="text-sm text-slate-400">Aucune position enregistrée.</p>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2 bg-slate-900/80 border border-slate-800 rounded-2xl p-6">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
            <h3 className="text-lg font-semibold text-white">Comptes & positions</h3>
            <div className="flex items-center gap-3">
              <span className="text-xs text-slate-400">Prix d'achat · Rendements</span>
              <select
                value={selectedAccountId}
                onChange={(event) => setSelectedAccountId(event.target.value)}
                className="bg-slate-950/60 border border-slate-800 text-slate-200 text-xs rounded-lg px-2 py-1"
              >
                <option value="ALL">Tous les comptes</option>
                {(accountData.accounts || []).map((account) => (
                  <option key={account.account_id} value={account.account_id}>
                    {account.account_name} · {account.account_type}
                  </option>
                ))}
              </select>
              <button
                type="button"
                onClick={exportAccountsCsv}
                className="text-xs text-indigo-300 hover:text-indigo-200"
                disabled={!accountData.accounts?.length}
              >
                Export CSV
              </button>
            </div>
          </div>
          {accountLoading ? (
            <p className="text-sm text-slate-400">Chargement…</p>
          ) : accountData.accounts?.length ? (
            <div className="space-y-4">
              {accountData.accounts.map((account) => (
                <div key={account.account_id} className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <p className="text-white font-semibold">
                        {account.account_name} · {account.account_type}
                      </p>
                      <p className="text-xs text-slate-400">Valeur actuelle {formatMoney(account.total_value)}</p>
                    </div>
                    <div className="text-xs text-slate-400">Coût total {formatMoney(account.total_cost)}</div>
                  </div>
                  <div className="mt-3 overflow-x-auto">
                    <table className="w-full text-xs text-slate-200">
                      <thead>
                        <tr className="text-slate-400">
                          <th className="text-left py-2">
                            <button type="button" onClick={() => toggleSort('ticker')} className="text-left">
                              Ticker {sortIndicator('ticker')}
                            </button>
                          </th>
                          <th className="text-right py-2">
                            <button type="button" onClick={() => toggleSort('avg_cost')} className="text-right">
                              Prix achat {sortIndicator('avg_cost')}
                            </button>
                          </th>
                          <th className="text-right py-2">
                            <button type="button" onClick={() => toggleSort('shares')} className="text-right">
                              Quantité {sortIndicator('shares')}
                            </button>
                          </th>
                          <th className="text-right py-2">
                            <button type="button" onClick={() => toggleSort('cost_value')} className="text-right">
                              Valeur achat {sortIndicator('cost_value')}
                            </button>
                          </th>
                          <th className="text-right py-2">
                            <button type="button" onClick={() => toggleSort('current_price')} className="text-right">
                              Prix actuel {sortIndicator('current_price')}
                            </button>
                          </th>
                          <th className="text-right py-2">
                            <button type="button" onClick={() => toggleSort('current_value')} className="text-right">
                              Valeur actuelle {sortIndicator('current_value')}
                            </button>
                          </th>
                          <th className="text-right py-2">
                            <button type="button" onClick={() => toggleSort('unrealized_pnl_pct')} className="text-right">
                              P/L % {sortIndicator('unrealized_pnl_pct')}
                            </button>
                          </th>
                          <th className="text-right py-2">
                            <button type="button" onClick={() => toggleSort('weekly_return_pct')} className="text-right">
                              7j {sortIndicator('weekly_return_pct')}
                            </button>
                          </th>
                          <th className="text-right py-2">
                            <button type="button" onClick={() => toggleSort('monthly_return_pct')} className="text-right">
                              30j {sortIndicator('monthly_return_pct')}
                            </button>
                          </th>
                          <th className="text-right py-2">
                            <button type="button" onClick={() => toggleSort('annual_return_pct')} className="text-right">
                              1a {sortIndicator('annual_return_pct')}
                            </button>
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {sortPositions(account.positions || []).map((pos) => (
                          <tr key={`${account.account_id}-${pos.ticker}`} className="border-t border-slate-800">
                            <td className="py-2 font-semibold text-white">
                              <a
                                href={`https://finance.yahoo.com/quote/${pos.ticker}`}
                                target="_blank"
                                rel="noreferrer"
                                className="hover:text-indigo-300"
                              >
                                {pos.ticker}
                              </a>
                            </td>
                            <td className="py-2 text-right">{formatMoney(pos.avg_cost)}</td>
                            <td className="py-2 text-right">{Number(pos.shares || 0).toFixed(2)}</td>
                            <td className="py-2 text-right">{formatMoney(pos.cost_value)}</td>
                            <td className="py-2 text-right">{formatMoney(pos.current_price)}</td>
                            <td className={`py-2 text-right ${currentValueClass(pos)}`}>{formatMoney(pos.current_value)}</td>
                            <td className={`py-2 text-right ${Number(pos.unrealized_pnl_pct || 0) >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>
                              {formatPct(pos.unrealized_pnl_pct)}
                            </td>
                            <td className={`py-2 text-right ${Number(pos.weekly_return_pct || 0) >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>
                              {formatPct(pos.weekly_return_pct)}
                            </td>
                            <td className={`py-2 text-right ${Number(pos.monthly_return_pct || 0) >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>
                              {formatPct(pos.monthly_return_pct)}
                            </td>
                            <td className={`py-2 text-right ${Number(pos.annual_return_pct || 0) >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>
                              {formatPct(pos.annual_return_pct)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-slate-400">Aucun compte trouvé.</p>
          )}
        </div>
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Top movers du jour</h3>
          {accountLoading ? (
            <p className="text-sm text-slate-400">Chargement…</p>
          ) : (
            <div className="space-y-4">
              {accountData.top_movers?.gainer ? (
                <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-3">
                  <p className="text-emerald-200 text-sm font-semibold">Top gainer</p>
                  <p className="text-white font-semibold">{accountData.top_movers.gainer.ticker}</p>
                  <p className="text-xs text-emerald-200">{formatPct(accountData.top_movers.gainer.day_change_pct)}</p>
                </div>
              ) : (
                <p className="text-sm text-slate-400">Aucun gainer disponible.</p>
              )}
              {accountData.top_movers?.loser ? (
                <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-3">
                  <p className="text-rose-200 text-sm font-semibold">Top loser</p>
                  <p className="text-white font-semibold">{accountData.top_movers.loser.ticker}</p>
                  <p className="text-xs text-rose-200">{formatPct(accountData.top_movers.loser.day_change_pct)}</p>
                </div>
              ) : (
                <p className="text-sm text-slate-400">Aucun loser disponible.</p>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h3 className="text-lg font-semibold text-white">Actualités liées au portefeuille</h3>
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Filtre</span>
            <select
              value={newsSymbol}
              onChange={(event) => setNewsSymbol(event.target.value)}
              className="bg-slate-950/60 border border-slate-800 text-slate-200 text-xs rounded-lg px-2 py-1"
            >
              <option value="ALL">Tous les titres</option>
              {(data.holdings || []).map((row) => (
                <option key={row.ticker} value={row.ticker}>
                  {row.ticker}
                </option>
              ))}
            </select>
          </div>
        </div>

        {newsLoading ? (
          <p className="text-sm text-slate-400">Chargement…</p>
        ) : (
          <div className="space-y-6">
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-slate-300 mb-2">News sur mes actions</p>
                <div className="space-y-3">
                  {(news.holdings || []).length ? (
                    news.holdings.slice(0, newsVisible.holdings).map(renderNewsItem)
                  ) : (
                    <p className="text-xs text-slate-500">Aucune news disponible.</p>
                  )}
                </div>
                {(news.holdings || []).length > newsVisible.holdings ? (
                  <button
                    type="button"
                    onClick={() => setNewsVisible((prev) => ({ ...prev, holdings: prev.holdings + 6 }))}
                    className="mt-3 text-xs text-indigo-300 hover:text-indigo-200"
                  >
                    Afficher plus →
                  </button>
                ) : null}
              </div>
              <div>
                <p className="text-sm text-slate-300 mb-2">News sur les secteurs</p>
                <div className="space-y-3">
                  {(news.sectors_news || []).length ? (
                    news.sectors_news.slice(0, newsVisible.sectors).map(renderNewsItem)
                  ) : (
                    <p className="text-xs text-slate-500">Aucune news secteur disponible.</p>
                  )}
                </div>
                {(news.sectors_news || []).length > newsVisible.sectors ? (
                  <button
                    type="button"
                    onClick={() => setNewsVisible((prev) => ({ ...prev, sectors: prev.sectors + 6 }))}
                    className="mt-3 text-xs text-indigo-300 hover:text-indigo-200"
                  >
                    Afficher plus →
                  </button>
                ) : null}
              </div>
            </div>
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-emerald-300 mb-2">Sentiment positif fort</p>
                <div className="space-y-3">
                  {(news.sentiment?.positive || []).length ? (
                    news.sentiment.positive.slice(0, newsVisible.positive).map(renderNewsItem)
                  ) : (
                    <p className="text-xs text-slate-500">Aucune news positive forte.</p>
                  )}
                </div>
                {(news.sentiment?.positive || []).length > newsVisible.positive ? (
                  <button
                    type="button"
                    onClick={() => setNewsVisible((prev) => ({ ...prev, positive: prev.positive + 6 }))}
                    className="mt-3 text-xs text-emerald-300 hover:text-emerald-200"
                  >
                    Afficher plus →
                  </button>
                ) : null}
              </div>
              <div>
                <p className="text-sm text-rose-300 mb-2">Sentiment négatif fort</p>
                <div className="space-y-3">
                  {(news.sentiment?.negative || []).length ? (
                    news.sentiment.negative.slice(0, newsVisible.negative).map(renderNewsItem)
                  ) : (
                    <p className="text-xs text-slate-500">Aucune news négative forte.</p>
                  )}
                </div>
                {(news.sentiment?.negative || []).length > newsVisible.negative ? (
                  <button
                    type="button"
                    onClick={() => setNewsVisible((prev) => ({ ...prev, negative: prev.negative + 6 }))}
                    className="mt-3 text-xs text-rose-300 hover:text-rose-200"
                  >
                    Afficher plus →
                  </button>
                ) : null}
              </div>
            </div>
          </div>
        )}
      </div>

    </div>
  );
}

export default GlobalPortfolio;
