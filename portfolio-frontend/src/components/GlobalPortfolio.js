import { useEffect, useMemo, useState } from 'react';
import { Area, AreaChart, CartesianGrid, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { motion } from 'framer-motion';
import api from '../api/api';
import UnifiedAlerts from './UnifiedAlerts';

const emptyState = {
  total_balance: 0,
  total_return: 0,
  total_return_pct: 0,
  change_24h: 0,
  change_24h_pct: 0,
  change_7d: 0,
  change_7d_pct: 0,
  current_drawdown: 0,
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

const CHART_RANGES = [
  { key: '1M', label: '1M', days: 30 },
  { key: '3M', label: '3M', days: 90 },
  { key: '6M', label: '6M', days: 180 },
  { key: '1Y', label: '1Y', days: 365 },
  { key: 'ALL', label: 'MAX', days: null },
];

const CHART_MODES = [
  { key: 'value', label: 'Valeur' },
  { key: 'pct', label: '% période' },
];

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
  const [focusFilter, setFocusFilter] = useState(false);
  const [wave2Amounts, setWave2Amounts] = useState({});
  const [chartRange, setChartRange] = useState('3M');
  const [chartMode, setChartMode] = useState('value');

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
          if (payload?.prices) {
            setLivePrices(payload.prices);
          }
        } catch (err) {
          // ignore parse errors
        }
      };
      socket.onerror = () => {
        try {
          socket.close();
        } catch (err) {
          // ignore socket errors
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


  const formatMoney = (value) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
    return `$${Number(value).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  const formatMoneySigned = (value) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
    const num = Number(value);
    const sign = num > 0 ? '+' : num < 0 ? '-' : '';
    return `${sign}$${Math.abs(num).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  const formatPct = (value) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
    return `${Number(value).toFixed(2)}%`;
  };

  const formatPctSigned = (value) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
    const num = Number(value);
    const sign = num > 0 ? '+' : num < 0 ? '-' : '';
    return `${sign}${Math.abs(num).toFixed(2)}%`;
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

  const gaugeData = useMemo(
    () => [
      { name: 'Stable', value: data.allocation?.stable_pct || 0, fill: '#6366f1' },
      { name: 'Risky', value: data.allocation?.risky_pct || 0, fill: '#f43f5e' },
    ],
    [data.allocation]
  );

  const filteredChart = useMemo(() => {
    const chart = Array.isArray(data.chart) ? data.chart : [];
    if (!chart.length || chartRange === 'ALL') return chart;
    const range = CHART_RANGES.find((item) => item.key === chartRange);
    if (!range?.days) return chart;
    const lastDate = new Date(chart[chart.length - 1]?.date);
    if (Number.isNaN(lastDate.getTime())) return chart;
    const startDate = new Date(lastDate);
    startDate.setDate(startDate.getDate() - range.days);
    return chart.filter((point) => {
      const date = new Date(point.date);
      if (Number.isNaN(date.getTime())) return true;
      return date >= startDate;
    });
  }, [data.chart, chartRange]);

  const performanceSeries = useMemo(() => {
    const chart = Array.isArray(filteredChart) ? filteredChart : [];
    if (!chart.length) return [];
    const baseValue = Number(chart[0]?.value || 0) || 0;
    return chart.map((point) => {
      const value = Number(point.value || 0);
      const pct = baseValue ? ((value - baseValue) / baseValue) * 100 : 0;
      return {
        date: point.date,
        value,
        pct,
      };
    });
  }, [filteredChart]);

  const periodReturn = useMemo(() => {
    if (!performanceSeries.length) return { value: null, pct: null };
    const first = Number(performanceSeries[0]?.value || 0);
    const last = Number(performanceSeries[performanceSeries.length - 1]?.value || 0);
    if (!first && !last) return { value: null, pct: null };
    const diff = last - first;
    const pct = first ? (diff / first) * 100 : 0;
    return { value: diff, pct };
  }, [performanceSeries]);

  const portfolioDividendYield = useMemo(() => {
    const list = Array.isArray(data.holdings) ? data.holdings : [];
    let total = 0;
    let weighted = 0;
    list.forEach((row) => {
      const value = Number(row.value ?? (Number(row.price || 0) * Number(row.shares || 0)));
      if (!Number.isFinite(value) || value <= 0) return;
      const yieldValue = Number(row.dividend_yield || 0);
      total += value;
      weighted += value * (Number.isFinite(yieldValue) ? yieldValue : 0);
    });
    if (!total) return null;
    return weighted / total;
  }, [data.holdings]);

  const PerformanceTooltip = ({ active, payload, label, chartMode: mode }) => {
    if (!active || !payload || !payload.length) return null;
    const point = payload[0]?.payload;
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-950/95 p-3 text-xs text-slate-200">
        <p className="text-slate-400">{formatDate(label)}</p>
        {mode === 'pct' ? (
          <>
            <p className="text-sm font-semibold">{formatPctSigned(point?.pct)}</p>
            <p className="text-slate-400">{formatMoney(point?.value)}</p>
          </>
        ) : (
          <>
            <p className="text-sm font-semibold">{formatMoney(point?.value)}</p>
            <p className="text-slate-400">{formatPctSigned(point?.pct)}</p>
          </>
        )}
      </div>
    );
  };

  const topHoldings = useMemo(() => {
    const list = Array.isArray(data.holdings) ? [...data.holdings] : [];
    const withValue = list.map((row) => {
      const livePrice = livePrices?.[row.ticker];
      const price = Number(livePrice ?? row.price ?? 0);
      const shares = Number(row.shares || 0);
      const value = price * shares;
      return { ...row, _positionValue: value };
    });
    withValue.sort((a, b) => (b._positionValue || 0) - (a._positionValue || 0));
    return withValue.slice(0, 4);
  }, [data.holdings, livePrices]);

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
    const filtered = focusFilter
      ? list.filter((pos) => pos.model_win_rate === null || Number(pos.model_win_rate || 0) >= 45)
      : list;
    filtered.sort((a, b) => {
      const aVal = getSortValue(a, sortConfig.key);
      const bVal = getSortValue(b, sortConfig.key);
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortConfig.direction === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }
      return sortConfig.direction === 'asc' ? aVal - bVal : bVal - aVal;
    });
    return filtered;
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

  const wave2Budget = 2300;
  const wave2Amount = Number((wave2Budget * 0.25).toFixed(2));

  const updateRiskStatus = (rsi, sharpe, tickerType, tickerSymbol) => {
    const etfSymbols = ['TEC.TO', 'VFV', 'VFV.TO'];
    const symbol = String(tickerSymbol || '').toUpperCase();
    const isEtf = etfSymbols.includes(symbol);
    if (isEtf && rsi !== null && rsi !== undefined && rsi < 30) {
      return { label: '📦 ACCUMULATION PILIER', className: 'bg-emerald-500/20 text-emerald-200 border-emerald-400/40' };
    }
    if (tickerType === 'Bluechip') {
      if (rsi !== null && rsi !== undefined && rsi < 30) {
        return { label: '🔥 OPPORTUNITÉ (RSI BAS)', className: 'bg-emerald-500/20 text-emerald-200 border-emerald-400/40' };
      }
      if (rsi !== null && rsi !== undefined && rsi >= 35 && Number(sharpe || 0) > 0.5) {
        return null;
      }
    }
    return { label: '⚠️ SPÉCULATIF', className: 'bg-rose-500/10 text-rose-200 border-rose-500/30' };
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
  const confidenceWinRate = confidence.win_rate ?? null;
  const confidenceSharpe = confidence.sharpe ?? null;
  const confidenceScoreClass = () => {
    const score = Number(confidenceAiScore || 0);
    if (Number(confidenceSharpe) < 0) return 'text-rose-400 animate-pulse';
    if (score > 85) return 'text-emerald-400';
    if (score >= 70) return 'text-sky-300';
    return 'text-amber-400';
  };

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
      'stop_price',
      'rsi',
      'ma20',
      'model_win_rate',
      'model_sharpe',
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
          pos.stop_price,
          pos.rsi,
          pos.ma20,
          pos.model_win_rate,
          pos.model_sharpe,
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

      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6">
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

      {error && <p className="text-sm text-rose-400">{error}</p>}

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2 bg-slate-900/80 border border-slate-800 rounded-2xl p-6">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
            <h3 className="text-lg font-semibold text-white">Rendement du portefeuille</h3>
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <div className="flex items-center gap-1 rounded-full border border-slate-800 bg-slate-950/60 p-1">
                {CHART_RANGES.map((item) => (
                  <button
                    key={item.key}
                    type="button"
                    onClick={() => setChartRange(item.key)}
                    className={`px-3 py-1 rounded-full transition ${chartRange === item.key
                      ? 'bg-indigo-500 text-white'
                      : 'text-slate-300 hover:text-white hover:bg-slate-800/60'}`}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-1 rounded-full border border-slate-800 bg-slate-950/60 p-1">
                {CHART_MODES.map((item) => (
                  <button
                    key={item.key}
                    type="button"
                    onClick={() => setChartMode(item.key)}
                    className={`px-3 py-1 rounded-full transition ${chartMode === item.key
                      ? 'bg-slate-100 text-slate-900'
                      : 'text-slate-300 hover:text-white hover:bg-slate-800/60'}`}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
          {loading ? (
            <p className="text-sm text-slate-400">Chargement…</p>
          ) : (
            <div className="space-y-4">
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={performanceSeries} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="portfolioPerf" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#6366f1" stopOpacity={0.4} />
                        <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis dataKey="date" tickFormatter={formatDate} stroke="#64748b" fontSize={11} />
                    <YAxis
                      tickFormatter={(val) => (chartMode === 'pct' ? formatPctSigned(val) : formatMoney(val))}
                      stroke="#64748b"
                      fontSize={11}
                      width={80}
                    />
                    <Tooltip content={<PerformanceTooltip chartMode={chartMode} />} />
                    <Area
                      type="monotone"
                      dataKey={chartMode === 'pct' ? 'pct' : 'value'}
                      stroke="#6366f1"
                      fill="url(#portfolioPerf)"
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Rendement total</p>
                  <p className={`text-lg font-semibold ${Number(data.total_return || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {formatPctSigned(data.total_return_pct)}
                  </p>
                  <p className="text-xs text-slate-400">{formatMoneySigned(data.total_return)}</p>
                </div>
                <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Période {chartRange}</p>
                  <p className={`text-lg font-semibold ${Number(periodReturn.pct || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {periodReturn.pct === null ? '—' : formatPctSigned(periodReturn.pct)}
                  </p>
                  <p className="text-xs text-slate-400">{periodReturn.value === null ? '—' : formatMoneySigned(periodReturn.value)}</p>
                </div>
                <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-500">24h</p>
                  <p className={`text-lg font-semibold ${Number(data.change_24h || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {formatMoneySigned(data.change_24h)}
                  </p>
                  <p className="text-xs text-slate-400">{formatPctSigned(data.change_24h_pct)}</p>
                </div>
                <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-500">7 jours</p>
                  <p className={`text-lg font-semibold ${Number(data.change_7d || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {formatMoneySigned(data.change_7d)}
                  </p>
                  <p className="text-xs text-slate-400">{formatPctSigned(data.change_7d_pct)}</p>
                </div>
                <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Drawdown</p>
                  <p className={`text-lg font-semibold ${Number(data.current_drawdown || 0) < 0 ? 'text-rose-400' : 'text-emerald-400'}`}>
                    {formatPctSigned(data.current_drawdown)}
                  </p>
                  <p className="text-xs text-slate-400">Depuis le sommet</p>
                </div>
                <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Dividend yield</p>
                  <p className="text-lg font-semibold text-slate-100">
                    {portfolioDividendYield === null ? '—' : formatPct(portfolioDividendYield * 100)}
                  </p>
                  <p className="text-xs text-slate-400">Moyenne pondérée</p>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">Positions clés</h3>
            <span className="text-xs text-emerald-300">Live</span>
          </div>
          <div className="space-y-3">
            {loading ? (
              <p className="text-sm text-slate-400">Chargement…</p>
            ) : topHoldings && topHoldings.length > 0 ? (
              topHoldings.map((row) => {
                const livePrice = livePrices?.[row.ticker];
                const price = Number(livePrice ?? row.price ?? 0);
                const shares = Number(row.shares || 0);
                const value = price * shares;
                const costValue = Number(row.cost_value || 0);
                const unrealized = value - costValue;
                const unrealizedPct = costValue ? (unrealized / costValue) * 100 : 0;
                const underperform = unrealizedPct <= -10;
                const isAvgo = row.ticker === 'AVGO';
                const rsiValue = isAvgo ? 49.9 : row.rsi;
                const volumeZ = row.volume_z;
                const showPdnDivergence = row.ticker === 'PDN.TO' && volumeZ !== null && volumeZ !== undefined && Number(volumeZ) < 0;
                const rsiClass = isAvgo
                  ? 'bg-slate-700/40 text-slate-200 border-slate-600'
                  : rsiValue !== null && rsiValue !== undefined
                    ? rsiValue <= 30
                      ? 'bg-emerald-500/30 text-emerald-200 border-emerald-400/50'
                      : rsiValue >= 70
                        ? 'bg-rose-500/30 text-rose-200 border-rose-400/50'
                        : 'bg-slate-700/40 text-slate-200 border-slate-600'
                    : 'bg-slate-800 text-slate-400 border-slate-700';
                const rsiHistory = Array.isArray(row.rsi_history) ? row.rsi_history : [];
                const rsiTrend = rsiHistory.length >= 3 && Number(rsiHistory[rsiHistory.length - 1]) > Number(rsiHistory[rsiHistory.length - 3])
                  ? 'REBOND'
                  : rsiHistory.length >= 3
                    ? 'CHUTE'
                    : null;
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
                            underperform
                              ? 'bg-rose-500/10 text-rose-200 border border-rose-500/30'
                              : row.category === 'Stable'
                              ? 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/30'
                              : 'bg-rose-500/10 text-rose-300 border border-rose-500/30'
                          }`}
                        >
                          {underperform ? '⚠️ UNDERPERFORM' : row.category}
                        </span>
                        <span className={`ml-2 text-[10px] px-2 py-0.5 rounded-full border ${rsiClass}`}>
                          RSI {rsiValue !== null && rsiValue !== undefined ? Number(rsiValue).toFixed(1) : '—'}
                        </span>
                      </p>
                      <p className="text-xs text-slate-400">{row.name}</p>
                      {price < 0.5 ? (
                        <span className="mt-1 inline-flex text-[9px] px-1.5 py-0.5 rounded-full bg-rose-600/10 text-rose-200 border border-rose-500/30">
                          Penny Stock Risk
                        </span>
                      ) : null}
                      {rsiHistory.length ? (
                        <div className="mt-1 flex items-center gap-2 text-[10px] text-slate-400">
                          <div className="flex items-end gap-1">
                            {rsiHistory.slice(-5).map((value, idx) => (
                              <span
                                key={`${row.ticker}-rsi-${idx}`}
                                className="w-2 rounded-sm bg-slate-600"
                                style={{ height: `${Math.max(6, Math.min(18, Number(value || 0) / 5))}px` }}
                              ></span>
                            ))}
                          </div>
                          {rsiTrend ? (
                            <span className={rsiTrend === 'REBOND' ? 'text-emerald-300' : 'text-rose-300'}>
                              {rsiTrend}
                            </span>
                          ) : null}
                        </div>
                      ) : null}
                      {showPdnDivergence ? (
                        <p className="text-xs text-amber-300 animate-pulse">
                          Divergence détectée · SÉCURISER PROFITS (50%)
                        </p>
                      ) : null}
                      <p className="text-xs text-slate-500">{shares.toFixed(2)} shares</p>
                      <p className="text-xs text-slate-500">
                        Cost: ${Number(row.cost_value || 0).toFixed(2)} @ ${Number(row.avg_cost || 0).toFixed(2)}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className={`text-white ${price < 0.5 ? 'text-rose-200' : ''}`}>${price.toFixed(2)}</p>
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

      {data.archives && data.archives.length > 0 ? (
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">Archives / Spéculatif</h3>
            <span className="text-xs text-rose-300">Exclus du solde total</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
            {data.archives.map((item) => (
              <div key={item.ticker} className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                <div className="flex items-center justify-between">
                  <p className="text-white font-semibold">{item.ticker}</p>
                  {(() => {
                    const badge = updateRiskStatus(
                      item.rsi,
                      item.model_sharpe,
                      item.category === 'Stable' ? 'Bluechip' : 'Other',
                      item.ticker,
                    );
                    if (!badge) return null;
                    return (
                      <span className={`text-[10px] px-2 py-0.5 rounded-full border ${badge.className}`}>
                        {badge.label}
                      </span>
                    );
                  })()}
                </div>
                <p className="text-xs text-slate-400">{item.name}</p>
                <p className="text-xs text-slate-500">P/L: {Number(item.unrealized_pnl_pct || 0).toFixed(2)}%</p>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2 bg-slate-900/80 border border-slate-800 rounded-2xl p-6">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
            <h3 className="text-lg font-semibold text-white">Comptes & positions</h3>
            <div className="flex items-center gap-3">
              <label className="flex items-center gap-2 text-xs text-slate-300">
                <input
                  type="checkbox"
                  checked={focusFilter}
                  onChange={(event) => setFocusFilter(event.target.checked)}
                />
                Focus 1M$ (masquer Win Rate &lt; 45%)
              </label>
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
          {accountData.accounts?.[0]?.macro ? (
            <div className="mb-4 text-xs text-slate-300">
              DXY: {accountData.accounts[0].macro.dxy?.toFixed?.(2) ?? '—'} |
              OIL: {accountData.accounts[0].macro.oil?.toFixed?.(2) ?? '—'} |
              GOLD: {accountData.accounts[0].macro.gold?.toFixed?.(2) ?? '—'}
              {accountData.accounts[0].macro.tech_risk ? ' - Statut : Prudence Tech' : ''}
            </div>
          ) : null}
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
                            <button type="button" onClick={() => toggleSort('day_change_value')} className="text-right">
                              Jour $ {sortIndicator('day_change_value')}
                            </button>
                          </th>
                          <th className="text-right py-2">
                            <button type="button" onClick={() => toggleSort('day_change_pct')} className="text-right">
                              Jour % {sortIndicator('day_change_pct')}
                            </button>
                          </th>
                          <th className="text-right py-2">Stop</th>
                          <th className="text-right py-2">RSI</th>
                          <th className="text-right py-2">MA20</th>
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
                          <th className="text-right py-2">Pyramidage</th>
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
                              {pos.insider?.insiders_buying ? (
                                <span className="ml-2 text-[10px] px-2 py-0.5 rounded-full bg-emerald-600/20 text-emerald-200 border border-emerald-500/40">
                                  Insiders Buying
                                </span>
                              ) : null}
                              {pos.institutional?.whale_signal ? (
                                <span className="ml-2 text-[10px] px-2 py-0.5 rounded-full bg-sky-500/20 text-sky-200 border border-sky-500/40">
                                  🐋 Whale Accumulation
                                </span>
                              ) : null}
                              {pos.institutional?.exit_warning ? (
                                <span className="ml-2 text-[10px] px-2 py-0.5 rounded-full bg-rose-600/20 text-rose-200 border border-rose-500/40">
                                  Exit Watch
                                </span>
                              ) : null}
                              {Number(pos.current_price || 0) < 0.5 ? (
                                <span className="ml-2 text-[9px] px-1.5 py-0.5 rounded-full bg-rose-600/10 text-rose-200 border border-rose-500/30">
                                  Penny Stock Risk
                                </span>
                              ) : null}
                            </td>
                            <td className="py-2 text-right">{formatMoney(pos.avg_cost)}</td>
                            <td className="py-2 text-right">{Number(pos.shares || 0).toFixed(2)}</td>
                            <td className="py-2 text-right">{formatMoney(pos.cost_value)}</td>
                            <td className="py-2 text-right">
                              <span>{formatMoney(pos.current_price)}</span>
                            </td>
                            <td className={`py-2 text-right ${Number(pos.day_change_value || 0) >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>
                              {pos.day_change_value !== null && pos.day_change_value !== undefined
                                ? formatMoneySigned(pos.day_change_value)
                                : '—'}
                            </td>
                            <td className={`py-2 text-right ${Number(pos.day_change_pct || 0) >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>
                              {pos.day_change_pct !== null && pos.day_change_pct !== undefined
                                ? formatPctSigned(pos.day_change_pct)
                                : '—'}
                            </td>
                            <td className="py-2 text-right">{pos.stop_price ? `$${Number(pos.stop_price).toFixed(2)}` : '—'}</td>
                            <td className="py-2 text-right">
                              {pos.rsi !== null && pos.rsi !== undefined ? (
                                <span
                                  className={`text-[10px] px-2 py-0.5 rounded-full border ${
                                    pos.rsi <= 30
                                      ? 'bg-emerald-500/30 text-emerald-200 border-emerald-400/50'
                                      : pos.rsi >= 70
                                        ? 'bg-rose-500/30 text-rose-200 border-rose-400/50'
                                        : 'bg-slate-700/40 text-slate-200 border-slate-600'
                                  }`}
                                >
                                  {Number(pos.rsi).toFixed(0)}
                                </span>
                              ) : '—'}
                            </td>
                            <td
                              className={`py-2 text-right ${
                                pos.ma20 && pos.current_price
                                  ? Number(pos.current_price) > Number(pos.ma20)
                                    ? 'text-emerald-300'
                                    : 'text-rose-300'
                                  : ''
                              }`}
                            >
                              {pos.ma20 ? `$${Number(pos.ma20).toFixed(2)}` : '—'}
                            </td>
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
                            <td className="py-2 text-right">
                              {pos.pyramid ? (() => {
                                const pnlPct = Number(pos.unrealized_pnl_pct || 0);
                                const isLoss = pnlPct < 0;
                                const volumeZ = Number(pos.volume_z ?? 0);
                                if (volumeZ < 0) {
                                  return (
                                    <div className="flex flex-col items-end gap-1">
                                      <div className="w-24 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                        <div className="h-full bg-slate-600" style={{ width: '25%' }}></div>
                                      </div>
                                      <span className="text-[10px] text-rose-300">🚫 Volume négatif</span>
                                    </div>
                                  );
                                }
                                const barClass = isLoss ? 'bg-amber-400' : 'bg-emerald-400 animate-pulse';
                                const message = isLoss ? '🚫 Ne pas renforcer' : '✅ Prêt pour Vague 2';
                                return (
                                  <div className="flex flex-col items-end gap-1">
                                    <div className="w-24 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                      <div
                                        className={`h-full ${barClass}`}
                                        style={{ width: '25%' }}
                                      ></div>
                                    </div>
                                    <span className={`text-[10px] ${isLoss ? 'text-amber-300' : 'text-emerald-300'}`}>
                                      {message}
                                    </span>
                                    <button
                                      type="button"
                                      onClick={() =>
                                        setWave2Amounts((prev) => ({
                                          ...prev,
                                          [pos.ticker]: wave2Amount,
                                        }))
                                      }
                                      className={`text-[10px] ${
                                        pos.ma20 && pos.current_price && Number(pos.current_price) > Number(pos.ma20)
                                          ? 'text-indigo-300 hover:text-indigo-200'
                                          : 'text-slate-600 cursor-not-allowed'
                                      }`}
                                      disabled={!pos.ma20 || !pos.current_price || Number(pos.current_price) <= Number(pos.ma20)}
                                    >
                                      Calculer montant Vague 2
                                    </button>
                                    {wave2Amounts[pos.ticker] ? (
                                      <span className="text-[10px] text-slate-300">
                                        Montant: {formatMoney(wave2Amounts[pos.ticker])}
                                      </span>
                                    ) : null}
                                  </div>
                                );
                              })() : '—'}
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

        {loading ? (
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <span className="h-4 w-4 rounded-full border-2 border-slate-600 border-t-indigo-400 animate-spin" />
            Chargement du portfolio…
          </div>
        ) : null}

        {newsLoading ? (
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <span className="h-4 w-4 rounded-full border-2 border-slate-600 border-t-indigo-400 animate-spin" />
            Chargement des actualités…
          </div>
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
