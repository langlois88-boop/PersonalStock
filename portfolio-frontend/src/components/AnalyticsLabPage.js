import { useEffect, useMemo, useRef, useState } from 'react';
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
import { cachedGet, invalidateCache } from '../api/cachedApi';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';

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

const downloadCsv = (filename, rows) => {
  if (!rows || rows.length === 0) return;
  const csv = rows.map((row) => row.map((cell) => {
    const value = cell === null || cell === undefined ? '' : String(cell);
    return value.includes(',') || value.includes('"') || value.includes('\n')
      ? `"${value.replace(/"/g, '""')}"`
      : value;
  }).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

function AnalyticsLabPage() {
  const reportRef = useRef(null);
  const [backtest, setBacktest] = useState(null);
  const [sandboxStats, setSandboxStats] = useState([]);
  const [alpacaStats, setAlpacaStats] = useState([]);
  const [calibration, setCalibration] = useState(null);
  const [sandboxCurve, setSandboxCurve] = useState({ dates: [], series: [] });
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [universe, setUniverse] = useState('BLUECHIP');
  const [sandboxFilter, setSandboxFilter] = useState('ALL');
  const [diagnostics, setDiagnostics] = useState([]);
  const [diagnosticMeta, setDiagnosticMeta] = useState(null);
  const [diagnosticLoading, setDiagnosticLoading] = useState(false);
  const [diagnosticError, setDiagnosticError] = useState(null);

  useEffect(() => {
    let isMounted = true;

    const loadBacktest = async () => {
      try {
        const backtestParams = { symbol: 'SPY', days: 365, universe };
        const [backtestData, sandboxData, alpacaData, curveData, healthData] = await Promise.all([
          cachedGet('backtester/', backtestParams, 60000),
          cachedGet('paper-trades/performance/', {}, 30000),
          cachedGet('paper-trades/performance/', { broker: 'ALPACA' }, 30000),
          cachedGet('paper-trades/equity/', {}, 60000),
          cachedGet('health/', {}, 60000),
        ]);
        if (!isMounted) return;
        setBacktest(backtestData);
        setSandboxStats(sandboxData?.results || []);
        setAlpacaStats(alpacaData?.results || []);
        setSandboxCurve(curveData || { dates: [], series: [] });
        setHealth(healthData || null);
        const calibrationSandbox = sandboxFilter === 'ALL' ? 'WATCHLIST' : sandboxFilter;
        try {
          const calibrationRes = await cachedGet('models/calibration/', { model: universe, sandbox: calibrationSandbox }, 60000);
          if (!isMounted) return;
          const latest = calibrationRes?.results?.[0] || null;
          setCalibration(latest);
        } catch (err) {
          if (!isMounted) return;
          setCalibration(null);
        }
      } catch (err) {
        if (!isMounted) return;
        setBacktest(null);
        setSandboxStats([]);
        setAlpacaStats([]);
        setCalibration(null);
        setSandboxCurve({ dates: [], series: [] });
        setHealth(null);
      } finally {
        if (isMounted) setLoading(false);
      }
    };

    loadBacktest();
    return () => {
      isMounted = false;
    };
  }, [sandboxFilter, universe]);

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

    const monthlyContribution = 2300;
    const monthlyRate = cagr / 12;
    let projected = lastValue;
    const projectionPoints = Array.from({ length: 60 }, (_, i) => {
      const month = i + 1;
      projected = projected * (1 + monthlyRate) + monthlyContribution;
      return {
        date: `+${month}m`,
        buyHold: null,
        strategy: null,
        projection: projected,
      };
    });

    return [...base, ...projectionPoints];
  }, [backtest]);

  const sandboxChartData = useMemo(() => {
    if (!sandboxCurve?.dates?.length || !sandboxCurve?.series?.length) return [];
    const series = sandboxFilter === 'ALL'
      ? sandboxCurve.series
      : sandboxCurve.series.filter((entry) => entry.sandbox === sandboxFilter);
    return sandboxCurve.dates.map((date, index) => {
      const row = { date };
      series.forEach((entry) => {
        row[entry.sandbox] = entry.equity_curve?.[index] ?? null;
      });
      return row;
    });
  }, [sandboxCurve, sandboxFilter]);

  const paperSharpe = useMemo(() => {
    if (!sandboxStats?.length) return null;
    const selected = sandboxFilter === 'ALL'
      ? sandboxStats.find((stat) => stat.sandbox === 'WATCHLIST')
      : sandboxStats.find((stat) => stat.sandbox === sandboxFilter);
    if (!selected) return null;
    return selected.sharpe_ratio ?? null;
  }, [sandboxStats, sandboxFilter]);

  const paperSummary = useMemo(() => {
    if (!sandboxStats?.length) return null;
    const selected = sandboxFilter === 'ALL'
      ? sandboxStats.find((stat) => stat.sandbox === 'WATCHLIST')
      : sandboxStats.find((stat) => stat.sandbox === sandboxFilter);
    if (!selected || !selected.trades) return null;
    return selected;
  }, [sandboxStats, sandboxFilter]);

  const winRateValue = paperSummary ? paperSummary.win_rate : backtest?.win_rate;
  const winRateLabel = paperSummary ? 'Win Rate (paper)' : 'Win Rate';
  const totalReturnValue = paperSummary ? paperSummary.total_return_pct : backtest?.total_return_pct;
  const totalReturnLabel = paperSummary ? 'Total Return (paper)' : 'Total Return';

  const reportMetrics = [
    { label: winRateLabel, value: formatPct(winRateValue, 2) },
    { label: totalReturnLabel, value: formatPct(totalReturnValue, 2) },
    { label: 'Sharpe Ratio', value: formatNumber(backtest?.sharpe_ratio, 3) },
    { label: 'Sharpe réel (paper)', value: paperSharpe === null ? '—' : formatNumber(paperSharpe, 3) },
    { label: 'Max Drawdown', value: formatPct((backtest?.max_drawdown ?? 0) * 100, 2) },
    { label: 'Final Balance', value: formatMoney(backtest?.final_balance) },
  ];

  const healthTasks = [
    { key: 'compute_continuous_evaluation_daily', label: 'Continuous Eval' },
    { key: 'auto_retrain_on_drift_daily', label: 'Drift Retrain' },
    { key: 'auto_rollback_models_daily', label: 'Auto Rollback' },
    { key: 'retrain_from_paper_trades_daily', label: 'Paper Retrain' },
  ];

  const statusStyle = (status) => {
    if (status === 'SUCCESS') return 'bg-emerald-500/20 text-emerald-300';
    if (status === 'FAILED') return 'bg-rose-500/20 text-rose-300';
    return 'bg-slate-500/20 text-slate-300';
  };

  const featureData = backtest?.feature_importance?.length
    ? backtest.feature_importance.map((item) => ({ name: item.name, value: item.value }))
    : [];

  const calibrationSummary = calibration
    ? `Brier ${formatNumber(calibration.brier_score, 3)} · n=${calibration.count}`
    : 'Calibration unavailable.';

  const sandboxLabels = {
    WATCHLIST: 'Sandbox 1 · Watchlist',
    AI_BLUECHIP: 'Sandbox 2 · AI Bluechip',
    AI_PENNY: 'Sandbox 3 · AI Penny',
  };

  const orderedSandboxStats = useMemo(() => {
    const map = new Map((sandboxStats || []).map((stat) => [stat.sandbox, stat]));
    return ['WATCHLIST', 'AI_BLUECHIP', 'AI_PENNY'].map((key) => (
      map.get(key) || {
        sandbox: key,
        initial_capital: 0,
        trades: 0,
        win_rate: 0,
        total_return_pct: 0,
        sharpe_ratio: 0,
        max_drawdown: 0,
        final_balance: 0,
      }
    ));
  }, [sandboxStats]);

  const orderedAlpacaStats = useMemo(() => {
    const map = new Map((alpacaStats || []).map((stat) => [stat.sandbox, stat]));
    return ['WATCHLIST', 'AI_BLUECHIP', 'AI_PENNY'].map((key) => (
      map.get(key) || {
        sandbox: key,
        initial_capital: 0,
        trades: 0,
        win_rate: 0,
        total_return_pct: 0,
        sharpe_ratio: 0,
        max_drawdown: 0,
        final_balance: 0,
      }
    ));
  }, [alpacaStats]);

  const exportPerformanceCsv = () => {
    if (!sandboxStats.length) return;
    const rows = [
      ['sandbox', 'initial_capital', 'trades', 'win_rate', 'total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'final_balance'],
      ...sandboxStats.map((stat) => ([
        stat.sandbox,
        stat.initial_capital,
        stat.trades,
        stat.win_rate,
        stat.total_return_pct,
        stat.sharpe_ratio,
        stat.max_drawdown,
        stat.final_balance,
      ])),
    ];
    downloadCsv('sandbox_performance.csv', rows);
  };

  const exportCurvesCsv = () => {
    if (!sandboxCurve?.dates?.length || !sandboxCurve?.series?.length) return;
    const header = ['date', ...sandboxCurve.series.map((entry) => entry.sandbox)];
    const rows = sandboxCurve.dates.map((date, index) => {
      const row = [date];
      sandboxCurve.series.forEach((entry) => {
        row.push(entry.equity_curve?.[index] ?? '');
      });
      return row;
    });
    downloadCsv('sandbox_equity_curves.csv', [header, ...rows]);
  };

  const exportPdf = async () => {
    if (!reportRef.current) return;
    const canvas = await html2canvas(reportRef.current, { scale: 2, backgroundColor: '#0f172a' });
    const imgData = canvas.toDataURL('image/png');
    const pdf = new jsPDF('p', 'mm', 'a4');
    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = pdf.internal.pageSize.getHeight();
    const imgWidth = pdfWidth;
    const imgHeight = (canvas.height * imgWidth) / canvas.width;
    let position = 0;

    pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
    let heightLeft = imgHeight - pdfHeight;
    while (heightLeft > 0) {
      position = heightLeft - imgHeight;
      pdf.addPage();
      pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
      heightLeft -= pdfHeight;
    }
    pdf.save('analytics_report.pdf');
  };

  const runDiagnostics = async () => {
    const sandbox = sandboxFilter === 'ALL' ? 'WATCHLIST' : sandboxFilter;
    setDiagnosticLoading(true);
    setDiagnosticError(null);
    try {
      invalidateCache('trading/diagnostics/', { sandbox });
      const res = await cachedGet('trading/diagnostics/', { sandbox }, 0);
      setDiagnostics(res?.results || []);
      setDiagnosticMeta(res || null);
    } catch (err) {
      setDiagnostics([]);
      setDiagnosticMeta(null);
      setDiagnosticError('Unable to load diagnostics.');
    } finally {
      setDiagnosticLoading(false);
    }
  };

  return (
    <div className="space-y-4" ref={reportRef}>
      <div className="flex items-center justify-between">
        <p className="text-white font-semibold">Rapport Analytics</p>
        <button
          type="button"
          onClick={exportPdf}
          className="text-xs text-indigo-300 hover:text-indigo-200"
        >
          Export PDF
        </button>
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2 space-y-6">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div className="flex items-center justify-between mb-4">
            <p className="text-white font-semibold">Backtest Comparator</p>
            <div className="flex items-center gap-3 text-xs text-slate-400">
              <span>{backtest?.symbol || 'SPY'} · 12m</span>
              <select
                value={universe}
                onChange={(event) => setUniverse(event.target.value)}
                className="bg-slate-950/60 border border-slate-800 text-slate-200 text-xs rounded-lg px-2 py-1"
              >
                <option value="BLUECHIP">Universe: Bluechip</option>
                <option value="PENNY">Universe: Penny</option>
              </select>
              <select
                value={sandboxFilter}
                onChange={(event) => setSandboxFilter(event.target.value)}
                className="bg-slate-950/60 border border-slate-800 text-slate-200 text-xs rounded-lg px-2 py-1"
              >
                <option value="ALL">Sandbox: All</option>
                <option value="WATCHLIST">Sandbox 1: Watchlist</option>
                <option value="AI_BLUECHIP">Sandbox 2: AI Bluechip</option>
                <option value="AI_PENNY">Sandbox 3: AI Penny</option>
              </select>
            </div>
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

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div className="flex items-center justify-between mb-4">
            <p className="text-white font-semibold">Sandbox Equity Curves</p>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400">Paper trades</span>
              <button
                type="button"
                onClick={exportCurvesCsv}
                className="text-xs text-indigo-300 hover:text-indigo-200"
                disabled={!sandboxCurve?.dates?.length}
              >
                Export CSV
              </button>
            </div>
          </div>
          <div className="h-64">
            {sandboxChartData.length === 0 ? (
              <p className="text-slate-400 text-sm">No sandbox equity data yet.</p>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={sandboxChartData} margin={{ top: 10, right: 24, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="date" hide />
                  <YAxis stroke="#64748b" />
                  <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1f2937' }} />
                  <Legend />
                  {(sandboxFilter === 'ALL' || sandboxFilter === 'WATCHLIST') && (
                    <Line type="monotone" dataKey="WATCHLIST" name="Sandbox 1 · Watchlist" stroke="#38bdf8" strokeWidth={2} dot={false} />
                  )}
                  {(sandboxFilter === 'ALL' || sandboxFilter === 'AI_BLUECHIP') && (
                    <Line type="monotone" dataKey="AI_BLUECHIP" name="Sandbox 2 · AI Bluechip" stroke="#22c55e" strokeWidth={2} dot={false} />
                  )}
                  {(sandboxFilter === 'ALL' || sandboxFilter === 'AI_PENNY') && (
                    <Line type="monotone" dataKey="AI_PENNY" name="Sandbox 3 · AI Penny" stroke="#f59e0b" strokeWidth={2} dot={false} />
                  )}
                  {sandboxCurve?.series?.some((entry) => entry.sandbox === 'SPY_BUY_HOLD') && (
                    <Line type="monotone" dataKey="SPY_BUY_HOLD" name="SPY Buy & Hold" stroke="#e2e8f0" strokeWidth={2} strokeDasharray="4 4" dot={false} />
                  )}
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {loading ? (
            [1, 2, 3].map((item) => (
              <div key={item} className="h-24 rounded-2xl border border-slate-800 bg-slate-900/60 animate-pulse" />
            ))
          ) : (
            reportMetrics.map((metric) => (
              <div key={metric.label} className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
                <p className="text-xs text-slate-400 uppercase tracking-[0.2em]">{metric.label}</p>
                <p className="text-2xl text-white font-semibold">{metric.value}</p>
              </div>
            ))
          )}
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div className="flex items-center justify-between mb-4">
            <p className="text-white font-semibold">Sandbox Performance</p>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400">Paper trades</span>
              <button
                type="button"
                onClick={exportPerformanceCsv}
                className="text-xs text-indigo-300 hover:text-indigo-200"
                disabled={!sandboxStats.length}
              >
                Export CSV
              </button>
            </div>
          </div>
          {sandboxStats.length === 0 ? (
            <p className="text-slate-400 text-sm">No sandbox results yet.</p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {orderedSandboxStats.map((stat) => (
                <div key={stat.sandbox} className="bg-slate-950/60 border border-slate-800 rounded-2xl p-4">
                  <p className="text-xs text-slate-400 uppercase tracking-[0.2em]">
                    {sandboxLabels[stat.sandbox] || stat.sandbox}
                  </p>
                  <div className="mt-3 space-y-2 text-sm text-slate-200">
                    <div className="flex justify-between"><span>Win Rate</span><span>{formatPct(stat.win_rate, 2)}</span></div>
                    <div className="flex justify-between"><span>Total Return</span><span>{formatPct(stat.total_return_pct, 2)}</span></div>
                    <div className="flex justify-between"><span>Sharpe Ratio</span><span>{formatNumber(stat.sharpe_ratio, 3)}</span></div>
                    <div className="flex justify-between"><span>Max Drawdown</span><span>{formatPct(stat.max_drawdown, 2)}</span></div>
                    <div className="flex justify-between"><span>Final Balance</span><span>{formatMoney(stat.final_balance)}</span></div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div className="flex items-center justify-between mb-4">
            <p className="text-white font-semibold">Alpaca Paper Performance</p>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400">Broker: Alpaca</span>
            </div>
          </div>
          {alpacaStats.length === 0 ? (
            <p className="text-slate-400 text-sm">No Alpaca paper trades yet.</p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {orderedAlpacaStats.map((stat) => (
                <div key={`alpaca-${stat.sandbox}`} className="bg-slate-950/60 border border-slate-800 rounded-2xl p-4">
                  <p className="text-xs text-slate-400 uppercase tracking-[0.2em]">
                    {sandboxLabels[stat.sandbox] || stat.sandbox}
                  </p>
                  <div className="mt-3 space-y-2 text-sm text-slate-200">
                    <div className="flex justify-between"><span>Win Rate</span><span>{formatPct(stat.win_rate, 2)}</span></div>
                    <div className="flex justify-between"><span>Total Return</span><span>{formatPct(stat.total_return_pct, 2)}</span></div>
                    <div className="flex justify-between"><span>Sharpe Ratio</span><span>{formatNumber(stat.sharpe_ratio, 3)}</span></div>
                    <div className="flex justify-between"><span>Max Drawdown</span><span>{formatPct(stat.max_drawdown, 2)}</span></div>
                    <div className="flex justify-between"><span>Final Balance</span><span>{formatMoney(stat.final_balance)}</span></div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 h-72">
          <div className="mb-4">
            <p className="text-white font-semibold">
              Feature Importance
              <span
                className="ml-2 text-xs text-slate-400 cursor-help"
                title="VolumeZ est le signal dominant pour éviter d'acheter quand le volume est faible."
              >
                ⓘ
              </span>
            </p>
            <p className="text-xs text-slate-400 mt-1">{calibrationSummary}</p>
          </div>
          {loading ? (
            <div className="h-48 rounded-xl bg-slate-900/60 animate-pulse" />
          ) : featureData.length === 0 ? (
            <p className="text-slate-400 text-sm">No feature importance yet.</p>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={featureData} layout="vertical">
                <XAxis type="number" stroke="#64748b" />
                <YAxis dataKey="name" type="category" stroke="#64748b" />
                <Bar dataKey="value" fill="#6366f1" radius={[8, 8, 8, 8]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
        <div className="space-y-6">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-white font-semibold">Live Diagnostics (Alpaca)</p>
              <p className="text-xs text-slate-400">Alpaca filters (confidence, sentiment, imbalance, spread)</p>
            </div>
            <button
              type="button"
              onClick={runDiagnostics}
              className="text-xs text-indigo-300 hover:text-indigo-200"
              disabled={diagnosticLoading}
            >
              {diagnosticLoading ? 'Running…' : 'Run Live Diagnostic'}
            </button>
          </div>
          {diagnosticMeta?.message && (
            <div className="text-xs text-amber-300 mb-3">{diagnosticMeta.message}</div>
          )}
          {diagnosticError ? (
            <p className="text-slate-400 text-sm">{diagnosticError}</p>
          ) : diagnostics.length === 0 ? (
            <p className="text-slate-400 text-sm">No diagnostics yet.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs text-slate-200">
                <thead className="text-slate-400">
                  <tr className="border-b border-slate-800">
                    <th className="px-2 py-2 text-left">Symbol</th>
                    <th className="px-2 py-2">Confidence</th>
                    <th className="px-2 py-2">Sentiment</th>
                    <th className="px-2 py-2">Imbalance</th>
                    <th className="px-2 py-2">Spread+Fees</th>
                  </tr>
                </thead>
                <tbody>
                  {diagnostics.map((row) => (
                    <tr key={row.symbol} className="border-b border-slate-800/60">
                      <td className="px-2 py-2 text-left">{row.symbol}</td>
                      <td className="px-2 py-2">
                        <span className={`rounded-full px-2 py-0.5 ${row.confidence_ok ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300'}`}>
                          {formatNumber(row.confidence, 2)}
                        </span>
                      </td>
                      <td className="px-2 py-2">
                        <span className={`rounded-full px-2 py-0.5 ${row.sentiment_ok ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300'}`}>
                          {formatNumber(row.sentiment, 2)}
                        </span>
                      </td>
                      <td className="px-2 py-2">
                        {row.imbalance === null ? (
                          <span className="rounded-full px-2 py-0.5 bg-slate-500/20 text-slate-300">n/a</span>
                        ) : (
                          <span className={`rounded-full px-2 py-0.5 ${row.imbalance_ok ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300'}`}>
                            {formatNumber(row.imbalance, 2)}
                          </span>
                        )}
                      </td>
                      <td className="px-2 py-2">
                        <span className={`rounded-full px-2 py-0.5 ${row.cost_ok ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300'}`}>
                          {formatPct((row.cost_pct || 0) * 100, 2)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <p className="text-white font-semibold mb-4">Health</p>
          {loading ? (
            <div className="space-y-2">
              {[1, 2, 3].map((item) => (
                <div key={item} className="h-10 rounded-lg bg-slate-900/60 animate-pulse" />
              ))}
            </div>
          ) : (
            <div className="space-y-2 text-xs text-slate-300">
              {healthTasks.map((task) => {
                const status = health?.tasks?.[task.key]?.status || 'UNKNOWN';
                return (
                  <div key={task.key} className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2">
                    <span>{task.label}</span>
                    <span className={`rounded-full px-2 py-0.5 ${statusStyle(status)}`}>{status}</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <p className="text-white font-semibold mb-4">Model Logs</p>
          <div className="space-y-3 text-xs text-slate-400 font-mono">
            {(backtest?.logs || []).map((log) => (
              <div key={log} className="bg-slate-950/60 p-3 rounded-lg">{log}</div>
            ))}
          </div>
        </div>
        </div>
      </div>
    </div>
  );
}

export default AnalyticsLabPage;
