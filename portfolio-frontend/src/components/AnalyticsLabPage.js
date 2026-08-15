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
import { Term } from './ui/Tooltip';
import { GLOSSARY } from '../glossary';
import { tickerDetailPath } from '../tickerUrl';

// Même iconographie que AnalysisScreener.js, pour la cohérence entre les
// deux pages (le screener et cette vue de performance montrent les mêmes
// verdicts sur les mêmes positions).
const LAB_VERDICT_BADGES = {
  confirmed: { icon: '✅', label: 'Confirmé', className: 'bg-emerald-500/15 text-emerald-200 border-emerald-500/30' },
  uncertain: { icon: '⚠️', label: 'Incertain', className: 'bg-amber-500/15 text-amber-200 border-amber-500/30' },
};

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
  const [labPerf, setLabPerf] = useState(null);
  const [labLoading, setLabLoading] = useState(true);
  const [labError, setLabError] = useState(null);
  const [labSortDesc, setLabSortDesc] = useState(true);
  const [labSuggestions, setLabSuggestions] = useState([]);
  const [labSuggestionsLoading, setLabSuggestionsLoading] = useState(true);

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

  // Fetch séparé, pas mergé dans le Promise.all ci-dessus : loadBacktest()
  // vide TOUS ses états sur la moindre erreur (un seul catch pour 5 appels),
  // et FUNDAMENTAL_LAB est un module distinct (analysis app) qui ne doit pas
  // pouvoir faire disparaître le reste de la page (backtest, sandboxes...)
  // s'il échoue. Même pattern de rafraîchissement que le reste de la page
  // malgré tout : fetch au chargement seulement, pas de polling.
  useEffect(() => {
    let isMounted = true;
    const loadLabPerf = async () => {
      setLabLoading(true);
      setLabError(null);
      try {
        const data = await cachedGet('analysis/lab/performance/', {}, 30000);
        if (!isMounted) return;
        setLabPerf(data);
      } catch (err) {
        if (!isMounted) return;
        setLabPerf(null);
        setLabError("Impossible de charger les performances du Fundamental Lab.");
      } finally {
        if (isMounted) setLabLoading(false);
      }
    };
    loadLabPerf();
    return () => {
      isMounted = false;
    };
  }, []);

  // Partie 3 (2026-08-14) : suggestions hebdomadaires automatiques
  // (analysis.tasks.generate_lab_weekly_suggestions) -- fetch séparé, même
  // raisonnement que labPerf ci-dessus (module distinct, ne doit jamais
  // faire échouer le reste de la page).
  useEffect(() => {
    let isMounted = true;
    const loadLabSuggestions = async () => {
      setLabSuggestionsLoading(true);
      try {
        const data = await cachedGet('analysis/lab/suggestions/', {}, 60000);
        if (!isMounted) return;
        setLabSuggestions(data?.suggestions || []);
      } catch (err) {
        if (!isMounted) return;
        setLabSuggestions([]);
      } finally {
        if (isMounted) setLabSuggestionsLoading(false);
      }
    };
    loadLabSuggestions();
    return () => {
      isMounted = false;
    };
  }, []);

  const labPositionsSorted = useMemo(() => {
    const positions = labPerf?.positions || [];
    return [...positions].sort((a, b) => {
      const av = a.current_return_pct;
      const bv = b.current_return_pct;
      if (av === null || av === undefined) return 1;
      if (bv === null || bv === undefined) return -1;
      return labSortDesc ? bv - av : av - bv;
    });
  }, [labPerf, labSortDesc]);

  // Used to plot a 60-month "Projection 5y" line by seeding a monthly
  // dollar contribution ($2300) onto the tail of the 12-month backtest
  // curve, then appending those points to the SAME chart/axis as the
  // "Backtest Comparator" (Buy & Hold vs Strategy, 12m). Confirmed live
  // (2026-08-15) this made the chart actively misleading, not just
  // busy: 5 years of accumulated contributions (~$138k) dwarf a 12-month
  // starting-capital curve, so Buy & Hold/Strategy render as a flat line
  // pinned near the bottom while a disconnected-looking green line shoots
  // up on the right -- exactly the opposite of what a comparator chart is
  // for (seeing whether Strategy beat Buy & Hold). Removed rather than
  // rescaled: nothing else on this page reads `chartData` expecting a
  // projection tail, and the stated chart title/subtitle ("Backtest
  // Comparator" · "12m") never promised a 5-year forward projection.
  const chartData = useMemo(() => {
    if (!backtest?.dates?.length) return [];
    return backtest.dates.map((date, index) => ({
      date,
      buyHold: backtest.buy_hold_curve?.[index] ?? null,
      strategy: backtest.equity_curve?.[index] ?? null,
    }));
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
    AI_CRYPTO: 'Sandbox 4 · AI Crypto',
  };

  const orderedSandboxStats = useMemo(() => {
    const map = new Map((sandboxStats || []).map((stat) => [stat.sandbox, stat]));
    return ['WATCHLIST', 'AI_BLUECHIP', 'AI_PENNY', 'AI_CRYPTO'].map((key) => (
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
    return ['WATCHLIST', 'AI_BLUECHIP', 'AI_PENNY', 'AI_CRYPTO'].map((key) => (
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

      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
        <div className="mb-4">
          <p className="text-white font-semibold">Fundamental Lab</p>
          <p className="text-xs text-slate-400">
            Picks du screener fondamental (Claude/DeepSeek) — confirmed vs uncertain vs picks manuels
          </p>
        </div>
        {labLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[1, 2, 3].map((item) => (
              <div key={item} className="h-20 rounded-2xl border border-slate-800 bg-slate-900/60 animate-pulse" />
            ))}
          </div>
        ) : labError ? (
          <p className="text-rose-300 text-sm">{labError}</p>
        ) : labPositionsSorted.length === 0 ? (
          <p className="text-slate-400 text-sm">
            Aucune position pour l'instant — lance un scan depuis le Screener fondamental.
          </p>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              <div className="bg-slate-950/60 border border-slate-800 rounded-2xl p-4">
                <p className="text-xs text-slate-400 uppercase tracking-[0.2em]">✅ Confirmed (n={labPerf.confirmed.count})</p>
                <p className="text-2xl text-white font-semibold">{formatPct(labPerf.confirmed.avg_return_pct)}</p>
              </div>
              <div className="bg-slate-950/60 border border-slate-800 rounded-2xl p-4">
                <p className="text-xs text-slate-400 uppercase tracking-[0.2em]">⚠️ Uncertain (n={labPerf.uncertain.count})</p>
                <p className="text-2xl text-white font-semibold">{formatPct(labPerf.uncertain.avg_return_pct)}</p>
              </div>
              <div className="bg-slate-950/60 border border-slate-800 rounded-2xl p-4">
                <p className="text-xs text-slate-400 uppercase tracking-[0.2em]">⭐ Picks manuels (n={labPerf.manual_picks.count})</p>
                <p className="text-2xl text-white font-semibold">{formatPct(labPerf.manual_picks.avg_return_pct)}</p>
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead className="text-slate-400">
                  <tr>
                    <th className="text-left px-3 py-2">Ticker</th>
                    <th className="text-left px-3 py-2">Verdict</th>
                    <th className="text-left px-3 py-2">Entrée</th>
                    <th className="text-left px-3 py-2">Actuel</th>
                    <th
                      className="text-left px-3 py-2 cursor-pointer select-none"
                      onClick={() => setLabSortDesc((prev) => !prev)}
                    >
                      Rendement {labSortDesc ? '↓' : '↑'}
                    </th>
                    <th className="text-left px-3 py-2">Manuel</th>
                    <th className="text-left px-3 py-2">Entrée le</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-900">
                  {labPositionsSorted.map((p) => {
                    const badge = LAB_VERDICT_BADGES[p.final_verdict] || {};
                    const entry = Number(p.entry_price);
                    const hasReturn = p.current_return_pct !== null && p.current_return_pct !== undefined;
                    const currentPrice = hasReturn ? entry * (1 + p.current_return_pct / 100) : null;
                    const priceDigits = entry < 5 ? 4 : 2;
                    const returnColor = !hasReturn
                      ? 'text-slate-400'
                      : p.current_return_pct >= 0 ? 'text-emerald-300' : 'text-rose-300';
                    return (
                      <tr key={p.id} className={`hover:bg-slate-900/40 ${!p.is_open ? 'opacity-50' : ''}`}>
                        <td className="px-3 py-2 text-slate-100">
                          <a href={tickerDetailPath(p.ticker)} className="hover:underline">{p.ticker}</a>
                          {!p.is_open && (
                            <span className="ml-2 text-xs text-slate-500" title={p.manual_note || undefined}>
                              (fermée)
                            </span>
                          )}
                        </td>
                        <td className="px-3 py-2">
                          <span className={`inline-flex items-center px-2 py-1 rounded-full border text-xs ${badge.className || 'bg-slate-700/30 text-slate-200 border-slate-600/40'}`}>
                            {badge.icon} {badge.label || p.final_verdict}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-slate-300">{formatNumber(entry, priceDigits)}</td>
                        <td className="px-3 py-2 text-slate-300">
                          {currentPrice === null ? '—' : formatNumber(currentPrice, priceDigits)}
                        </td>
                        <td className={`px-3 py-2 font-medium ${returnColor}`}>
                          {hasReturn ? formatPct(p.current_return_pct) : '—'}
                        </td>
                        <td className="px-3 py-2 text-center" title={p.manually_selected ? p.manual_note || undefined : undefined}>
                          {p.manually_selected ? '⭐' : '—'}
                        </td>
                        <td className="px-3 py-2 text-slate-400 text-xs">
                          {new Date(p.entry_date).toLocaleDateString('fr-CA')}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
        <div className="mb-3">
          <p className="text-white font-semibold">💡 Suggestions de la semaine</p>
          <p className="text-xs text-slate-400">
            Patterns automatiques (rendement réel vs facteur -- secteur, confiance Claude, ROE, FCF Yield)
            détectés sur les positions Fundamental Lab fermées. Jamais appliqué automatiquement -- à valider
            manuellement.
          </p>
        </div>
        {labSuggestionsLoading ? (
          <div className="h-16 rounded-xl border border-slate-800 bg-slate-900/60 animate-pulse" />
        ) : labSuggestions.length === 0 ? (
          <p className="text-slate-400 text-sm">Aucune suggestion générée pour l'instant.</p>
        ) : (
          <div className="space-y-2">
            {labSuggestions.map((s) => (
              <div
                key={s.id}
                className={`rounded-xl border px-3 py-2 text-sm ${
                  s.has_pattern
                    ? 'border-amber-500/30 bg-amber-500/10 text-amber-100'
                    : 'border-slate-800 bg-slate-950/60 text-slate-400'
                }`}
              >
                <div className="flex items-center justify-between text-xs mb-1 opacity-80">
                  <span>Semaine du {new Date(s.week_start).toLocaleDateString('fr-CA')}</span>
                  <span>{s.positions_closed_analyzed} position(s) analysée(s)</span>
                </div>
                <p className="whitespace-pre-wrap break-words">{s.summary}</p>
              </div>
            ))}
          </div>
        )}
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
                <option value="AI_CRYPTO">Sandbox 4: AI Crypto</option>
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
                  {(sandboxFilter === 'ALL' || sandboxFilter === 'AI_CRYPTO') && (
                    <Line type="monotone" dataKey="AI_CRYPTO" name="Sandbox 4 · AI Crypto" stroke="#8b5cf6" strokeWidth={2} dot={false} />
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
                    <div className="flex justify-between"><Term text={GLOSSARY.winRate}><span>Win Rate</span></Term><span>{formatPct(stat.win_rate, 2)}</span></div>
                    <div className="flex justify-between"><Term text={GLOSSARY.totalReturn}><span>Total Return</span></Term><span>{formatPct(stat.total_return_pct, 2)}</span></div>
                    <div className="flex justify-between"><Term text={GLOSSARY.sharpeRatio}><span>Sharpe Ratio</span></Term><span>{formatNumber(stat.sharpe_ratio, 3)}</span></div>
                    <div className="flex justify-between"><Term text={GLOSSARY.maxDrawdown}><span>Max Drawdown</span></Term><span>{formatPct(stat.max_drawdown, 2)}</span></div>
                    <div className="flex justify-between"><Term text={GLOSSARY.finalBalance}><span>Final Balance</span></Term><span>{formatMoney(stat.final_balance)}</span></div>
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
                    <div className="flex justify-between"><Term text={GLOSSARY.winRate}><span>Win Rate</span></Term><span>{formatPct(stat.win_rate, 2)}</span></div>
                    <div className="flex justify-between"><Term text={GLOSSARY.totalReturn}><span>Total Return</span></Term><span>{formatPct(stat.total_return_pct, 2)}</span></div>
                    <div className="flex justify-between"><Term text={GLOSSARY.sharpeRatio}><span>Sharpe Ratio</span></Term><span>{formatNumber(stat.sharpe_ratio, 3)}</span></div>
                    <div className="flex justify-between"><Term text={GLOSSARY.maxDrawdown}><span>Max Drawdown</span></Term><span>{formatPct(stat.max_drawdown, 2)}</span></div>
                    <div className="flex justify-between"><Term text={GLOSSARY.finalBalance}><span>Final Balance</span></Term><span>{formatMoney(stat.final_balance)}</span></div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 h-72">
          <div className="mb-4">
            <p className="text-white font-semibold">
              <Term text={GLOSSARY.featureImportance}>Feature Importance</Term>
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
                    <th className="px-2 py-2"><Term text={GLOSSARY.confidence}>Confidence</Term></th>
                    <th className="px-2 py-2"><Term text={GLOSSARY.sentiment}>Sentiment</Term></th>
                    <th className="px-2 py-2"><Term text={GLOSSARY.imbalance}>Imbalance</Term></th>
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
