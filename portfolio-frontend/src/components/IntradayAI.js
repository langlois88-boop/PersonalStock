import { useEffect, useMemo, useRef, useState } from 'react';
import { createChart } from 'lightweight-charts';
import api from '../api/api';
import MarketScannerPanel from './MarketScannerPanel';

const defaultSymbol = 'ONCY';

function IntradayAI() {
  const [symbol, setSymbol] = useState(defaultSymbol);
  const [bars, setBars] = useState([]);
  const [markers, setMarkers] = useState([]);
  const [guidance, setGuidance] = useState([]);
  const [stats, setStats] = useState(null);
  const [gemini, setGemini] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);

  const chartOptions = useMemo(
    () => ({
      layout: {
        background: { color: '#0b1120' },
        textColor: '#e2e8f0',
      },
      grid: {
        vertLines: { color: '#1e293b' },
        horzLines: { color: '#1e293b' },
      },
      timeScale: {
        borderColor: '#334155',
      },
      rightPriceScale: {
        borderColor: '#334155',
      },
      height: 480,
    }),
    []
  );

  useEffect(() => {
    if (!chartContainerRef.current) return undefined;
    const container = chartContainerRef.current;
    const chart = createChart(container, {
      ...chartOptions,
      width: container.clientWidth || 0,
    });
    const series = chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    });
    chartRef.current = chart;
    seriesRef.current = series;
    const handleResize = () => {
      if (!chartRef.current || !chartContainerRef.current) return;
      const width = chartContainerRef.current.clientWidth || 0;
      chartRef.current.applyOptions({ width });
      chartRef.current.timeScale().fitContent();
    };
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [chartOptions]);

  useEffect(() => {
    if (!seriesRef.current) return;
    seriesRef.current.setData(bars);
    seriesRef.current.setMarkers(markers);
  }, [bars, markers]);

  const fetchData = async (overrideSymbol) => {
    const finalSymbol = String(overrideSymbol || symbol || '').trim().toUpperCase();
    if (!finalSymbol) return;
    if (finalSymbol !== symbol) {
      setSymbol(finalSymbol);
    }
    setLoading(true);
    setError(null);
    try {
      const response = await api.get(`/alpaca/intraday/?symbol=${encodeURIComponent(finalSymbol)}`);
      const payload = response.data || {};
      if (payload.error || !payload.bars || payload.bars.length === 0) {
        setBars([]);
        setMarkers([]);
        setGuidance(payload.guidance || []);
        setStats(null);
        setGemini(payload.gemini || null);
        setError(payload.error || 'Aucune donnée intraday disponible.');
        if (payload.symbol && payload.symbol !== finalSymbol) {
          setSymbol(payload.symbol);
        }
        return;
      }
      setBars(payload.bars || []);
      setGuidance(payload.guidance || []);
      setStats(payload.stats || null);
      setGemini(payload.gemini || null);
      const mapped = (payload.annotations || []).map((item) => {
        const isBull = (item.signal || 0) >= 0;
        return {
          time: item.time,
          position: isBull ? 'belowBar' : 'aboveBar',
          color: isBull ? '#38bdf8' : '#f87171',
          shape: isBull ? 'arrowUp' : 'arrowDown',
          text: item.text,
        };
      });
      setMarkers(mapped);
    } catch (err) {
      const message = err?.response?.data?.error || 'Impossible de charger les bougies Alpaca.';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const potentialProfitPct = useMemo(() => {
    if (!stats) return null;
    const entry = Number(stats.last_close || 0);
    const target = Number(stats.suggested_target || 0);
    if (!entry || !target || target <= 0) return null;
    return ((target - entry) / entry) * 100;
  }, [stats]);

  return (
    <section className="space-y-6">
      <header className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-slate-500">Intraday AI</p>
          <h2 className="text-2xl font-semibold text-white">Candles 1m + Signaux IA</h2>
        </div>
        <div className="flex flex-wrap gap-3">
          <input
            value={symbol}
            onChange={(event) => setSymbol(event.target.value.toUpperCase())}
            placeholder="Ticker (ex: ONCY)"
            className="bg-slate-900 border border-slate-700 rounded-xl px-4 py-2 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
          <button
            type="button"
            onClick={() => fetchData()}
            className="px-4 py-2 rounded-xl bg-indigo-500/90 text-white text-sm font-semibold hover:bg-indigo-500"
          >
            Charger
          </button>
        </div>
      </header>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,2fr),minmax(0,1fr)]">
        <div className="min-w-0 bg-slate-950 border border-slate-900 rounded-3xl p-4 overflow-hidden">
          <div ref={chartContainerRef} className="w-full max-w-full" />
        </div>
        <MarketScannerPanel
          onSelect={(nextSymbol) => {
            fetchData(nextSymbol);
          }}
        />
      </div>

      {loading && <p className="text-slate-400">Chargement des bougies...</p>}
      {error && <p className="text-red-400">{error}</p>}

      {stats && (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4">
            <p className="text-xs uppercase text-slate-500">Dernier prix</p>
            <p className="text-lg font-semibold">{stats.last_close?.toFixed?.(4) ?? stats.last_close}</p>
          </div>
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4">
            <p className="text-xs uppercase text-slate-500">Pattern Signal</p>
            <p className="text-lg font-semibold">{stats.pattern_signal?.toFixed?.(2) ?? stats.pattern_signal}</p>
          </div>
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4">
            <p className="text-xs uppercase text-slate-500">RVOL</p>
            <p className="text-lg font-semibold">{stats.rvol?.toFixed?.(2) ?? stats.rvol}</p>
          </div>
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4">
            <p className="text-xs uppercase text-slate-500">Volatilité</p>
            <p className="text-lg font-semibold">{stats.volatility?.toFixed?.(4) ?? stats.volatility}</p>
          </div>
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4">
            <p className="text-xs uppercase text-slate-500">Probabilité IA</p>
            <p className="text-lg font-semibold">
              {stats.probability != null ? `${(stats.probability * 100).toFixed(1)}%` : 'n/a'}
            </p>
          </div>
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4">
            <p className="text-xs uppercase text-slate-500">Stop-Loss suggéré</p>
            <p className="text-lg font-semibold">
              {stats.suggested_stop?.toFixed?.(4) ?? stats.suggested_stop}
            </p>
          </div>
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4">
            <p className="text-xs uppercase text-slate-500">Target suggérée</p>
            <p className="text-lg font-semibold">
              {stats.suggested_target?.toFixed?.(4) ?? stats.suggested_target}
            </p>
          </div>
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4">
            <p className="text-xs uppercase text-slate-500">Profit potentiel</p>
            <p className="text-lg font-semibold">
              {potentialProfitPct != null ? `${potentialProfitPct.toFixed(2)}%` : 'n/a'}
            </p>
          </div>
          {gemini && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4">
              <p className="text-xs uppercase text-slate-500">Score Gemini</p>
              <p className="text-lg font-semibold">
                {gemini.score != null ? `${Number(gemini.score).toFixed(0)}%` : 'n/a'}
              </p>
              {gemini.verdict && (
                <p className="text-xs text-slate-400 mt-1">Avis: {gemini.verdict}</p>
              )}
            </div>
          )}
        </div>
      )}

      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-2">
        <h3 className="text-sm uppercase tracking-[0.2em] text-slate-400">Guidage IA</h3>
        {guidance.length > 0 ? (
          guidance.map((item) => (
            <p key={item} className="text-slate-200 text-sm">{item}</p>
          ))
        ) : (
          <p className="text-slate-500 text-sm">Aucune guidance disponible.</p>
        )}
      </div>

      {gemini && (gemini.summary || (gemini.guidance || []).length > 0) && (
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-2">
          <h3 className="text-sm uppercase tracking-[0.2em] text-slate-400">Avis Gemini</h3>
          {gemini.summary && <p className="text-slate-200 text-sm">{gemini.summary}</p>}
          {(gemini.guidance || []).map((item, index) => (
            <p key={`${item}-${index}`} className="text-slate-400 text-sm">• {item}</p>
          ))}
        </div>
      )}
    </section>
  );
}

export default IntradayAI;
