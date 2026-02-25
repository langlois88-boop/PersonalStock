import { useEffect, useState } from 'react';
import api from '../api/api';

function MarketScannerPanel({ onSelect }) {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchResults = async () => {
    setLoading(true);
    try {
      const response = await api.get('/market/scanner/');
      setResults(response.data?.results || []);
    } catch (err) {
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchResults();
  }, []);

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm uppercase tracking-[0.2em] text-slate-400">Watchlist IA</h3>
        <button
          type="button"
          onClick={fetchResults}
          className="text-xs text-indigo-300 hover:text-indigo-200"
        >
          Rafraîchir
        </button>
      </div>
      {loading && <p className="text-slate-500 text-sm">Chargement…</p>}
      {!loading && results.length === 0 && (
        <p className="text-slate-500 text-sm">Aucune cible détectée.</p>
      )}
      <div className="space-y-2">
        {results.map((item) => {
          const patternText = (item.patterns || [])
            .map((pattern) => {
              if (!pattern) return null;
              if (typeof pattern === 'string') return pattern;
              return pattern?.text || pattern?.name || pattern?.label || null;
            })
            .filter(Boolean)
            .join(', ');
          return (
          <button
            key={item.symbol}
            type="button"
            onClick={() => onSelect?.(item.symbol)}
            className="w-full text-left px-3 py-2 rounded-xl border border-slate-800 bg-slate-950/60 hover:bg-slate-900"
          >
            <div className="flex items-center justify-between">
              <span className="font-semibold text-slate-200">{item.symbol}</span>
              <span className="text-xs text-emerald-300">+{item.change_pct}%</span>
            </div>
            <div className="flex flex-wrap gap-2 text-xs text-slate-400 mt-1">
              <span>RVOL {item.rvol}</span>
              {patternText && <span>{patternText}</span>}
              <span>Score {item.score}</span>
            </div>
          </button>
          );
        })}
      </div>
    </div>
  );
}

export default MarketScannerPanel;
