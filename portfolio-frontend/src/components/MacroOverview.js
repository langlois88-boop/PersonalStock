import { useEffect, useState } from 'react';
import api from '../api/api';

const formatNumber = (value, digits = 2) => {
  if (value === null || value === undefined) return '—';
  const num = Number(value);
  if (Number.isNaN(num)) return '—';
  return num.toFixed(digits);
};

function MacroOverview() {
  const [latest, setLatest] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .get('macro/', { params: { limit: 1 } })
      .then((res) => {
        const rows = res.data || [];
        setLatest(rows[0] || null);
      })
      .catch(() => setLatest(null))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-white">Macro Context</h3>
          <p className="text-xs text-slate-400">Dernière mise à jour</p>
        </div>
        <span className="text-xs text-slate-400">{latest?.date || '—'}</span>
      </div>

      {loading ? (
        <p className="text-slate-400 text-sm">Chargement…</p>
      ) : (
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-3">
            <p className="text-xs text-slate-400">S&P 500 (proxy)</p>
            <p className="text-lg font-semibold text-white">{formatNumber(latest?.sp500_close, 2)}</p>
          </div>
          <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-3">
            <p className="text-xs text-slate-400">VIX</p>
            <p className="text-lg font-semibold text-white">{formatNumber(latest?.vix_index, 2)}</p>
          </div>
          <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-3">
            <p className="text-xs text-slate-400">US 10Y</p>
            <p className="text-lg font-semibold text-white">{formatNumber(latest?.interest_rate_10y, 2)}%</p>
          </div>
          <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-3">
            <p className="text-xs text-slate-400">Inflation (CPI)</p>
            <p className="text-lg font-semibold text-white">{formatNumber(latest?.inflation_rate, 2)}</p>
          </div>
          <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-3 col-span-2">
            <p className="text-xs text-slate-400">Oil (WTI)</p>
            <p className="text-lg font-semibold text-white">{formatNumber(latest?.oil_price, 2)}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default MacroOverview;
