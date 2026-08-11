import { useCallback, useEffect, useRef, useState } from 'react';
import api from '../api/api';
import { cachedGet, invalidateCache } from '../api/cachedApi';

const VERDICT_BADGES = {
  confirmed: { icon: '✅', label: 'Confirmé', className: 'bg-emerald-500/15 text-emerald-200 border-emerald-500/30' },
  uncertain: { icon: '⚠️', label: 'Incertain', className: 'bg-amber-500/15 text-amber-200 border-amber-500/30' },
};

// 'value-catalyst' retiré le 2026-08-10 : offert dans le menu mais jamais
// créé côté backend (aucun ScreenerPreset avec ce slug en base -- 404
// systématique confirmé). 'value-catalyst' n'existait que comme fixture de
// test (analysis/tests.py), pas comme vrai preset seedé en production. À
// réintroduire ici une fois qu'un vrai ScreenerPreset "Value + Catalyst"
// existe côté backend, pas avant.
const PRESETS = [
  { slug: 'undervalued-without-reason', label: 'Undervalued Without Reason' },
];

const formatNumber = (value, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toFixed(digits);
};

// Encode le point en %2E plutôt que de le laisser littéral dans l'URL du
// lien -- trouvé le 2026-08-10 : un filtre réseau/logiciel de sécurité côté
// client peut bloquer/altérer toute URL contenant ".to" (domaine national
// des Tonga, associé au piratage), même dans un simple segment de chemin
// comme "BTO.TO" qui n'a rien d'un vrai nom de domaine. encodeURIComponent
// ne suffit pas ici : le point est un caractère "unreserved" qu'il laisse
// toujours tel quel. Voir TickerDetail.js pour le décodage côté réception.
const encodeTickerForUrl = (ticker) => String(ticker || '').replace(/\./g, '%2E');

function AnalysisScreener() {
  const [selectedPreset, setSelectedPreset] = useState(PRESETS[0].slug);
  const [results, setResults] = useState([]);
  const [scanMeta, setScanMeta] = useState(null);
  const [loading, setLoading] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState(null);
  const [sortBy, setSortBy] = useState('quant_score');

  const pollRef = useRef(null);

  const fetchResults = useCallback(async (presetSlug) => {
    setLoading(true);
    setError(null);
    try {
      const data = await cachedGet(`analysis/screener/${presetSlug}/results/`, {}, 5000);
      setResults(data.results || []);
      setScanMeta({
        scannedAt: data.scanned_at,
        tickersScanned: data.tickers_scanned,
        candidatesFound: data.candidates_found,
      });
    } catch (e) {
      setError(e.message || "Échec de la récupération des résultats.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchResults(selectedPreset);
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [selectedPreset, fetchResults]);

  const handleRunScan = async () => {
    setScanning(true);
    setError(null);
    try {
      await api.post(`analysis/screener/${selectedPreset}/run/`);

      // Polling simple toutes les 5s, jusqu'à 5 minutes max — le scan
      // tourne en tâche Celery async côté backend.
      let attempts = 0;
      const maxAttempts = 60;
      pollRef.current = setInterval(async () => {
        attempts += 1;
        invalidateCache(`analysis/screener/${selectedPreset}/results/`, {});
        await fetchResults(selectedPreset);
        if (attempts >= maxAttempts && pollRef.current) {
          clearInterval(pollRef.current);
          pollRef.current = null;
          setScanning(false);
        }
      }, 5000);
    } catch (e) {
      setError(e.message || "Échec du lancement du scan.");
      setScanning(false);
    }
  };

  const handleMarkManual = async (labPositionId) => {
    if (!labPositionId) return;
    try {
      await api.post(`analysis/lab/${labPositionId}/mark-manual/`, { note: '' });
      invalidateCache(`analysis/screener/${selectedPreset}/results/`, {});
      await fetchResults(selectedPreset);
    } catch {
      setError("Impossible de marquer ce pick comme sélection manuelle.");
    }
  };

  const sortedResults = [...results].sort((a, b) => {
    if (sortBy === 'quant_score') return b.quant_score - a.quant_score;
    if (sortBy === 'ticker') return a.ticker.localeCompare(b.ticker);
    return 0;
  });

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-xl font-semibold text-white">Screener fondamental</h1>
        <div className="flex items-center gap-2">
          <select
            value={selectedPreset}
            onChange={(e) => setSelectedPreset(e.target.value)}
            disabled={scanning}
            className="bg-slate-950/60 border border-slate-800 text-slate-200 text-xs rounded-lg px-2 py-2"
          >
            {PRESETS.map((p) => (
              <option key={p.slug} value={p.slug}>
                {p.label}
              </option>
            ))}
          </select>
          <button
            onClick={handleRunScan}
            disabled={scanning}
            className="px-3 py-2 rounded-lg text-xs font-medium bg-indigo-500/15 text-indigo-200 border border-indigo-500/30 hover:bg-indigo-500/25 disabled:opacity-50"
          >
            {scanning ? 'Scan en cours…' : "Lancer l'analyse"}
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 text-rose-200 text-sm px-4 py-3">
          {error}
        </div>
      )}

      {scanMeta && (
        <div className="flex flex-wrap gap-4 text-xs text-slate-400">
          <span>{scanMeta.tickersScanned ?? 0} tickers scannés</span>
          <span>{scanMeta.candidatesFound ?? 0} candidats retenus</span>
          {scanMeta.scannedAt && (
            <span>Dernier scan : {new Date(scanMeta.scannedAt).toLocaleString('fr-CA')}</span>
          )}
        </div>
      )}

      <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
        {loading ? (
          <div className="p-6 text-sm text-slate-400">Chargement…</div>
        ) : sortedResults.length === 0 ? (
          <div className="p-6 text-sm text-slate-400">
            Aucun résultat pour ce preset. Lance un scan pour commencer.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-900 text-slate-300">
                <tr>
                  <th className="text-left px-3 py-2 cursor-pointer" onClick={() => setSortBy('ticker')}>Ticker</th>
                  <th className="text-left px-3 py-2">Verdict</th>
                  <th className="text-left px-3 py-2 cursor-pointer" onClick={() => setSortBy('quant_score')}>Score</th>
                  <th className="text-left px-3 py-2">P/E</th>
                  <th className="text-left px-3 py-2">PEG</th>
                  <th className="text-left px-3 py-2">ROE</th>
                  <th className="text-left px-3 py-2">Raisonnement Claude</th>
                  <th className="text-left px-3 py-2">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-900">
                {sortedResults.map((r) => {
                  const badge = VERDICT_BADGES[r.final_verdict] || {};
                  return (
                    <tr key={r.id} className="hover:bg-slate-900/40">
                      <td className="px-3 py-2 text-slate-100">
                        <a href={`/analysis/ticker/${encodeTickerForUrl(r.ticker)}`} className="hover:underline">{r.ticker}</a>
                      </td>
                      <td className="px-3 py-2">
                        <span className={`inline-flex items-center px-2 py-1 rounded-full border text-xs ${badge.className || 'bg-slate-700/30 text-slate-200 border-slate-600/40'}`}>
                          {badge.icon} {badge.label || r.final_verdict}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-slate-200">{formatNumber(r.quant_score)}</td>
                      <td className="px-3 py-2 text-slate-300">{formatNumber(r.quant_details?.pe_ratio)}</td>
                      <td className="px-3 py-2 text-slate-300">{formatNumber(r.quant_details?.peg_ratio)}</td>
                      <td className="px-3 py-2 text-slate-300">
                        {r.quant_details?.roe !== null && r.quant_details?.roe !== undefined ? `${formatNumber(r.quant_details.roe, 1)}%` : '—'}
                      </td>
                      <td className="px-3 py-2 text-slate-400 max-w-xl">
                        <div className="whitespace-pre-wrap break-words">{r.claude_reasoning || '—'}</div>
                      </td>
                      <td className="px-3 py-2">
                        <button
                          onClick={() => handleMarkManual(r.lab_position_id)}
                          disabled={!r.lab_position_id}
                          className="px-2 py-1 rounded-lg text-xs bg-slate-800 text-slate-200 border border-slate-700 hover:bg-slate-700 disabled:opacity-40"
                        >
                          Je crois en celle-là
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default AnalysisScreener;
