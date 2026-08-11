import { useCallback, useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts';
import { cachedGet } from '../api/cachedApi';

const VERDICT_BADGES = {
  confirmed: { icon: '✅', label: 'Confirmé', className: 'bg-emerald-500/15 text-emerald-200 border-emerald-500/30' },
  uncertain: { icon: '⚠️', label: 'Incertain', className: 'bg-amber-500/15 text-amber-200 border-amber-500/30' },
  rejected_claude: { icon: '❌', label: 'Rejeté (Claude)', className: 'bg-rose-500/15 text-rose-200 border-rose-500/30' },
  rejected_deepseek: { icon: '❌', label: 'Rejeté (DeepSeek)', className: 'bg-rose-500/15 text-rose-200 border-rose-500/30' },
};

const formatNumber = (value, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toFixed(digits);
};

const clamp = (value, min = 0, max = 100) => Math.max(min, Math.min(max, value));

// Mappe current_data sur 5 catégories (Croissance/Valeur/Revenus/Qualité/
// Momentum). Chaque score est 0-100, calculé uniquement à partir de champs
// réellement retournés par l'API -- si les champs nécessaires manquent,
// la catégorie est marquée insuffisante plutôt que d'inventer un score.
// Momentum n'a aucun champ correspondant dans current_data (pas de RSI, pas
// de variation de prix) -- toujours insuffisante avec les données actuelles,
// voir le rapport final pour cette limitation connue.
function computeRadarFactors(data) {
  if (!data) return null;
  const { pe_ratio, peg_ratio, pb_ratio, ev_ebitda, roe, debt_to_ebitda, fcf_yield, gross_margin_trend, margin_inflection_detected, thresholds_used } = data;

  // Valeur : moyenne des composantes disponibles parmi P/E, PEG, P/B, EV/EBITDA.
  const valueParts = [];
  if (pe_ratio !== null && pe_ratio !== undefined) valueParts.push(clamp(100 - (pe_ratio / 40) * 100));
  if (peg_ratio !== null && peg_ratio !== undefined) valueParts.push(clamp(100 - (peg_ratio / 2) * 100));
  if (pb_ratio !== null && pb_ratio !== undefined) valueParts.push(clamp(100 - (pb_ratio / 6) * 100));
  if (ev_ebitda !== null && ev_ebitda !== undefined) valueParts.push(clamp(100 - (ev_ebitda / 24) * 100));

  // Qualité : ROE (vs seuil du preset si dispo) + endettement (vs seuil).
  const qualityParts = [];
  if (roe !== null && roe !== undefined) {
    const roeTarget = (thresholds_used?.roe_min || 15) * 2;
    qualityParts.push(clamp((roe / roeTarget) * 100));
  }
  if (debt_to_ebitda !== null && debt_to_ebitda !== undefined) {
    const debtTarget = (thresholds_used?.debt_ebitda_max || 3) * 2;
    qualityParts.push(clamp(100 - (debt_to_ebitda / debtTarget) * 100));
  }

  // Revenus : seul FCF yield est disponible dans current_data (pas de
  // dividend yield retourné par l'API).
  const incomeParts = [];
  if (fcf_yield !== null && fcf_yield !== undefined) incomeParts.push(clamp((fcf_yield / 10) * 100));

  // Croissance : pente de la tendance de marge brute + bonus si une
  // inflexion de marge a été détectée.
  const growthParts = [];
  if (Array.isArray(gross_margin_trend) && gross_margin_trend.length >= 2) {
    const slope = gross_margin_trend[gross_margin_trend.length - 1] - gross_margin_trend[0];
    let score = clamp(50 + slope * 5);
    if (margin_inflection_detected) score = clamp(score + 15);
    growthParts.push(score);
  }

  return [
    { factor: 'Valeur', score: valueParts.length ? Math.round(valueParts.reduce((a, b) => a + b, 0) / valueParts.length) : null },
    { factor: 'Qualité', score: qualityParts.length ? Math.round(qualityParts.reduce((a, b) => a + b, 0) / qualityParts.length) : null },
    { factor: 'Revenus', score: incomeParts.length ? Math.round(incomeParts.reduce((a, b) => a + b, 0) / incomeParts.length) : null },
    { factor: 'Croissance', score: growthParts.length ? Math.round(growthParts.reduce((a, b) => a + b, 0) / growthParts.length) : null },
    { factor: 'Momentum', score: null }, // aucune donnée momentum dans current_data
  ];
}

function TickerDetail() {
  // useParams() ne décode PAS automatiquement les séquences %XX (vérifié
  // directement avec matchPath de react-router avant d'écrire ce code) --
  // rawTicker peut donc contenir "%2E" à la place d'un point littéral (voir
  // AnalysisScreener.js::encodeTickerForUrl, contournement d'un filtre
  // réseau qui bloque/altère les URL contenant ".to"). On garde la forme
  // encodée pour l'appel API (pour ne jamais envoyer ".to" en clair sur le
  // réseau non plus) et on décode uniquement pour l'affichage.
  const { ticker: rawTicker } = useParams();
  const ticker = decodeURIComponent(rawTicker || '');
  const navigate = useNavigate();
  const [payload, setPayload] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchDetail = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await cachedGet(`analysis/ticker/${rawTicker}/`, {}, 30000);
      setPayload(data);
    } catch (e) {
      setError(e.message || "Échec de la récupération de la fiche.");
    } finally {
      setLoading(false);
    }
  }, [rawTicker]);

  useEffect(() => {
    fetchDetail();
  }, [fetchDetail]);

  if (loading) {
    return (
      <div className="space-y-4">
        <BackLink navigate={navigate} />
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-sm text-slate-400">
          Chargement de la fiche {ticker}…
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-4">
        <BackLink navigate={navigate} />
        <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 text-rose-200 text-sm px-4 py-3">
          {error}
        </div>
      </div>
    );
  }

  const data = payload?.current_data;
  if (!data) {
    return (
      <div className="space-y-4">
        <BackLink navigate={navigate} />
        <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 text-rose-200 text-sm px-4 py-3">
          Aucune donnée disponible pour "{ticker}" — vérifie que le symbole est correct.
        </div>
      </div>
    );
  }

  const radarFactors = computeRadarFactors(data);
  const badge = payload.latest_verdict ? VERDICT_BADGES[payload.latest_verdict] : null;
  const scoreHistory = payload.score_history || [];
  const chartData = scoreHistory.map((h) => ({
    date: new Date(h.created_at).toLocaleDateString('fr-CA'),
    score: h.quant_score,
    verdict: h.final_verdict,
  }));

  return (
    <div className="space-y-4">
      <BackLink navigate={navigate} />

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-white flex items-center gap-2">
            {data.ticker}
            {badge && (
              <span className={`inline-flex items-center px-2 py-1 rounded-full border text-xs ${badge.className}`}>
                {badge.icon} {badge.label}
              </span>
            )}
          </h1>
          <p className="text-sm text-slate-400">{data.sector || 'Secteur inconnu'}</p>
        </div>
        <div className="text-right">
          <p className="text-2xl font-semibold text-white">{formatNumber(data.price)}$</p>
          <p className="text-xs text-slate-500">
            {data.market_cap ? `Cap. ${(data.market_cap / 1e9).toFixed(2)}G$` : ''}
          </p>
        </div>
      </div>

      {/* Bloc 1 — Radar 5 facteurs */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
        <p className="text-white font-semibold mb-4">Radar 5 facteurs</p>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={radarFactors.map((f) => ({ ...f, score: f.score ?? 0 }))}>
              <PolarGrid stroke="#1f2937" />
              <PolarAngleAxis dataKey="factor" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#1f2937" tick={false} />
              <Radar name={data.ticker} dataKey="score" stroke="#6366f1" fill="#6366f1" fillOpacity={0.35} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
        <div className="flex flex-wrap gap-3 mt-2">
          {radarFactors.map((f) => (
            <span key={f.factor} className="text-xs text-slate-400">
              {f.factor} : {f.score === null ? <em className="text-slate-600">données insuffisantes</em> : <span className="text-slate-200">{f.score}/100</span>}
            </span>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Bloc 2 — Fil d'événements */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <p className="text-white font-semibold mb-3">Fil d'événements</p>
          {scoreHistory.length === 0 ? (
            <p className="text-sm text-slate-400">Aucun historique de scan pour ce ticker.</p>
          ) : (
            <ul className="space-y-2 max-h-64 overflow-y-auto">
              {[...scoreHistory].reverse().map((h, idx) => {
                const b = VERDICT_BADGES[h.final_verdict] || {};
                return (
                  <li key={idx} className="flex items-center justify-between text-sm border-b border-slate-800 pb-2">
                    <span className="text-slate-400">{new Date(h.created_at).toLocaleString('fr-CA')}</span>
                    <span className="text-slate-300">Score {formatNumber(h.quant_score)}</span>
                    <span className={`inline-flex items-center px-2 py-0.5 rounded-full border text-xs ${b.className || 'bg-slate-700/30 text-slate-200 border-slate-600/40'}`}>
                      {b.icon} {b.label || h.final_verdict}
                    </span>
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        {/* Bloc 3 — Interprète */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <p className="text-white font-semibold mb-3">🧠 Interprète</p>
          {!payload.latest_verdict ? (
            <p className="text-sm text-slate-400">Ce ticker n'a pas encore été analysé par le screener.</p>
          ) : (
            <div className="space-y-3">
              {payload.latest_reasoning?.claude && (
                <div>
                  <p className="text-xs uppercase tracking-wide text-indigo-300 mb-1">Claude</p>
                  <p className="text-sm text-slate-300 whitespace-pre-wrap">{payload.latest_reasoning.claude}</p>
                </div>
              )}
              {payload.latest_reasoning?.deepseek && (
                <div>
                  <p className="text-xs uppercase tracking-wide text-sky-300 mb-1">DeepSeek</p>
                  <p className="text-sm text-slate-300 whitespace-pre-wrap">{payload.latest_reasoning.deepseek}</p>
                </div>
              )}
              {!payload.latest_reasoning?.claude && !payload.latest_reasoning?.deepseek && (
                <p className="text-sm text-slate-400">Aucun raisonnement disponible pour ce verdict.</p>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Bloc 4 — Note composite avec historique */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
        <p className="text-white font-semibold mb-3">Historique du score</p>
        {chartData.length === 0 ? (
          <p className="text-sm text-slate-400">Pas assez d'historique pour un graphique.</p>
        ) : (
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 24, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis stroke="#64748b" />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1f2937' }} />
                <Line type="monotone" dataKey="score" name="Score quantitatif" stroke="#6366f1" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Ratios détaillés */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
        <p className="text-white font-semibold mb-3">Ratios fondamentaux</p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
          <Ratio label="P/E" value={formatNumber(data.pe_ratio)} />
          <Ratio label="PEG" value={formatNumber(data.peg_ratio)} />
          <Ratio label="P/B" value={formatNumber(data.pb_ratio)} />
          <Ratio label="EV/EBITDA" value={formatNumber(data.ev_ebitda)} />
          <Ratio label="ROE" value={data.roe !== null && data.roe !== undefined ? `${formatNumber(data.roe, 1)}%` : '—'} />
          <Ratio label="Dette/EBITDA" value={formatNumber(data.debt_to_ebitda)} />
          <Ratio label="FCF Yield" value={data.fcf_yield !== null && data.fcf_yield !== undefined ? `${formatNumber(data.fcf_yield, 2)}%` : '—'} />
          <Ratio label="Analystes" value={data.num_analysts ?? '—'} />
        </div>
      </div>
    </div>
  );
}

function Ratio({ label, value }) {
  return (
    <div className="bg-slate-950/60 border border-slate-800 rounded-lg px-3 py-2">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="text-slate-200 font-medium">{value}</p>
    </div>
  );
}

function BackLink({ navigate }) {
  return (
    <button
      onClick={() => navigate('/analysis')}
      className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1"
    >
      ← Retour au screener
    </button>
  );
}

export default TickerDetail;
