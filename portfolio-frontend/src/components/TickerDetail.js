import { useCallback, useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts';
import { cachedGet } from '../api/cachedApi';
import { InfoTooltip } from './ui/Tooltip';

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

const RATIO_LEVEL_STYLES = {
  good: 'border-emerald-500/40 bg-emerald-500/5',
  neutral: 'border-amber-500/40 bg-amber-500/5',
  bad: 'border-rose-500/40 bg-rose-500/5',
  unknown: 'border-slate-800',
};
const RATIO_VALUE_COLOR = {
  good: 'text-emerald-300',
  neutral: 'text-amber-300',
  bad: 'text-rose-300',
  unknown: 'text-slate-200',
};

// Explication + verdict par ratio, avec seuils dynamiques : réutilise
// thresholds_used renvoyé par l'API quand disponible (les vrais seuils
// utilisés par le screener pour CE ticker précis, ex. roe_min=10 pour
// BTO.TO vs 15 pour AAPL) plutôt que des bandes fixes arbitraires -- reste
// cohérent avec le verdict confirmed/rejected déjà affiché sur la page.
// Retombe sur des bandes par défaut raisonnables quand l'API ne fournit
// pas de seuil précis pour ce ratio (ex. FCF Yield).
function describeRatio(key, value, thresholds = {}) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    const na = {
      pe: "P/E non disponible -- généralement le signe de bénéfices négatifs ou trop instables pour calculer un ratio prix/bénéfices significatif.",
      peg: "PEG non disponible -- nécessite un P/E ET une prévision de croissance des bénéfices ; l'un des deux manque.",
      pb: "P/B non disponible.",
      ev_ebitda: "EV/EBITDA non disponible -- souvent le cas si l'EBITDA est négatif ou proche de zéro.",
      fcf_yield: "FCF Yield non disponible -- flux de trésorerie libre ou capitalisation manquants pour ce calcul.",
    }[key] || 'Donnée non disponible.';
    return { level: 'unknown', text: na };
  }
  const v = Number(value);
  switch (key) {
    case 'pe': {
      const level = v <= 0 ? 'bad' : v < 15 ? 'good' : v <= 30 ? 'neutral' : 'bad';
      return {
        level,
        text: `P/E (Price/Earnings) = prix de l'action ÷ bénéfice par action. Sous 15x, l'action se paie relativement peu cher par rapport à ses profits (bon signe, mais vérifie pourquoi -- ça peut aussi signaler un risque perçu par le marché). Entre 15x et 30x : valorisation dans la moyenne. Au-dessus de 30x : le marché paie cher pour chaque dollar de profit, ce qui suppose une forte croissance future pour se justifier. Ici : ${formatNumber(v)}x.`,
      };
    }
    case 'peg': {
      const max = thresholds.peg_max || 1.5;
      const level = v < 1 ? 'good' : v <= max ? 'neutral' : 'bad';
      return {
        level,
        text: `PEG = P/E ÷ taux de croissance des bénéfices attendu. Corrige le P/E pour tenir compte de la croissance : sous 1, l'action est potentiellement sous-évaluée par rapport à sa croissance. Entre 1 et ${formatNumber(max, 1)} (seuil utilisé par le screener pour ce titre) : raisonnable. Au-dessus : cher par rapport à la croissance anticipée. Ici : ${formatNumber(v)}.`,
      };
    }
    case 'pb': {
      const max = thresholds.pb_max || 3;
      const level = v < 1 ? 'good' : v <= max ? 'neutral' : 'bad';
      return {
        level,
        text: `P/B (Price/Book) = prix de l'action ÷ valeur comptable par action. Sous 1x, l'action se négocie en dessous de la valeur nette de ses actifs selon les livres comptables. Jusqu'à ${formatNumber(max, 1)}x (seuil du screener ici) : normal pour la plupart des secteurs. Au-dessus : le marché valorise l'entreprise bien au-delà de ses actifs nets (courant pour les entreprises à forte marge/actifs immatériels, à surveiller sinon). Ici : ${formatNumber(v)}x.`,
      };
    }
    case 'ev_ebitda': {
      const level = v < 10 ? 'good' : v <= 15 ? 'neutral' : 'bad';
      return {
        level,
        text: `EV/EBITDA = valeur d'entreprise (capitalisation + dette - trésorerie) ÷ EBITDA. Permet de comparer des entreprises avec des structures de dette différentes, contrairement au P/E. Sous 10x : valorisation attractive. 10x-15x : dans la norme. Au-dessus de 15x : cher par rapport aux profits d'exploitation générés. Ici : ${formatNumber(v)}x.`,
      };
    }
    case 'roe': {
      const min = thresholds.roe_min || 12;
      const level = v < 0 ? 'bad' : v < min ? 'neutral' : v < min * 2.5 ? 'good' : 'neutral';
      return {
        level,
        text: `ROE (Return on Equity) = bénéfice net ÷ capitaux propres. Mesure l'efficacité avec laquelle l'entreprise génère du profit à partir de l'argent des actionnaires. Le screener exige au moins ${formatNumber(min, 1)}% pour ce titre. Un ROE très élevé (>35-40%) peut aussi signaler un fort endettement plutôt qu'une vraie efficacité -- à croiser avec Dette/EBITDA. Ici : ${formatNumber(v, 1)}%.`,
      };
    }
    case 'debt_ebitda': {
      const max = thresholds.debt_ebitda_max || 3;
      const level = v <= max * 0.5 ? 'good' : v <= max ? 'neutral' : 'bad';
      return {
        level,
        text: `Dette/EBITDA = dette totale ÷ EBITDA -- combien d'années de profits d'exploitation seraient nécessaires pour rembourser toute la dette. Le screener exige que ce soit sous ${formatNumber(max, 1)}x pour ce titre. Plus c'est bas, moins l'entreprise est vulnérable à une hausse des taux d'intérêt ou un ralentissement. Au-dessus du seuil : endettement jugé trop lourd par le filtre. Ici : ${formatNumber(v)}x.`,
      };
    }
    case 'fcf_yield': {
      const level = v >= 5 ? 'good' : v >= 2 ? 'neutral' : 'bad';
      return {
        level,
        text: `FCF Yield = flux de trésorerie libre ÷ capitalisation boursière. Indique combien de cash réel l'entreprise génère chaque année par rapport à son prix -- contrairement au bénéfice comptable, plus difficile à manipuler. Au-dessus de 5% : généreux, l'entreprise génère beaucoup de cash par rapport à son prix. Sous 2% : faible génération de cash relative au prix payé. Ici : ${formatNumber(v)}%.`,
      };
    }
    default:
      return { level: 'unknown', text: '' };
  }
}

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
        <p className="text-white font-semibold mb-1">Ratios fondamentaux</p>
        <p className="text-xs text-slate-500 mb-3">
          <span className="inline-block w-2 h-2 rounded-full bg-emerald-400 mr-1" /> favorable
          <span className="inline-block w-2 h-2 rounded-full bg-amber-400 ml-3 mr-1" /> neutre
          <span className="inline-block w-2 h-2 rounded-full bg-rose-400 ml-3 mr-1" /> défavorable
          <span className="ml-3">Survole (i) pour l'explication</span>
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
          <Ratio label="P/E" value={formatNumber(data.pe_ratio)} verdict={describeRatio('pe', data.pe_ratio, data.thresholds_used)} />
          <Ratio label="PEG" value={formatNumber(data.peg_ratio)} verdict={describeRatio('peg', data.peg_ratio, data.thresholds_used)} />
          <Ratio label="P/B" value={formatNumber(data.pb_ratio)} verdict={describeRatio('pb', data.pb_ratio, data.thresholds_used)} />
          <Ratio label="EV/EBITDA" value={formatNumber(data.ev_ebitda)} verdict={describeRatio('ev_ebitda', data.ev_ebitda, data.thresholds_used)} />
          <Ratio
            label="ROE"
            value={data.roe !== null && data.roe !== undefined ? `${formatNumber(data.roe, 1)}%` : '—'}
            verdict={describeRatio('roe', data.roe, data.thresholds_used)}
          />
          <Ratio label="Dette/EBITDA" value={formatNumber(data.debt_to_ebitda)} verdict={describeRatio('debt_ebitda', data.debt_to_ebitda, data.thresholds_used)} />
          <Ratio
            label="FCF Yield"
            value={data.fcf_yield !== null && data.fcf_yield !== undefined ? `${formatNumber(data.fcf_yield, 2)}%` : '—'}
            verdict={describeRatio('fcf_yield', data.fcf_yield, data.thresholds_used)}
          />
          <Ratio
            label="Analystes"
            value={data.num_analysts ?? '—'}
            verdict={{ level: 'unknown', text: "Nombre d'analystes qui suivent activement ce titre. Peu de couverture (0-2) veut dire moins d'yeux sur l'entreprise et potentiellement plus d'inefficience de marché -- dans un sens comme dans l'autre." }}
          />
        </div>
      </div>
    </div>
  );
}

function Ratio({ label, value, verdict }) {
  const level = verdict?.level || 'unknown';
  return (
    <div className={`border rounded-lg px-3 py-2 ${RATIO_LEVEL_STYLES[level]}`}>
      <p className="text-xs text-slate-500 flex items-center">
        {label}
        {verdict?.text && <InfoTooltip text={verdict.text} />}
      </p>
      <p className={`font-medium ${RATIO_VALUE_COLOR[level]}`}>{value}</p>
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
