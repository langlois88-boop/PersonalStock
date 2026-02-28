import { motion } from 'framer-motion';
import { useCallback, useEffect, useMemo, useState } from 'react';
import AIRecommendations from './AIRecommendations';
import api from '../api/api';

function OptimizerPage() {
    const core12ySymbols = ['ATD', 'ATD.TO', 'RY', 'RY.TO', 'TEC', 'TEC.TO'];
    const isCore12y = (item) => core12ySymbols.includes(String(item?.ticker || '').toUpperCase());
    const normalizeScore = (value) => {
      const raw = Number(value || 0);
      if (!Number.isFinite(raw)) return 0;
      return raw <= 1 ? raw * 100 : raw;
    };
    const scoreColor = (value) => {
      const score = normalizeScore(value);
      if (score >= 70) return 'text-emerald-400';
      if (score >= 50) return 'text-sky-300';
      return 'text-rose-400';
    };
    const isEtfOversold = (item) => {
      const symbol = String(item?.ticker || '').toUpperCase();
      const isEtf = ['TEC.TO', 'VFV', 'VFV.TO'].includes(symbol);
      const rsiRaw = item?.rsi ?? item?.current_rsi ?? item?.current_rsi_value ?? item?.rsi_14;
      const rsi = Number(rsiRaw);
      if (!isEtf || !Number.isFinite(rsi)) return false;
      return rsi < 30;
    };
    const signalLabel = (item) => {
      if (isCore12y(item)) {
        return { text: 'HOLD / KEEP', className: 'bg-sky-500/20 text-sky-200 border border-sky-500/40' };
      }
      if (isEtfOversold(item)) {
        return { text: '✅ BUY (Vague 1)', className: 'bg-emerald-500/20 text-emerald-200 border border-emerald-500/40' };
      }
      const score = normalizeScore(item.ai_score ?? item.confidence);
      const volumeZ = Number(item.volume_z ?? 0);
      const unrealizedPct = Number(item.unrealized_pnl_pct ?? 0);
      const winRateRaw = Number(item.win_rate ?? 0);
      const winRate = winRateRaw <= 1 ? winRateRaw * 100 : winRateRaw;
      if (winRate && winRate < 50) {
        return { text: 'STATISTIQUEMENT FAIBLE', className: 'bg-slate-700 text-slate-200 border border-slate-500/40' };
      }
      if (volumeZ < -1 && unrealizedPct > 20) {
        return { text: 'VENDRE 50% (SÉCURISER)', className: 'bg-rose-600/30 text-rose-100 border border-rose-500/60 animate-pulse' };
      }
      if (score > 70 && volumeZ > 0) {
        return { text: 'ACHETER', className: 'bg-emerald-500/20 text-emerald-200 border border-emerald-500/40' };
      }
      if (score > 60 && volumeZ < 0) {
        return { text: 'ATTENDRE VOLUME', className: 'bg-amber-500/20 text-amber-200 border border-amber-500/40' };
      }
      if (score >= 50) {
        return { text: 'HOLD / KEEP', className: 'bg-sky-500/20 text-sky-200 border border-sky-500/40' };
      }
      return { text: 'WAITING', className: 'bg-slate-700/40 text-slate-300 border border-slate-600' };
    };
  const [actions, setActions] = useState([]);
  const [suggestions, setSuggestions] = useState([]);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState('core');
  const [loadDuration, setLoadDuration] = useState(null);
  const [lastLoadedAt, setLastLoadedAt] = useState(null);
  const [optimizerError, setOptimizerError] = useState('');
  const [fastMode, setFastMode] = useState(true);
  const [geminiReport, setGeminiReport] = useState('');
  const [geminiReview, setGeminiReview] = useState(null);
  const [hoverGemini, setHoverGemini] = useState({});
  const [hoverLoading, setHoverLoading] = useState({});
  const [hoverPlacement, setHoverPlacement] = useState({});
  const [trackForm, setTrackForm] = useState({ ticker: '', entry: '', targetPct: '0.1', stopPct: '0.05' });
  const [trackStatus, setTrackStatus] = useState('');
  const [trackedSignals, setTrackedSignals] = useState([]);
  const [trackLoading, setTrackLoading] = useState(false);

  const loadOptimizer = useCallback(async () => {
    let isMounted = true;
    setIsRefreshing(true);
    setOptimizerError('');
    const startedAt = performance.now();

    const applyPayload = (payload) => {
      if (!isMounted) return;
      setActions(Array.isArray(payload?.actions) ? payload.actions : []);
      setSuggestions(Array.isArray(payload?.suggestions) ? payload.suggestions : []);
      setGeminiReport(payload?.gemini_report || '');
      setGeminiReview(payload?.gemini_review || null);
    };

    const finalize = () => {
      if (!isMounted) return;
      const duration = (performance.now() - startedAt) / 1000;
      setLoadDuration(duration);
      setLastLoadedAt(new Date());
      setIsRefreshing(false);
    };

    try {
      const res = await api.get('optimizer/', {
        params: { fast: fastMode ? 1 : 0 },
        timeout: 60000,
      });
      const payload = res?.data || {};
      if (Array.isArray(payload.actions)) {
        applyPayload(payload);
        finalize();
        return () => {
          isMounted = false;
        };
      }
    } catch (err) {
      // fall through to direct fetch
      const isTimeout = String(err?.message || '').toLowerCase().includes('timeout')
        || err?.code === 'ECONNABORTED';
      if (isTimeout) {
        setOptimizerError('Délai dépassé. Passage en mode rapide…');
      }
    }

    try {
      const fallbackUrl = `${window.location.protocol}//${window.location.hostname}:8001/api/optimizer/`;
      const params = fastMode ? '?fast=1' : '?fast=0';
      const fallbackRes = await fetch(`${fallbackUrl}${params}`);
      const fallbackPayload = await fallbackRes.json();
      applyPayload(fallbackPayload);
    } catch (err) {
      applyPayload({ actions: [], suggestions: [] });
      setOptimizerError('Impossible de charger les données (optimizer).');
    } finally {
      finalize();
    }

    return () => {
      isMounted = false;
    };
  }, [fastMode]);
  const submitTracking = useCallback(async () => {
    const ticker = String(trackForm.ticker || '').trim().toUpperCase();
    if (!ticker) {
      setTrackStatus('Ticker requis.');
      return;
    }
    setTrackStatus('Envoi…');
    try {
      await api.post('active-signals/manual/', {
        ticker,
        entry_price: trackForm.entry ? Number(trackForm.entry) : undefined,
        target_pct: trackForm.targetPct ? Number(trackForm.targetPct) : 0.1,
        stop_pct: trackForm.stopPct ? Number(trackForm.stopPct) : 0.05,
        daytrade: true,
        note: 'Daytrade - Achat manuel (tracker UI)',
      });
      setTrackStatus(`Tracking activé pour ${ticker}.`);
      setTrackForm((prev) => ({ ...prev, ticker: '', entry: '' }));
      setTrackLoading(true);
      const res = await api.get('active-signals/', { params: { status: 'OPEN', manual: 1, daytrade: 1 } });
      setTrackedSignals(Array.isArray(res?.data?.results) ? res.data.results : []);
    } catch (err) {
      setTrackStatus('Impossible d’activer le tracking.');
    } finally {
      setTrackLoading(false);
    }
  }, [trackForm]);

  const loadTrackedSignals = useCallback(async () => {
    setTrackLoading(true);
    try {
      const res = await api.get('active-signals/', { params: { status: 'OPEN', manual: 1, daytrade: 1 } });
      setTrackedSignals(Array.isArray(res?.data?.results) ? res.data.results : []);
    } catch (err) {
      setTrackedSignals([]);
    } finally {
      setTrackLoading(false);
    }
  }, []);

  const closeTrackedSignal = useCallback(async (signal) => {
    if (!signal) return;
    try {
      await api.post('active-signals/close/', { id: signal.id, ticker: signal.ticker });
      setTrackedSignals((prev) => prev.filter((item) => item.id !== signal.id));
    } catch (err) {
      setTrackStatus('Impossible de fermer le tracking.');
    }
  }, []);

  const loadHoverGemini = useCallback(async (ticker) => {
    if (!fastMode) return;
    const symbol = String(ticker || '').trim();
    if (!symbol) return;
    if (hoverGemini[symbol] || hoverLoading[symbol]) return;
    setHoverLoading((prev) => ({ ...prev, [symbol]: true }));
    try {
      const res = await api.get('optimizer/', {
        params: { hover: 1, ticker: symbol },
        timeout: 30000,
      });
      setHoverGemini((prev) => ({ ...prev, [symbol]: res?.data || null }));
    } catch (err) {
      setHoverGemini((prev) => ({ ...prev, [symbol]: { error: true } }));
    } finally {
      setHoverLoading((prev) => ({ ...prev, [symbol]: false }));
    }
  }, [fastMode, hoverGemini, hoverLoading]);

  const tooltipClass = (ticker) =>
    hoverPlacement[ticker] === 'bottom'
      ? 'left-4 top-full mt-3'
      : 'left-4 bottom-full mb-3';

  const handleHover = useCallback((event, ticker) => {
    const symbol = String(ticker || '').trim();
    if (!symbol) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const placeBelow = rect.top < 260;
    setHoverPlacement((prev) => ({ ...prev, [symbol]: placeBelow ? 'bottom' : 'top' }));
    loadHoverGemini(symbol);
  }, [loadHoverGemini]);

  useEffect(() => {
    loadOptimizer();
  }, [loadOptimizer]);

  useEffect(() => {
    loadTrackedSignals();
  }, [loadTrackedSignals]);

  const isCoreCandidate = useCallback(
    (item) => {
      const price = Number(item.price ?? 0);
      const marketCap = Number(item.market_cap ?? 0);
      if (marketCap >= 100_000_000_000) return true;
      if (isCore12y(item)) return true;
      return price > 5 && marketCap > 2_000_000_000;
    },
    [isCore12y],
  );
  const coreActions = useMemo(() => actions.filter(isCoreCandidate), [actions, isCoreCandidate]);
  const moonshotActions = useMemo(
    () => actions.filter((item) => !isCoreCandidate(item)),
    [actions, isCoreCandidate],
  );
  const hasActions = useMemo(() => actions.length > 0, [actions.length]);
  const visibleActions = activeTab === 'core' ? coreActions : moonshotActions;

  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
      <div className="xl:col-span-2 space-y-4">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400">AI Portfolio Optimizer</p>
              <p className="text-sm text-slate-300">
                {loadDuration !== null
                  ? `Chargé en ${loadDuration.toFixed(2)}s`
                  : 'Chargement en cours…'}
                {lastLoadedAt ? ` · ${lastLoadedAt.toLocaleTimeString()}` : ''}
              </p>
              {optimizerError ? (
                <p className="text-xs text-rose-300 mt-1">{optimizerError}</p>
              ) : null}
            </div>
            <div className="flex items-center gap-2">
              <label className="flex items-center gap-2 text-xs text-slate-300">
                <input
                  type="checkbox"
                  checked={fastMode}
                  onChange={(event) => setFastMode(event.target.checked)}
                />
                Mode rapide
              </label>
              <button
                className="px-3 py-1.5 rounded-lg text-xs bg-indigo-500/20 text-indigo-200 border border-indigo-400/40"
                type="button"
                onClick={loadOptimizer}
                disabled={isRefreshing}
              >
                {isRefreshing ? 'Actualisation…' : 'Rafraîchir'}
              </button>
            </div>
          </div>
        </div>
        {!hasActions ? (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 text-slate-400">
            Aucune position détectée. Ajoute des transactions pour obtenir des recommandations.
          </div>
        ) : null}
        {!fastMode && (geminiReview || geminiReport) ? (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 text-slate-200 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Avis Gemini</p>
              {geminiReview?.score != null && (
                <span className={`text-xs font-semibold ${scoreColor(geminiReview.score)}`}>
                  Score Gemini {normalizeScore(geminiReview.score).toFixed(0)}%
                </span>
              )}
            </div>
            {(geminiReview?.summary || geminiReport) && (
              <p className="text-sm text-slate-200 whitespace-pre-line">
                {geminiReview?.summary || geminiReport}
              </p>
            )}
            {geminiReview?.exposure ? (
              <p className="text-xs text-slate-300">{geminiReview.exposure}</p>
            ) : null}
            {Array.isArray(geminiReview?.audit) && geminiReview.audit.length > 0 && (
              <div className="text-xs text-slate-300 space-y-1">
                {geminiReview.audit.map((item, index) => (
                  <p key={`${item}-${index}`}>• {item}</p>
                ))}
              </div>
            )}
            {Array.isArray(geminiReview?.advice) && geminiReview.advice.length > 0 && (
              <div className="text-xs text-slate-300 space-y-1">
                {geminiReview.advice.map((item, index) => (
                  <p key={`${item}-${index}`}>• {item}</p>
                ))}
              </div>
            )}
            {Array.isArray(geminiReview?.risks) && geminiReview.risks.length > 0 && (
              <div className="text-xs text-rose-200 space-y-1">
                {geminiReview.risks.map((item, index) => (
                  <p key={`${item}-${index}`}>⚠️ {item}</p>
                ))}
              </div>
            )}
          </div>
        ) : null}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setActiveTab('core')}
            className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-colors ${
              activeTab === 'core'
                ? 'bg-indigo-500/20 text-indigo-100 border-indigo-400/40'
                : 'bg-slate-800 text-slate-300 border-slate-700'
            }`}
          >
            Core Portfolio ({coreActions.length}) · AI
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('moonshots')}
            className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-colors ${
              activeTab === 'moonshots'
                ? 'bg-rose-500/20 text-rose-100 border-rose-400/40'
                : 'bg-slate-800 text-slate-300 border-slate-700'
            }`}
          >
            Moonshots ({moonshotActions.length}) · AI
          </button>
        </div>
        {visibleActions.map((item, idx) => (
          <motion.div
            key={item.ticker}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: idx * 0.05 }}
            className="bg-slate-900 border border-slate-800 rounded-2xl p-5 relative group/action"
            onMouseEnter={(event) => handleHover(event, item.ticker)}
          >
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <p className="text-white text-lg font-semibold">{item.ticker}</p>
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-700 text-slate-300 border border-slate-600">
                    {item.type}
                  </span>
                  {(() => {
                    const tags = [];
                    if (item.growth) tags.push({ key: 'GRW', className: 'bg-emerald-500/15 text-emerald-200 border-emerald-500/40' });
                    if (item.bluechip) tags.push({ key: 'BLU', className: 'bg-sky-500/15 text-sky-200 border-sky-500/40' });
                    if (item.speculative) tags.push({ key: 'SPC', className: 'bg-amber-500/15 text-amber-200 border-amber-500/40' });
                    if (item.zombie) tags.push({ key: 'ZOM', className: 'bg-rose-500/15 text-rose-200 border-rose-500/40' });
                    if (!tags.length) return null;
                    return tags.map((tag) => (
                      <span
                        key={`${item.ticker}-${tag.key}`}
                        className={`text-[10px] px-2 py-0.5 rounded-full border ${tag.className}`}
                      >
                        [{tag.key}]
                      </span>
                    ));
                  })()}
                </div>
                <p className="text-xs text-slate-400">{item.name}</p>
                <p className="text-xs text-slate-400">{item.reason}</p>
              </div>
              {(() => {
                const label = signalLabel(item);
                return (
                  <span className={`text-xs px-3 py-1 rounded-full ${label.className}`}>
                    {label.text}
                  </span>
                );
              })()}
            </div>
            {isEtfOversold(item) ? (
              <div className="mt-2 inline-flex items-center gap-2 text-xs px-2 py-1 rounded-full border border-emerald-500/40 bg-emerald-500/10 text-emerald-200">
                📦 ACCUMULATION PILIER
              </div>
            ) : item.speculative ? (
              <div className="mt-2 inline-flex items-center gap-2 text-xs px-2 py-1 rounded-full border border-rose-500/40 bg-rose-500/10 text-rose-200">
                ⚠️ SPÉCULATIF
              </div>
            ) : null}
            {(() => {
              const winRateRaw = Number(item.win_rate ?? 0);
              const winRate = winRateRaw <= 1 ? winRateRaw * 100 : winRateRaw;
              return winRate && winRate < 50 ? (
                <div className="mt-2 inline-flex items-center gap-2 text-xs px-2 py-1 rounded-full border border-amber-500/40 bg-amber-500/10 text-amber-200">
                  ⚠️ STATISTIQUEMENT FAIBLE
                </div>
              ) : null;
            })()}
            <div className="mt-3 grid grid-cols-2 gap-2 text-[11px] text-slate-300">
              <div>
                Prix: {item.price ?? '—'}
                {item.alerts?.some((note) => String(note).includes('Divergence')) ? (
                  <span className="ml-1 text-amber-300">⚠️</span>
                ) : null}
              </div>
              <div className={Number(item.volume_z ?? 0) < -1 ? 'rounded bg-rose-600/20 px-2 py-0.5 text-rose-100' : ''}>
                Volume Z: {item.volume_z ?? '—'}
              </div>
              <div>Sharpe: {item.sharpe ?? '—'}</div>
              <div>Win rate: {item.win_rate ?? '—'}</div>
            </div>
            {Number(item.volume_z ?? 0) < -1 ? (
              <div
                className={`mt-2 rounded-lg border px-2 py-1 text-xs ${
                  isCore12y(item)
                    ? 'border-amber-500/40 bg-amber-500/10 text-amber-200'
                    : 'border-rose-500/50 bg-rose-600/20 text-rose-100'
                }`}
              >
                {isCore12y(item)
                  ? '🌡️ SURCHAUFFE TEMPORAIRE - ATTENDRE REPLI POUR ACHETER'
                  : '⚠️ SORTIE DE CAPITAL DÉTECTÉE'}
              </div>
            ) : null}
            <div className={`absolute z-20 hidden group-hover/action:block ${tooltipClass(item.ticker)} w-[22rem] max-h-[60vh] overflow-auto bg-slate-950 border border-slate-800 rounded-xl p-4 text-sm text-slate-200 shadow-xl`}>
              <p className="text-slate-100 font-semibold text-base">{item.ticker} · Détails IA</p>
              <p className="text-slate-300 mt-1">{item.reason}</p>
              <div className="mt-3 space-y-2">
                {item.advice?.length ? (
                  <div>
                    <p className="text-[11px] uppercase tracking-widest text-slate-500">Conseils</p>
                    <ul className="mt-1 list-disc list-inside text-sm text-amber-200">
                      {item.advice.map((note) => (
                        <li key={note}>{note}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                {item.alerts?.length ? (
                  <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-2 text-amber-200 text-xs">
                    {item.alerts.map((note) => (
                      <p key={`${item.ticker}-${note}`}>{note}</p>
                    ))}
                  </div>
                ) : null}
                <div>
                  <p className="text-[11px] uppercase tracking-widest text-slate-500">Pourquoi</p>
                  <ul className="mt-1 list-disc list-inside text-sm">
                    {item.drivers?.map((note) => (
                      <li key={note}>{note}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <p className="text-[11px] uppercase tracking-widest text-slate-500">Signaux clés</p>
                  <ul className="mt-1 space-y-1">
                    {item.metrics?.map((metric) => (
                      <li key={`${item.ticker}-${metric.label}`}>
                        <span className="text-slate-400">{metric.label}:</span> {metric.value}
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <p className="text-[11px] uppercase tracking-widest text-slate-500">Risques</p>
                  <ul className="mt-1 list-disc list-inside space-y-1 text-rose-300">
                    {item.risks?.map((risk) => (
                      <li key={`${item.ticker}-${risk}`}>{risk}</li>
                    ))}
                  </ul>
                </div>
                {fastMode ? (
                  <div>
                    <p className="text-[11px] uppercase tracking-widest text-slate-500">Gemini (checkup rapide)</p>
                    {hoverLoading[item.ticker] ? (
                      <p className="mt-1 text-xs text-slate-400">Analyse rapide en cours…</p>
                    ) : hoverGemini[item.ticker]?.error ? (
                      <p className="mt-1 text-xs text-rose-300">Checkup indisponible.</p>
                    ) : hoverGemini[item.ticker] ? (
                      <div className="mt-1 text-xs text-slate-200 space-y-1">
                        {hoverGemini[item.ticker]?.category_override_label ? (
                          <p className="text-amber-200">{hoverGemini[item.ticker].category_override_label}</p>
                        ) : null}
                        {hoverGemini[item.ticker]?.gemini_confidence != null ? (
                          <p>Confiance: {Number(hoverGemini[item.ticker].gemini_confidence).toFixed(0)}%</p>
                        ) : null}
                        {hoverGemini[item.ticker]?.gemini_verdict ? (
                          <p>Verdict: {hoverGemini[item.ticker].gemini_verdict}</p>
                        ) : null}
                        {hoverGemini[item.ticker]?.gemini_note ? (
                          <p className="text-slate-300">{hoverGemini[item.ticker].gemini_note}</p>
                        ) : null}
                      </div>
                    ) : (
                      <p className="mt-1 text-xs text-slate-400">Survol pour lancer le checkup.</p>
                    )}
                  </div>
                ) : null}
              </div>
            </div>
            <div className="mt-4">
              <div className="flex justify-between text-xs text-slate-400">
                <span>Confiance IA</span>
                <span className={scoreColor(item.ai_score ?? item.confidence)}>
                  {normalizeScore(item.ai_score ?? item.confidence).toFixed(0)}%
                </span>
              </div>
              <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden mt-2">
                <div
                  className={`h-full ${scoreColor(item.ai_score ?? item.confidence).replace('text-', 'bg-')}`}
                  style={{ width: `${normalizeScore(item.ai_score ?? item.confidence)}%` }}
                ></div>
              </div>
              {!fastMode && (item.gemini_verdict || item.gemini_note || item.gemini_confidence !== null) ? (
                <div className="mt-2 text-xs text-slate-300">
                  <span className="text-slate-400">Confiance Gemini</span>
                  {item.gemini_confidence !== null && item.gemini_confidence !== undefined ? (
                    <span className="ml-2 text-emerald-300 font-semibold">{Number(item.gemini_confidence).toFixed(0)}%</span>
                  ) : null}
                  {item.gemini_verdict ? (
                    <span className="ml-2 text-indigo-300">{item.gemini_verdict}</span>
                  ) : null}
                  {item.gemini_note ? (
                    <div className="mt-1 text-slate-300">{item.gemini_note}</div>
                  ) : null}
                </div>
              ) : null}
            </div>
          </motion.div>
        ))}
        <button
          className="w-full py-3 bg-indigo-600 text-white rounded-xl disabled:opacity-60"
          type="button"
          onClick={loadOptimizer}
          disabled={isRefreshing}
        >
          {isRefreshing ? 'Actualisation…' : 'Rebalance'}
        </button>
      </div>
      <AIRecommendations
        items={suggestions}
        title="Ajouts recommandés"
        emptyMessage="Aucune suggestion pour le moment."
        onRefresh={loadOptimizer}
        isRefreshing={isRefreshing}
      />
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-3">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Suivi manuel</p>
          <p className="text-xs text-slate-500 mt-1">Déclare un achat pour recevoir les alertes de suivi.</p>
        </div>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <p className="col-span-2 text-[11px] text-slate-500">
            Exemple: Ticker BTO.TO · Prix d’entrée 8.18 · Target % 0.10 (= +10%) · Stop % 0.05 (= -5%).
          </p>
          <input
            className="col-span-2 rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2 text-slate-200"
            placeholder="Ticker (ex: BTO.TO)"
            value={trackForm.ticker}
            onChange={(e) => setTrackForm((prev) => ({ ...prev, ticker: e.target.value }))}
          />
          <input
            className="rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2 text-slate-200"
            placeholder="Prix d’entrée (ex: 8.18)"
            value={trackForm.entry}
            onChange={(e) => setTrackForm((prev) => ({ ...prev, entry: e.target.value }))}
          />
          <input
            className="rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2 text-slate-200"
            placeholder="Target % (ex: 0.10 = +10%)"
            value={trackForm.targetPct}
            onChange={(e) => setTrackForm((prev) => ({ ...prev, targetPct: e.target.value }))}
          />
          <input
            className="rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2 text-slate-200"
            placeholder="Stop % (ex: 0.05 = -5%)"
            value={trackForm.stopPct}
            onChange={(e) => setTrackForm((prev) => ({ ...prev, stopPct: e.target.value }))}
          />
          <button
            type="button"
            className="col-span-2 rounded-lg border border-indigo-400/40 bg-indigo-500/20 px-3 py-2 text-xs text-indigo-100"
            onClick={submitTracking}
          >
            Activer le tracking
          </button>
        </div>
        {trackStatus ? <p className="text-xs text-slate-400">{trackStatus}</p> : null}
        <div className="border-t border-slate-800 pt-3 space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Positions suivies</span>
            <button
              type="button"
              className="text-xs text-indigo-300 hover:text-indigo-200"
              onClick={loadTrackedSignals}
              disabled={trackLoading}
            >
              {trackLoading ? 'Chargement…' : 'Rafraîchir'}
            </button>
          </div>
          {trackedSignals.length === 0 ? (
            <p className="text-xs text-slate-500">Aucun suivi actif.</p>
          ) : (
            <div className="space-y-2">
              {trackedSignals.map((signal) => (
                <div key={signal.id} className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2 text-xs text-slate-200">
                  <div className="flex flex-col">
                    <span className="font-semibold text-white">{signal.ticker}</span>
                    <span className="text-[11px] text-slate-400">
                      Entrée {Number(signal.entry_price).toFixed(4)} · Target {Number(signal.target_price).toFixed(4)} · Stop {Number(signal.stop_loss).toFixed(4)}
                    </span>
                  </div>
                  <button
                    type="button"
                    className="text-rose-300 hover:text-rose-200"
                    onClick={() => closeTrackedSignal(signal)}
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default OptimizerPage;
