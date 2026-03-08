import { useAICenter } from '../hooks/useAICenter';
import { SkeletonCard } from './ui/Skeleton';

function AICenterPage() {
  const { data: payload, isLoading, error } = useAICenter();

  const formatChange = (value) => {
    if (typeof value !== 'number') {
      return { label: '—', className: 'text-slate-400' };
    }
    return {
      label: `${value}%`,
      className: value >= 0 ? 'text-emerald-300' : 'text-rose-300',
    };
  };

  const ai = payload?.ai_center || {};
  const indices = payload?.indices || [];
  const macro = payload?.macro || [];
  const stockPredictions = ai?.stock_predictions || [];
  const addSuggestions = ai?.add_suggestions || [];
  const swingTrades = ai?.swing_trades || [];
  const sectorFocus = ai?.sector_focus || [];
  const macroCalls = ai?.macro_calls || [];
  const risks = ai?.risks || [];
  const consensus = payload?.consensus?.results || [];
  const consensusAsOf = payload?.consensus?.as_of;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.25em] text-slate-400">AI Center</p>
          <h2 className="text-2xl font-semibold text-white">Daily Quant Briefing</h2>
        </div>
        <span className="text-xs text-slate-400">{isLoading ? 'Analyse…' : 'Live'}</span>
      </div>

      {error ? <p className="text-sm text-rose-300">Impossible de charger l'AI Center.</p> : null}

      <div className="grid gap-4 md:grid-cols-3">
        {isLoading ? (
          [...Array(3)].map((_, idx) => <SkeletonCard key={`indices-${idx}`} />)
        ) : (
          indices.map((item) => (
            <div key={item.symbol} className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
              {(() => {
                const change = formatChange(item.change_pct);
                return (
                  <>
              <p className="text-xs uppercase tracking-[0.2em] text-slate-500">{item.label}</p>
              <p className="text-lg font-semibold text-white">{item.price ?? '—'}</p>
              <p className={`text-xs ${change.className}`}>
                {change.label}
              </p>
                  </>
                );
              })()}
            </div>
          ))
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {isLoading ? (
          [...Array(3)].map((_, idx) => <SkeletonCard key={`macro-${idx}`} />)
        ) : (
          macro.map((item) => (
            <div key={item.symbol} className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
              {(() => {
                const change = formatChange(item.change_pct);
                return (
                  <>
              <p className="text-xs uppercase tracking-[0.2em] text-slate-500">{item.label}</p>
              <p className="text-lg font-semibold text-white">{item.price ?? '—'}</p>
              <p className={`text-xs ${change.className}`}>
                {change.label}
              </p>
                  </>
                );
              })()}
            </div>
          ))
        )}
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-3">
        <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Synthèse</p>
        <p className="text-sm text-slate-200">{ai?.yesterday_summary || '—'}</p>
        <p className="text-sm text-slate-200">{ai?.today_outlook || '—'}</p>
        {sectorFocus.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {sectorFocus.map((item) => (
              <span key={item} className="px-3 py-1 rounded-full text-xs bg-indigo-500/20 text-indigo-200">
                {item}
              </span>
            ))}
          </div>
        ) : null}
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-3">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Predictions sur tes actions</p>
          {stockPredictions.length === 0 ? (
            <p className="text-sm text-slate-400">Aucune prédiction disponible.</p>
          ) : (
            stockPredictions.map((item, idx) => (
              <div key={`${item.ticker}-${idx}`} className="text-sm text-slate-200">
                <p className="font-semibold">{item.ticker} — {item.bias}</p>
                <p className="text-xs text-slate-400">{item.reason}</p>
              </div>
            ))
          )}
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-3">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Suggestions d'ajout</p>
          {addSuggestions.length === 0 ? (
            <p className="text-sm text-slate-400">Aucune suggestion.</p>
          ) : (
            addSuggestions.map((item, idx) => (
              <div key={`${item.ticker}-${idx}`} className="text-sm text-slate-200">
                <p className="font-semibold">{item.ticker}</p>
                <p className="text-xs text-slate-400">{item.reason}</p>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-3">
        <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Swing Trades du jour</p>
        {swingTrades.length === 0 ? (
          <p className="text-sm text-slate-400">Aucune idée de swing trade.</p>
        ) : (
          swingTrades.map((item, idx) => (
            <div key={`${item.ticker}-${idx}`} className="text-sm text-slate-200">
              <p className="font-semibold">{item.ticker}</p>
              <p className="text-xs text-slate-400">Entrée: {item.entry} · Stop: {item.stop} · TP: {item.tp}</p>
              <p className="text-xs text-slate-400">{item.reason}</p>
            </div>
          ))
        )}
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-3">
        <div className="flex items-center justify-between gap-2">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Consensus Danas (ML + DeepSeek)</p>
          {consensusAsOf ? (
            <span className="text-[11px] text-slate-500">{new Date(consensusAsOf).toLocaleString()}</span>
          ) : null}
        </div>
        {consensus.length === 0 ? (
          <p className="text-sm text-slate-400">Aucun signal validé pour le moment.</p>
        ) : (
          <div className="space-y-3">
            {consensus.map((item, idx) => (
              <div key={`${item.symbol}-${idx}`} className="text-sm text-slate-200">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="font-semibold">{item.symbol}</p>
                  <span className="text-xs text-emerald-300">Validation {item.validation_score}/10</span>
                </div>
                <p className="text-xs text-slate-400">{item.diagnostic || 'Diagnostic: —'}</p>
                <p className="text-xs text-slate-400">Action: {item.action || '—'}</p>
                {item.news?.length ? (
                  <ul className="mt-2 list-disc pl-5 text-xs text-slate-500">
                    {item.news.map((news, newsIdx) => (
                      <li key={`${item.symbol}-news-${newsIdx}`}>{news}</li>
                    ))}
                  </ul>
                ) : null}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-2">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Macro Calls</p>
          {macroCalls.length === 0 ? (
            <p className="text-sm text-slate-400">—</p>
          ) : (
            macroCalls.map((item, idx) => (
              <p key={`${item}-${idx}`} className="text-sm text-slate-200">{item}</p>
            ))
          )}
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-2">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Risques</p>
          {risks.length === 0 ? (
            <p className="text-sm text-slate-400">—</p>
          ) : (
            risks.map((item, idx) => (
              <p key={`${item}-${idx}`} className="text-sm text-rose-300">{item}</p>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

export default AICenterPage;
