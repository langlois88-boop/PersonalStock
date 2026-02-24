import { useState } from 'react';
import { X, TrendingUp, ShieldAlert } from 'lucide-react';
import api from '../api/api';

const Gauge = ({ value, label, color }) => (
  <div className="flex flex-col gap-2">
    <div className="flex items-center justify-between text-xs text-slate-400">
      <span>{label}</span>
      <span className="text-slate-200 font-semibold">{value}%</span>
    </div>
    <div className="h-2 rounded-full bg-slate-800">
      <div
        className="h-2 rounded-full"
        style={{ width: `${value}%`, background: color }}
      />
    </div>
  </div>
);

function AnalysisModal({ open, loading, error, data, onClose }) {
  const [tradeStatus, setTradeStatus] = useState(null);
  const [tradeLoading, setTradeLoading] = useState(false);
  if (!open) return null;

  const confidence = data?.confidence || 0;
  const sentiment = data?.sentiment === 'Positif' ? 85 : data?.sentiment === 'Négatif' ? 25 : 55;
  const verdict = data?.summary?.includes('VENDRE') ? 'VENTE' : data?.summary?.includes('ACHETER') ? 'ACHAT' : 'ATTENTE';

  const handlePaperTrade = async () => {
    if (!data?.symbol || tradeLoading) return;
    setTradeLoading(true);
    setTradeStatus(null);
    try {
      const res = await api.post('paper-trades/manual/', {
        ticker: data.symbol,
        price: data.price,
        stop_loss: data.stop_loss,
        suggested_investment: data.suggested_investment,
        confidence: data.confidence,
        sandbox: 'WATCHLIST',
      });
      const status = res?.data?.status || 'created';
      setTradeStatus(status === 'exists' ? 'Trade déjà ouvert.' : 'Paper trade créé.');
    } catch (err) {
      setTradeStatus(err?.response?.data?.error || 'Échec du paper trade.');
    } finally {
      setTradeLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="w-full max-w-2xl rounded-3xl bg-slate-950 border border-slate-800 shadow-2xl">
        <div className="flex items-center justify-between border-b border-slate-800 px-6 py-4">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-slate-500">Quick Analysis</p>
            <h2 className="text-lg font-semibold text-white">{data?.symbol || 'Analyse IA'}</h2>
          </div>
          <button type="button" onClick={onClose} className="text-slate-400 hover:text-white">
            <X size={20} />
          </button>
        </div>

        <div className="px-6 py-5 space-y-5">
          {loading && (
            <div className="text-sm text-slate-300">Analyse en cours...</div>
          )}
          {error && (
            <div className="text-sm text-red-400">{error}</div>
          )}
          {data && !loading && !error && (
            <>
              <div className="grid gap-4 md:grid-cols-3">
                <div className="rounded-2xl bg-slate-900/70 p-4">
                  <p className="text-xs text-slate-400">Verdict</p>
                  <p className="mt-2 text-2xl font-semibold text-white">{verdict}</p>
                  <p className="text-xs text-slate-400">Prix: {data.price}</p>
                </div>
                <div className="rounded-2xl bg-slate-900/70 p-4">
                  <p className="text-xs text-slate-400">Mise suggérée</p>
                  <p className="mt-2 text-2xl font-semibold text-emerald-300">${data.suggested_investment}</p>
                  <p className="text-xs text-slate-400">Risque earnings: {data.earnings_risk}</p>
                </div>
                <div className="rounded-2xl bg-slate-900/70 p-4">
                  <p className="text-xs text-slate-400">Targets</p>
                  <p className="mt-2 text-sm text-slate-200">🎯 {data.target_price}</p>
                  <p className="text-sm text-slate-200">🛡️ {data.stop_loss}</p>
                </div>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <Gauge value={Math.min(100, Math.max(0, confidence))} label="Confidence" color="#6366f1" />
                <Gauge value={sentiment} label="Sentiment" color="#10b981" />
              </div>

              <div className="rounded-2xl bg-slate-900/70 p-4 space-y-3">
                <div className="flex items-center gap-2 text-slate-200">
                  <TrendingUp size={18} />
                  <span className="font-semibold">Synthèse de l'Analyste</span>
                </div>
                <p className="text-sm text-slate-300 whitespace-pre-line">{data.summary || 'Synthèse indisponible.'}</p>
                {data.earnings_risk === 'HAUT' && (
                  <div className="flex items-center gap-2 text-xs text-red-300">
                    <ShieldAlert size={16} />
                    ⚠️ Earnings imminents : risque de volatilité élevé.
                  </div>
                )}
                  {tradeStatus && (
                    <p className="text-xs text-slate-300">{tradeStatus}</p>
                  )}
              </div>
            </>
          )}
        </div>

        <div className="flex items-center justify-end gap-3 border-t border-slate-800 px-6 py-4">
          <button type="button" className="px-4 py-2 text-xs text-slate-400 hover:text-white" onClick={onClose}>
            Fermer
          </button>
          <button
            type="button"
            className="px-4 py-2 rounded-xl bg-emerald-500/80 text-white text-xs font-semibold hover:bg-emerald-500"
            onClick={handlePaperTrade}
            disabled={tradeLoading || !data}
          >
            {tradeLoading ? 'Exécution...' : 'Exécuter le Paper Trade'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default AnalysisModal;
