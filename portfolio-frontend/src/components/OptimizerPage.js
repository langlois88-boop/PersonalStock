import { motion } from 'framer-motion';
import AIRecommendations from './AIRecommendations';

const actions = [
  {
    ticker: 'FLT.V',
    signal: 'SELL',
    confidence: 85,
    reason: 'Dilution détectée + RSI > 70.',
    aiNotes: ['Momentum court terme faible.', 'Pression vendeuse institutionnelle.', 'Risque de volatilité élevé.'],
    details: {
      dilution: 'Émission d’actions détectée le 2025-12-18 (+12% de flottant).',
      dilutionSource: 'Filing SEDAR/SEDI: prospectus court.',
      rsi: 'RSI 14j à 73 (zone surachetée, >70).',
    },
  },
  {
    ticker: 'ATD.TO',
    signal: 'KEEP',
    confidence: 92,
    reason: 'Support macro solide + yield attractif.',
    aiNotes: ['Flux de trésorerie stable.', 'Résilience en phase macro lente.', 'Risque de drawdown modéré.'],
    details: {
      dilution: 'Aucune dilution récente détectée.',
      dilutionSource: 'Registre des émissions: N/A',
      rsi: 'RSI 14j à 54 (zone neutre).',
    },
  },
  {
    ticker: 'ADA',
    signal: 'SELL',
    confidence: 60,
    reason: 'Sentiment social en baisse.',
    aiNotes: ['Corrélation marché élevée.', 'Momentum en perte.', 'Risque news-driven.'],
    details: {
      dilution: 'Offre secondaire détectée (press release 2025-11-22).',
      dilutionSource: 'Communiqué + dépôt prospectus.',
      rsi: 'RSI 14j à 68 (proche surachat).',
    },
  },
];

function OptimizerPage() {
  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
      <div className="xl:col-span-2 space-y-4">
        {actions.map((item, idx) => (
          <motion.div
            key={item.ticker}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: idx * 0.05 }}
            className="bg-slate-900 border border-slate-800 rounded-2xl p-5 relative"
          >
            <div className="flex items-start justify-between">
              <div>
                <div className="relative inline-block group/ticker">
                  <p className="text-white text-lg font-semibold">{item.ticker}</p>
                  <div className="absolute z-20 hidden group-hover/ticker:block left-0 top-full mt-2 w-80 bg-slate-950 border border-slate-800 rounded-xl p-4 text-sm text-slate-200 shadow-xl">
                    <p className="text-slate-100 font-semibold text-base">{item.ticker} · Décision IA</p>
                    <p className="text-slate-300 mt-1">{item.reason}</p>
                    <div className="mt-3 space-y-2">
                      <div>
                        <p className="text-slate-400">Pourquoi</p>
                        <ul className="text-slate-200 text-sm list-disc list-inside">
                          {item.aiNotes.map((note) => (
                            <li key={note}>{note}</li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <p className="text-slate-400">Dilution détectée</p>
                        <p className="text-slate-200">{item.details.dilution}</p>
                        <p className="text-slate-500 text-xs">Source: {item.details.dilutionSource}</p>
                      </div>
                      <div>
                        <p className="text-slate-400">RSI</p>
                        <p className="text-slate-200">{item.details.rsi}</p>
                        <p className="text-slate-500 text-xs">Seuils: <span className="text-slate-400">&gt;70 suracheté</span> · <span className="text-slate-400">&lt;30 survendu</span></p>
                      </div>
                    </div>
                  </div>
                </div>
                <p className="text-xs text-slate-400">{item.reason}</p>
              </div>
              <span
                className={`text-xs px-3 py-1 rounded-full ${
                  item.signal === 'SELL'
                    ? 'bg-rose-500/10 text-rose-400 border border-rose-500/30'
                    : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                }`}
              >
                {item.signal}
              </span>
            </div>
            <div className="mt-4">
              <div className="flex justify-between text-xs text-slate-400">
                <span>Confiance IA</span>
                <span>{item.confidence}%</span>
              </div>
              <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden mt-2">
                <div
                  className="h-full bg-indigo-500"
                  style={{ width: `${item.confidence}%` }}
                ></div>
              </div>
            </div>
          </motion.div>
        ))}
        <button className="w-full py-3 bg-indigo-600 text-white rounded-xl">Rebalance</button>
      </div>
      <AIRecommendations />
    </div>
  );
}

export default OptimizerPage;
