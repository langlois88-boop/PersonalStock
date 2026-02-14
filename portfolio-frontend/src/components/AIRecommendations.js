import { AlertCircle, CheckCircle2, Minus, TrendingDown, TrendingUp } from 'lucide-react';

const recommendations = [
  {
    ticker: 'FLT.V',
    name: 'Volatus Aerospace',
    signal: 'SELL',
    confidence: 85,
    reason: 'Signal de dilution détecté + RSI en zone de surachat (72).',
    type: 'Penny',
    drivers: [
      'Émission d’actions récente → pression sur le flottant.',
      'RSI élevé → risque de correction à court terme.',
      'Volatilité supérieure à la moyenne du secteur.',
    ],
    metrics: [
      { label: 'RSI 14j', value: '72 (surachat)' },
      { label: 'Dilution', value: '+12% flottant (12/2025)' },
      { label: 'Momentum 20j', value: 'En décélération' },
    ],
    risks: ['Liquidité fragile', 'News-driven spikes'],
  },
  {
    ticker: 'ATD.TO',
    name: 'Couche-Tard',
    signal: 'KEEP',
    confidence: 94,
    reason: 'Support institutionnel fort + croissance constante du dividende.',
    type: 'Blue Chip',
    drivers: [
      'Flux de trésorerie résilient en phase macro lente.',
      'Croissance régulière du dividende.',
      'Base d’investisseurs long terme solide.',
    ],
    metrics: [
      { label: 'RSI 14j', value: '54 (neutre)' },
      { label: 'Dividende', value: '+7% YoY' },
      { label: 'Volatilité', value: 'Modérée' },
    ],
    risks: ['Ralentissement conso', 'Compression des marges'],
  },
  {
    ticker: 'ADA',
    name: 'Cardano',
    signal: 'HOLD',
    confidence: 62,
    reason: "Volume social en baisse, attend d'un retour de momentum.",
    type: 'Crypto',
    drivers: [
      'Sentiment social en retrait.',
      'Momentum court terme incertain.',
      'Sensibilité élevée au marché global crypto.',
    ],
    metrics: [
      { label: 'RSI 14j', value: '51 (neutre)' },
      { label: 'Social Volume', value: 'En baisse' },
      { label: 'Corrélation BTC', value: 'Élevée' },
    ],
    risks: ['Volatilité élevée', 'Risque régulatoire'],
  },
];

const badgeStyles = (signal) => {
  if (signal === 'SELL') {
    return 'bg-red-500/10 text-red-500 border border-red-500/20';
  }
  if (signal === 'KEEP') {
    return 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20';
  }
  return 'bg-amber-500/10 text-amber-500 border border-amber-500/20';
};

function AIRecommendations() {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <AlertCircle className="text-indigo-400" size={24} />
          AI Portfolio Optimizer
        </h2>
        <span className="text-xs text-slate-400 uppercase tracking-widest">Mise à jour : 10m ago</span>
      </div>

      <div className="space-y-4">
        {recommendations.map((item, index) => (
          <div
            key={index}
            className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4 transition-all hover:border-indigo-500/50"
          >
            <div className="flex justify-between items-start mb-3">
              <div>
                <div className="flex items-center gap-2">
                  <div className="relative group/ticker">
                    <span className="text-lg font-bold text-white cursor-help">{item.ticker}</span>
                    <div className="absolute left-0 top-full mt-2 hidden group-hover/ticker:block w-80 z-20 bg-slate-950 border border-slate-800 rounded-xl p-4 text-xs text-slate-200 shadow-xl">
                      <p className="text-sm text-slate-100 font-semibold">{item.ticker} · Détails IA</p>
                      <p className="text-slate-400 mt-1">{item.reason}</p>
                      <div className="mt-3 space-y-2">
                        <div>
                          <p className="text-[11px] uppercase tracking-widest text-slate-500">Pourquoi</p>
                          <ul className="mt-1 list-disc list-inside space-y-1">
                            {item.drivers.map((note) => (
                              <li key={note}>{note}</li>
                            ))}
                          </ul>
                        </div>
                        <div>
                          <p className="text-[11px] uppercase tracking-widest text-slate-500">Signaux clés</p>
                          <ul className="mt-1 space-y-1">
                            {item.metrics.map((metric) => (
                              <li key={`${item.ticker}-${metric.label}`}>
                                <span className="text-slate-400">{metric.label}:</span> {metric.value}
                              </li>
                            ))}
                          </ul>
                        </div>
                        <div>
                          <p className="text-[11px] uppercase tracking-widest text-slate-500">Risques</p>
                          <ul className="mt-1 list-disc list-inside space-y-1 text-rose-300">
                            {item.risks.map((risk) => (
                              <li key={`${item.ticker}-${risk}`}>{risk}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-700 text-slate-300 border border-slate-600">
                    {item.type}
                  </span>
                </div>
                <p className="text-sm text-slate-400">{item.name}</p>
              </div>

              <div className={`px-4 py-1.5 rounded-lg font-black text-xs flex items-center gap-1.5 ${badgeStyles(item.signal)}`}>
                {item.signal === 'SELL' && <TrendingDown size={14} />}
                {item.signal === 'KEEP' && <TrendingUp size={14} />}
                {item.signal === 'HOLD' && <Minus size={14} />}
                {item.signal}
              </div>
            </div>

            <div className="mb-3">
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-slate-400">Confiance de l'IA</span>
                <span className="text-indigo-400 font-medium">{item.confidence}%</span>
              </div>
              <div className="w-full bg-slate-700 h-1.5 rounded-full overflow-hidden">
                <div
                  className="bg-indigo-500 h-full rounded-full transition-all duration-1000"
                  style={{ width: `${item.confidence}%` }}
                ></div>
              </div>
            </div>

            <p className="text-xs text-slate-300 bg-slate-900/50 p-2 rounded-md italic">
              " {item.reason} "
            </p>

            <div className="mt-3 grid gap-3">
              <div>
                <p className="text-[11px] uppercase tracking-widest text-slate-500">Pourquoi</p>
                <ul className="mt-1 text-xs text-slate-200 list-disc list-inside space-y-1">
                  {item.drivers.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
              </div>

              <div>
                <p className="text-[11px] uppercase tracking-widest text-slate-500">Signaux clés</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {item.metrics.map((metric) => (
                    <span
                      key={`${item.ticker}-${metric.label}`}
                      className="text-[11px] px-2 py-1 rounded-lg bg-slate-900/60 border border-slate-700 text-slate-200"
                    >
                      <span className="text-slate-400">{metric.label}:</span> {metric.value}
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <p className="text-[11px] uppercase tracking-widest text-slate-500">Risques</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {item.risks.map((risk) => (
                    <span
                      key={`${item.ticker}-${risk}`}
                      className="text-[11px] px-2 py-1 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300"
                    >
                      {risk}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <button className="w-full mt-6 py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-sm font-semibold transition-colors flex items-center justify-center gap-2">
        <CheckCircle2 size={18} />
        Appliquer les recommandations
      </button>
    </div>
  );
}

export default AIRecommendations;
