import { useEffect, useState } from 'react';
import { Area, AreaChart, ResponsiveContainer } from 'recharts';
import api from '../api/api';

const mockCards = [
  { ticker: 'FLT.V', score: 82, trend: [2, 3, 4, 3, 5, 6, 7], type: 'penny', price: 0.31, sector: 'Aerospace', confidence: 74, roe: 4.2, roeMean: 6.1, roeNote: 'Lower is better for cash preservation.', currentRatio: 1.6, currentRatioMean: 1.4, dividendYield: 0.0, dividendYieldMean: 0.2, revenueGrowth: 18.4, revenueGrowthMean: 15.0, altmanZ: 1.72 },
  { ticker: 'ONCY', score: 79, trend: [1, 2, 2, 3, 4, 6, 6], type: 'penny', price: 1.06, sector: 'Biotech', confidence: 68, roe: -2.5, roeMean: 3.4, roeNote: 'Negative ROE indicates losses.', currentRatio: 2.1, currentRatioMean: 1.8, dividendYield: 0.0, dividendYieldMean: 0.1, revenueGrowth: 32.8, revenueGrowthMean: 20.0, altmanZ: 1.18 },
  { ticker: 'FF', score: 76, trend: [1, 1, 2, 3, 3, 4, 5], type: 'penny', price: 3.76, sector: 'Finance', confidence: 61, roe: 8.1, roeMean: 7.2, roeNote: 'Above average profitability.', currentRatio: 1.3, currentRatioMean: 1.4, dividendYield: 0.0, dividendYieldMean: 0.4, revenueGrowth: 14.2, revenueGrowthMean: 13.0, altmanZ: 2.04 },
  { ticker: 'ADV', score: 74, trend: [2, 2, 3, 3, 4, 4, 5], type: 'penny', price: 0.52, sector: 'Healthcare', confidence: 59, roe: 3.6, roeMean: 5.0, roeNote: 'Below peer mean.', currentRatio: 1.9, currentRatioMean: 1.6, dividendYield: 0.0, dividendYieldMean: 0.2, revenueGrowth: 21.6, revenueGrowthMean: 17.0, altmanZ: 1.66 },
  { ticker: 'ARAY', score: 73, trend: [1, 2, 2, 2, 3, 4, 4], type: 'penny', price: 2.31, sector: 'MedTech', confidence: 57, roe: 5.4, roeMean: 6.5, roeNote: 'Improving trend.', currentRatio: 1.7, currentRatioMean: 1.5, dividendYield: 0.0, dividendYieldMean: 0.2, revenueGrowth: 26.3, revenueGrowthMean: 19.0, altmanZ: 2.22 },
  { ticker: 'SNDL', score: 72, trend: [2, 2, 2, 3, 3, 4, 5], type: 'penny', price: 1.63, sector: 'Consumer', confidence: 55, roe: 1.2, roeMean: 4.0, roeNote: 'Low profitability.', currentRatio: 2.4, currentRatioMean: 1.7, dividendYield: 0.0, dividendYieldMean: 0.1, revenueGrowth: 28.9, revenueGrowthMean: 18.0, altmanZ: 1.44 },
  { ticker: 'NAK', score: 70, trend: [1, 1, 2, 2, 3, 3, 4], type: 'penny', price: 0.58, sector: 'Mining', confidence: 52, roe: -1.8, roeMean: 2.9, roeNote: 'Risky earnings profile.', currentRatio: 1.2, currentRatioMean: 1.5, dividendYield: 0.0, dividendYieldMean: 0.1, revenueGrowth: 12.3, revenueGrowthMean: 14.0, altmanZ: 0.98 },
  { ticker: 'MVIS', score: 69, trend: [2, 2, 3, 3, 3, 4, 4], type: 'penny', price: 2.04, sector: 'Tech', confidence: 50, roe: 2.0, roeMean: 4.8, roeNote: 'Below sector mean.', currentRatio: 1.5, currentRatioMean: 1.6, dividendYield: 0.0, dividendYieldMean: 0.2, revenueGrowth: 16.7, revenueGrowthMean: 17.0, altmanZ: 1.52 },
  { ticker: 'ENB', score: 74, trend: [4, 4, 4, 5, 5, 6, 6], type: 'value', price: 48.12, sector: 'Energy', confidence: 71, roe: 10.4, roeMean: 9.1, roeNote: 'Stable dividend strength.', currentRatio: 0.9, currentRatioMean: 1.0, dividendYield: 7.1, dividendYieldMean: 5.3, revenueGrowth: 6.4, revenueGrowthMean: 5.2, altmanZ: 3.08 },
  { ticker: 'RY.TO', score: 71, trend: [3, 3, 4, 4, 5, 5, 6], type: 'value', price: 133.45, sector: 'Banking', confidence: 69, roe: 13.2, roeMean: 11.8, roeNote: 'Strong capital efficiency.', currentRatio: 1.1, currentRatioMean: 1.0, dividendYield: 4.2, dividendYieldMean: 4.1, revenueGrowth: 7.6, revenueGrowthMean: 6.5, altmanZ: 2.74 },
  { ticker: 'TD.TO', score: 70, trend: [3, 3, 3, 4, 4, 5, 5], type: 'value', price: 79.18, sector: 'Banking', confidence: 67, roe: 12.1, roeMean: 11.8, roeNote: 'Near peer mean.', currentRatio: 1.0, currentRatioMean: 1.0, dividendYield: 4.6, dividendYieldMean: 4.1, revenueGrowth: 6.9, revenueGrowthMean: 6.5, altmanZ: 2.61 },
  { ticker: 'BMO.TO', score: 69, trend: [3, 3, 3, 4, 4, 4, 5], type: 'value', price: 118.61, sector: 'Banking', confidence: 66, roe: 11.4, roeMean: 11.8, roeNote: 'Slightly below peers.', currentRatio: 1.0, currentRatioMean: 1.0, dividendYield: 4.9, dividendYieldMean: 4.1, revenueGrowth: 6.2, revenueGrowthMean: 6.5, altmanZ: 2.55 },
  { ticker: 'AAPL', score: 68, trend: [4, 4, 5, 5, 6, 6, 7], type: 'value', price: 189.12, sector: 'Tech', confidence: 65, roe: 28.4, roeMean: 22.0, roeNote: 'High profitability.', currentRatio: 1.1, currentRatioMean: 1.4, dividendYield: 0.6, dividendYieldMean: 1.2, revenueGrowth: 8.4, revenueGrowthMean: 7.5, altmanZ: 3.45 },
  { ticker: 'MSFT', score: 67, trend: [4, 4, 4, 5, 5, 6, 6], type: 'value', price: 412.23, sector: 'Tech', confidence: 64, roe: 24.9, roeMean: 22.0, roeNote: 'Above mean.', currentRatio: 1.9, currentRatioMean: 1.4, dividendYield: 0.8, dividendYieldMean: 1.2, revenueGrowth: 10.2, revenueGrowthMean: 7.5, altmanZ: 3.62 },
  { ticker: 'COST', score: 66, trend: [3, 3, 4, 4, 5, 5, 5], type: 'value', price: 811.02, sector: 'Retail', confidence: 62, roe: 19.6, roeMean: 16.3, roeNote: 'Consistent return profile.', currentRatio: 1.0, currentRatioMean: 1.2, dividendYield: 0.6, dividendYieldMean: 1.1, revenueGrowth: 8.1, revenueGrowthMean: 6.2, altmanZ: 3.12 },
];

const metricWeights = [
  { key: 'roe', meanKey: 'roeMean', label: 'ROE', unit: '%', stable: 30, penny: 5, reason: 'La rentabilité est clé pour les géants.' },
  { key: 'currentRatio', meanKey: 'currentRatioMean', label: 'Current Ratio', unit: 'x', stable: 10, penny: 40, reason: 'La survie est clé pour les petits stocks.' },
  { key: 'dividendYield', meanKey: 'dividendYieldMean', label: 'Dividend Yield', unit: '%', stable: 20, penny: 0, reason: 'Inutile pour les spéculatifs.' },
  { key: 'revenueGrowth', meanKey: 'revenueGrowthMean', label: 'Revenue Growth', unit: '%', stable: 10, penny: 35, reason: 'On veut voir une explosion des ventes.' },
];

const pickFallbackBase = (ticker, type) => {
  const poolType = type === 'bluechip' ? 'value' : type;
  const pool = mockCards.filter((card) => card.type === poolType);
  if (!pool.length) return mockCards[0];
  const hash = String(ticker || '')
    .split('')
    .reduce((acc, ch) => acc + ch.charCodeAt(0), 0);
  return pool[hash % pool.length];
};

const buildRoeNote = (roe, roeMean, fallback) => {
  if (!Number.isFinite(roe) || !Number.isFinite(roeMean)) return fallback || 'Données indisponibles.';
  if (roe >= roeMean * 1.1) return 'Au-dessus de la moyenne du secteur.';
  if (roe <= roeMean * 0.9) return 'En-dessous de la moyenne du secteur.';
  return 'Proche de la moyenne.';
};

const isCryptoSymbol = (symbol) => {
  if (!symbol) return false;
  const upper = String(symbol).toUpperCase();
  return upper.endsWith('-CAD') || upper.endsWith('-USD');
};

function ScoutEyePage() {
  const [tab, setTab] = useState('penny');
  const [pennyCards, setPennyCards] = useState(mockCards.filter((card) => card.type === 'penny'));
  const [stableCards, setStableCards] = useState(mockCards.filter((card) => card.type === 'value'));
  const [bluechipCards, setBluechipCards] = useState([]);

  useEffect(() => {
    let isMounted = true;

    const buildCard = (base, overrides) => ({
      ...base,
      ...overrides,
      roe: overrides.roe ?? base.roe ?? null,
      roeMean: overrides.roeMean ?? base.roeMean ?? null,
      currentRatio: overrides.currentRatio ?? base.currentRatio ?? null,
      currentRatioMean: overrides.currentRatioMean ?? base.currentRatioMean ?? null,
      dividendYield: overrides.dividendYield ?? base.dividendYield ?? null,
      dividendYieldMean: overrides.dividendYieldMean ?? base.dividendYieldMean ?? null,
      revenueGrowth: overrides.revenueGrowth ?? base.revenueGrowth ?? null,
      revenueGrowthMean: overrides.revenueGrowthMean ?? base.revenueGrowthMean ?? null,
      roeNote: overrides.roeNote ?? base.roeNote ?? 'Données indisponibles pour le moment.',
    });

    const enrichFundamentals = async (cards) => {
      const symbols = cards.map((card) => card.ticker).filter(Boolean);
      if (!symbols.length) return cards;
      try {
        const res = await api.get('scout/fundamentals/', { params: { symbols: symbols.join(',') } });
        const map = res.data?.results || {};
        return cards.map((card) => {
          const fund = map[card.ticker];
          if (!fund) return card;
          const roe = fund.roe ?? card.roe;
          return buildCard(card, {
            sector: fund.sector || card.sector,
            price: Number(fund.price ?? card.price),
            roe: fund.roe ?? card.roe,
            currentRatio: fund.current_ratio ?? card.currentRatio,
            dividendYield: fund.dividend_yield ?? card.dividendYield,
            revenueGrowth: fund.revenue_growth ?? card.revenueGrowth,
            roeNote: buildRoeNote(roe, card.roeMean, card.roeNote),
          });
        });
      } catch (err) {
        return cards;
      }
    };

    const loadPenny = async () => {
      try {
        const res = await api.get('penny-scout/', { params: { limit: 15 } });
        if (!isMounted) return;
        const data = Array.isArray(res.data) ? res.data : [];
        const mapped = data.map((item) => {
          const base = mockCards.find((card) => card.ticker === item.ticker) || pickFallbackBase(item.ticker, 'penny');
          const aiScore = Number(
            item.ai_score?.score ??
            (item.ai_score?.confidence != null ? item.ai_score.confidence * 100 : 0)
          );
          return buildCard(base, {
            ticker: item.ticker,
            type: 'penny',
            sector: item.sector || base.sector,
            price: Number(item.latest_price ?? base.price),
            score: Math.round(aiScore),
            confidence: Math.round(aiScore),
          });
        });
        const enriched = await enrichFundamentals(mapped);
        setPennyCards(enriched.sort((a, b) => (b.score || 0) - (a.score || 0)));
      } catch (err) {
        // Keep mock data on failure
      }
    };

    const loadStable = async () => {
      try {
        const res = await api.get('dashboard/portfolio/');
        if (!isMounted) return;
        const items = res.data?.holdings || res.data?.items || [];
        const stable = items.filter((item) => item.category === 'Stable');

        const mapped = await Promise.all(
          stable.slice(0, 15).map(async (item) => {
            const base = mockCards.find((card) => card.ticker === item.ticker) || pickFallbackBase(item.ticker, 'value');
            let aiScore = 50;
            if (!isCryptoSymbol(item.ticker)) {
              try {
                const pred = await api.get(`predict/stable/${item.ticker}/`);
                const predicted = Number(pred.data?.predicted_20d_return ?? 0);
                aiScore = Math.max(0, Math.min(100, Math.round(50 + predicted * 1000)));
              } catch (err) {
                aiScore = 50;
              }
            }

            return buildCard(base, {
              ticker: item.ticker,
              type: 'value',
              sector: item.name || base.sector,
              price: Number(item.price ?? base.price),
              score: Math.round(aiScore),
              confidence: Math.round(aiScore),
            });
          })
        );

        const enriched = await enrichFundamentals(mapped);
        setStableCards(enriched);
      } catch (err) {
        // Keep mock data on failure
      }
    };

    const loadBluechip = async () => {
      try {
        const res = await api.get('bluechip-hunter/', { params: { limit: 15 } });
        if (!isMounted) return;
        const data = Array.isArray(res.data) ? res.data : [];
        const fallback = mockCards
          .filter((card) => card.type === 'value')
          .map((card) => buildCard(card, { type: 'bluechip' }));

        const mapped = data.map((item) => {
          const base = mockCards.find((card) => card.ticker === item.ticker) || pickFallbackBase(item.ticker, 'bluechip');
          return buildCard(base, {
            ticker: item.ticker,
            type: 'bluechip',
            sector: item.sector || item.name || base.sector,
            price: Number(item.latest_price ?? base.price),
            dividendYield: Number(item.dividend_yield ?? base.dividendYield),
            revenueGrowth: Number(item.revenue_growth ?? base.revenueGrowth),
            score: Math.round(Number(item.ai_score ?? base.score)),
            confidence: Math.round(Number(item.ai_score ?? base.confidence)),
          });
        });
        const enriched = await enrichFundamentals(mapped.length ? mapped : fallback);
        const sortedBluechip = enriched
          .sort((a, b) => (b.score || 0) - (a.score || 0));
        setBluechipCards(sortedBluechip);
      } catch (err) {
        const fallback = mockCards
          .filter((card) => card.type === 'value')
          .map((card) => buildCard(card, { type: 'bluechip' }))
          .sort((a, b) => (b.score || 0) - (a.score || 0));
        setBluechipCards(fallback);
      }
    };

    loadPenny();
    loadStable();
    loadBluechip();

    return () => {
      isMounted = false;
    };
  }, []);

  const filteredCards = tab === 'penny'
    ? pennyCards
    : tab === 'bluechip'
      ? bluechipCards
      : stableCards;
  const columns = 3;
  const lastRowStart = Math.max(0, Math.floor((filteredCards.length - 1) / columns) * columns);

  const getAltmanLabel = (score) => {
    if (score < 1.81) return 'Risque élevé';
    if (score <= 2.99) return 'Zone grise';
    return 'Sain';
  };

  const getAltmanLevel = (score) => {
    if (score < 1.2) return { label: 'Dangereux', color: 'bg-red-500/20 text-red-300 border-red-500/30' };
    if (score < 1.81) return { label: 'Moyen', color: 'bg-orange-500/20 text-orange-300 border-orange-500/30' };
    if (score <= 2.2) return { label: 'Neutre', color: 'bg-slate-700/40 text-slate-300 border-slate-600' };
    if (score <= 2.99) return { label: 'Stable', color: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30' };
    return { label: 'Super stable', color: 'bg-teal-500/20 text-teal-300 border-teal-500/30' };
  };

  const getMetricColor = (value, mean, mode) => {
    if (!Number.isFinite(value) || !Number.isFinite(mean) || mean === 0) {
      return 'text-slate-300 bg-slate-800/60 border-slate-700';
    }

    const ratio = value / mean;
    if (ratio < 0.7) return 'text-red-300 bg-red-500/15 border-red-500/30';
    if (ratio < 0.9) return 'text-orange-300 bg-orange-500/15 border-orange-500/30';
    if (ratio < 1.05) return 'text-slate-200 bg-slate-700/40 border-slate-600';
    if (ratio < 1.25) return 'text-emerald-300 bg-emerald-500/15 border-emerald-500/30';
    return 'text-teal-300 bg-teal-500/15 border-teal-500/30';
  };

  return (
    <div className="space-y-6">
      <div className="flex gap-2 flex-wrap">
        {['penny', 'bluechip', 'value'].map((id) => (
          <button
            key={id}
            className={`px-4 py-2 rounded-full text-sm ${
              tab === id ? 'bg-indigo-500 text-white' : 'bg-slate-900 text-slate-400'
            }`}
            onClick={() => setTab(id)}
          >
            {id === 'penny' ? 'Penny Hunter' : id === 'bluechip' ? 'Bluechip Hunter' : 'Value Navigator'}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {filteredCards.map((card, index) => (
          <div key={card.ticker} className="bg-slate-900 border border-slate-800 rounded-2xl p-4 relative group">
            <div className="flex items-center justify-between">
              <a
                href={`https://finance.yahoo.com/quote/${card.ticker}`}
                target="_blank"
                rel="noreferrer"
                className="text-white font-semibold hover:text-indigo-300"
              >
                {card.ticker}
              </a>
              <span className="text-xs text-indigo-300">Score {card.score}</span>
            </div>
            <div className="mt-2 flex items-center justify-between text-xs text-slate-400">
              <span>{card.sector}</span>
              <span>${card.price.toFixed(2)}</span>
            </div>
            <div className="mt-2 flex items-center justify-between text-[11px]">
              <span className="text-amber-300">Altman Z {card.altmanZ.toFixed(2)}</span>
              <span className="text-slate-400">{getAltmanLabel(card.altmanZ)}</span>
            </div>
            <div className="mt-2">
              <span
                className={`inline-flex items-center gap-2 px-2 py-1 text-[10px] rounded-full border ${getAltmanLevel(card.altmanZ).color}`}
              >
                Indice {getAltmanLevel(card.altmanZ).label}
              </span>
            </div>
            <div className="h-20 mt-3">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={card.trend.map((value, idx) => ({ idx, value }))}>
                  <Area type="monotone" dataKey="value" stroke="#6366f1" fill="#6366f1" fillOpacity={0.2} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-3">
              <div className="flex justify-between text-[11px] text-slate-400">
                <span>Confiance IA</span>
                <span>{card.confidence}%</span>
              </div>
              <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mt-1">
                <div
                  className="h-full bg-indigo-500"
                  style={{ width: `${card.confidence}%` }}
                ></div>
              </div>
            </div>
            <div
              className={`absolute z-20 hidden group-hover:block left-4 right-4 bg-slate-950 border border-slate-800 rounded-xl p-4 text-sm text-slate-200 shadow-xl ${
                lastRowStart > 0 && index >= lastRowStart ? 'bottom-full mb-2' : 'top-full mt-1'
              }`}
            >
              <p className="text-slate-100 font-semibold text-base">{card.ticker} · Infos clés</p>
              <p className="text-slate-300 text-sm mt-1">
                {tab === 'penny'
                  ? 'Penny = petits titres, priorité à la survie (liquidité, croissance).'
                  : tab === 'bluechip'
                    ? 'Bluechip = leaders stables, priorité au rendement, croissance et solidité.'
                    : 'Stable = grosses capitalisations, priorité à la rentabilité et au rendement.'}
              </p>
              <p className="text-slate-300 mt-3">ROE (Return on Equity)</p>
              <p className="text-slate-300 mb-3">{card.roeNote} ROE: {card.roe}% · Moyenne: {card.roeMean}%</p>
              <div className="grid grid-cols-3 gap-2 text-sm text-slate-400 mb-2">
                <span className="col-span-2">Indice</span>
                <span className="text-right">{tab === 'penny' ? 'Penny' : tab === 'bluechip' ? 'Bluechip' : 'Stable'}</span>
              </div>
              <div className="space-y-2">
                {metricWeights.map((metric) => (
                  <div key={metric.key} className="grid grid-cols-3 gap-2 items-start">
                    <div className="col-span-2">
                      <p className="text-slate-100">{metric.label}</p>
                      <p className="text-slate-400 text-xs">{metric.reason}</p>
                      <div className="flex flex-wrap items-center gap-2 text-xs mt-1">
                        <span
                          className={`px-2 py-0.5 rounded-full border ${getMetricColor(
                            card[metric.key],
                            card[metric.meanKey],
                            tab
                          )}`}
                        >
                          Valeur: {card[metric.key]}{metric.unit}
                        </span>
                        <span className="text-slate-400">
                          Moyenne: {card[metric.meanKey]}{metric.unit}
                        </span>
                      </div>
                    </div>
                    <span className="text-right text-slate-200">
                      {tab === 'penny' ? metric.penny : metric.stable}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default ScoutEyePage;
