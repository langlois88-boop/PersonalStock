import { useEffect, useMemo, useState } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

import api from '../api/api';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const DEFAULT_YIELD = 0.02;
const YEARS = 10;

function normalizeWeights(stocks) {
  const weights = stocks.map((s) => Number(s.target_weight || 0));
  const sum = weights.reduce((a, b) => a + b, 0);

  if (sum > 1.5) {
    return weights.map((w) => w / 100);
  }

  return weights;
}

function calcYieldFromStocks(stocks) {
  if (!stocks || stocks.length === 0) return DEFAULT_YIELD;

  const weights = normalizeWeights(stocks);
  const sumWeights = weights.reduce((a, b) => a + b, 0) || 1;

  return stocks.reduce((acc, stock, idx) => {
    const weight = weights[idx] / sumWeights;
    const dy = Number(stock.dividend_yield || 0);
    return acc + weight * dy;
  }, 0);
}

function calcYieldFromTransactions(transactions) {
  if (!transactions || transactions.length === 0) return null;

  const perSymbol = new Map();
  let totalInvested = 0;

  transactions.forEach((tx) => {
    const sign = tx.transaction_type === 'SELL' ? -1 : 1;
    const shares = Number(tx.shares || 0) * sign;
    const price = Number(tx.price_per_share || 0);
    const invested = shares * price;
    const stock = tx.stock_details || {};
    const symbol = stock.symbol || tx.stock;
    const dividendYield = Number(stock.dividend_yield || 0);

    if (!perSymbol.has(symbol)) {
      perSymbol.set(symbol, { invested: 0, dividendYield });
    }

    const entry = perSymbol.get(symbol);
    entry.invested += invested;
    if (dividendYield) {
      entry.dividendYield = dividendYield;
    }

    totalInvested += invested;
  });

  if (totalInvested <= 0) return null;

  let weightedYield = 0;
  perSymbol.forEach((entry) => {
    if (entry.invested > 0) {
      weightedYield += (entry.invested / totalInvested) * entry.dividendYield;
    }
  });

  return { yieldRate: weightedYield, invested: totalInvested };
}

function buildProjection(portfolio) {
  const txResult = calcYieldFromTransactions(portfolio.transactions || []);
  const capital = Number(
    txResult?.invested || portfolio.capital || 0
  );
  const yieldRate = txResult?.yieldRate ?? calcYieldFromStocks(portfolio.stocks);
  const labels = Array.from({ length: YEARS + 1 }, (_, i) => `Year ${i}`);
  const values = [capital];

  for (let i = 1; i <= YEARS; i += 1) {
    values.push(values[i - 1] * (1 + yieldRate));
  }

  return { labels, values, yieldRate };
}

function Portfolio() {
  const [portfolios, setPortfolios] = useState([]);
  const [error, setError] = useState('');

  useEffect(() => {
    api
      .get('portfolios/')
      .then((res) => setPortfolios(res.data))
      .catch((err) => {
        console.error(err);
        setError('Failed to load portfolios. Is the backend running?');
      });
  }, []);

  const projections = useMemo(
    () => portfolios.map((p) => ({ id: p.id, name: p.name, ...buildProjection(p) })),
    [portfolios]
  );

  return (
    <div className="portfolio-page">
      <h1>Portfolio Overview</h1>
      {error && <p className="error">{error}</p>}
      {portfolios.length === 0 && !error && (
        <p>No portfolios found. Create one in the backend to see data.</p>
      )}

      {portfolios.map((p) => {
        const projection = projections.find((proj) => proj.id === p.id);
        const chartData = {
          labels: projection?.labels || [],
          datasets: [
            {
              label: `DRIP Projection (${((projection?.yieldRate || 0) * 100).toFixed(2)}% yield)`,
              data: projection?.values || [],
              borderColor: '#4f46e5',
              backgroundColor: 'rgba(79, 70, 229, 0.2)',
              tension: 0.25,
            },
          ],
        };

        const options = {
          responsive: true,
          plugins: {
            legend: { position: 'top' },
            title: { display: true, text: 'DRIP Projection (10 years)' },
          },
        };

        return (
          <div className="portfolio-card" key={p.id}>
            <div className="portfolio-header">
              <h2>{p.name}</h2>
              <p>Capital: ${Number(p.capital || 0).toLocaleString()}</p>
            </div>

            <div className="chart-wrap">
              <Line data={chartData} options={options} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default Portfolio;
