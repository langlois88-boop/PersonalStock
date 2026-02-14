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

function PriceHistory() {
  const [stocks, setStocks] = useState([]);
  const [selected, setSelected] = useState('');
  const [history, setHistory] = useState([]);
  const [error, setError] = useState('');

  useEffect(() => {
    api
      .get('stocks/')
      .then((res) => {
        setStocks(res.data);
        if (res.data.length > 0) {
          setSelected(res.data[0].symbol);
        }
      })
      .catch((err) => {
        console.error(err);
        setError('Failed to load stocks.');
      });
  }, []);

  useEffect(() => {
    if (!selected) return;

    api
      .get('prices/', { params: { symbol: selected } })
      .then((res) => setHistory(res.data))
      .catch((err) => {
        console.error(err);
        setError('Failed to load price history.');
      });
  }, [selected]);

  const chartData = useMemo(() => {
    const ordered = [...history].reverse();
    const labels = ordered.map((p) => p.date);
    const values = ordered.map((p) => p.close_price);

    return {
      labels,
      datasets: [
        {
          label: `${selected} Close Price`,
          data: values,
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.2)',
          tension: 0.25,
        },
      ],
    };
  }, [history, selected]);

  return (
    <div className="portfolio-card">
      <div className="portfolio-header">
        <h2>Price History</h2>
        <select
          className="stock-select"
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
        >
          {stocks.map((s) => (
            <option key={s.id} value={s.symbol}>
              {s.symbol} — {s.name}
            </option>
          ))}
        </select>
      </div>

      {error && <p className="error">{error}</p>}
      {!error && history.length === 0 && (
        <p>No price history yet. Run the fetch prices task.</p>
      )}

      {history.length > 0 && (
        <div className="chart-wrap">
          <Line data={chartData} />
        </div>
      )}
    </div>
  );
}

export default PriceHistory;
