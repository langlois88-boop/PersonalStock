import { useEffect, useMemo, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Slider,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';
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

const MONTHS = 24;

function VisualizationIntelligence() {
  const [portfolios, setPortfolios] = useState([]);
  const [selected, setSelected] = useState('');
  const [holdings, setHoldings] = useState([]);
  const [stocks, setStocks] = useState([]);
  const [contribution, setContribution] = useState(500);
  const [reinvestRate, setReinvestRate] = useState(1);

  useEffect(() => {
    api.get('portfolios/').then((res) => {
      setPortfolios(res.data);
      if (res.data.length) {
        setSelected(res.data[0].id);
      }
    });
    api.get('stocks/').then((res) => setStocks(res.data));
  }, []);

  useEffect(() => {
    if (!selected) return;
    api
      .get('holdings/', { params: { portfolio_id: selected } })
      .then((res) => setHoldings(res.data))
      .catch(console.error);
  }, [selected]);

  const stockMap = useMemo(() => {
    const map = new Map();
    stocks.forEach((s) => map.set(s.symbol, s));
    return map;
  }, [stocks]);

  const { sectorWeights, totalValue, weightedYield } = useMemo(() => {
    const sectors = new Map();
    let total = 0;
    let totalShares = 0;
    let yieldSum = 0;

    holdings.forEach((h) => {
      const stock = stockMap.get(h.stock_symbol || h.stock);
      if (!stock) return;
      const price = Number(stock.latest_price || 0);
      const value = Number(h.shares || 0) * price;
      const shares = Number(h.shares || 0);
      total += value;
      totalShares += shares;
      yieldSum += value * Number(stock.dividend_yield || 0);

      const sector = stock.sector && stock.sector.trim() ? stock.sector : 'Unknown';
      const bucketValue = value > 0 ? value : shares;
      sectors.set(sector, (sectors.get(sector) || 0) + bucketValue);
    });

    const denom = total > 0 ? total : totalShares || 1;
    const weights = Array.from(sectors.entries()).map(([sector, value]) => ({
      sector,
      value,
      weight: denom ? value / denom : 0,
    }));

    return {
      sectorWeights: weights.sort((a, b) => b.weight - a.weight),
      totalValue: total,
      weightedYield: total ? yieldSum / total : 0,
    };
  }, [holdings, stockMap]);

  const projection = useMemo(() => {
    const base = totalValue || 0;
    const labels = Array.from({ length: MONTHS + 1 }, (_, i) => `M${i}`);
    const values = [base];
    const monthlyYield = (weightedYield || 0) / 12;

    for (let i = 1; i <= MONTHS; i += 1) {
      const prev = values[i - 1];
      const dividend = prev * monthlyYield * reinvestRate;
      values.push(prev + dividend + contribution);
    }

    return { labels, values };
  }, [totalValue, weightedYield, reinvestRate, contribution]);

  const chartData = useMemo(
    () => ({
      labels: projection.labels,
      datasets: [
        {
          label: 'Projected Portfolio Value',
          data: projection.values,
          borderColor: '#0ea5e9',
          backgroundColor: 'rgba(14, 165, 233, 0.2)',
          tension: 0.25,
        },
      ],
    }),
    [projection]
  );

  const getHeatColor = (weight) => {
    const intensity = Math.min(0.9, Math.max(0.1, weight * 2));
    return `rgba(59, 130, 246, ${intensity})`;
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
        <Typography variant="h5">Visualization Intelligence</Typography>
        <FormControl size="small" sx={{ minWidth: 220 }}>
          <InputLabel>Portfolio</InputLabel>
          <Select
            value={selected}
            label="Portfolio"
            onChange={(e) => setSelected(e.target.value)}
          >
            {portfolios.map((p) => (
              <MenuItem key={p.id} value={p.id}>
                {p.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>

      <Box display="grid" gap={2}>
        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Sector Weight Heatmap
            </Typography>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Sector</TableCell>
                  <TableCell align="right">Weight</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {sectorWeights.map((row) => (
                  <TableRow key={row.sector}>
                    <TableCell>{row.sector}</TableCell>
                    <TableCell
                      align="right"
                      sx={{
                        backgroundColor: getHeatColor(row.weight),
                        color: '#fff',
                      }}
                    >
                      {(row.weight * 100).toFixed(2)}%
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" gutterBottom>
              What‑If Simulation
            </Typography>
            <Typography color="text.secondary">
              Monthly contribution: ${contribution}
            </Typography>
            <Slider
              value={contribution}
              min={0}
              max={5000}
              step={50}
              onChange={(_, value) => setContribution(value)}
            />
            <Typography color="text.secondary">
              Dividend reinvest rate: {(reinvestRate * 100).toFixed(0)}%
            </Typography>
            <Slider
              value={reinvestRate}
              min={0}
              max={1}
              step={0.05}
              onChange={(_, value) => setReinvestRate(value)}
            />
            <Box mt={2}>
              <Line data={chartData} />
            </Box>
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
}

export default VisualizationIntelligence;
