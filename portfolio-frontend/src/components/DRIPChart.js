import { useEffect, useMemo, useState } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Box,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Typography,
} from '@mui/material';
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

function DRIPChart() {
  const [portfolios, setPortfolios] = useState([]);
  const [selected, setSelected] = useState('');
  const [snapshots, setSnapshots] = useState([]);

  useEffect(() => {
    api.get('portfolios/').then((res) => {
      setPortfolios(res.data);
      if (res.data.length) {
        setSelected(res.data[0].id);
      }
    });
  }, []);

  useEffect(() => {
    if (!selected) return;
    api
      .get('drip-snapshots/', { params: { portfolio_id: selected } })
      .then((res) => setSnapshots(res.data))
      .catch(console.error);
  }, [selected]);

  const chartData = useMemo(() => {
    const ordered = [...snapshots].reverse();
    return {
      labels: ordered.map((s) => s.as_of),
      datasets: [
        {
          label: 'Monthly DRIP Capital',
          data: ordered.map((s) => s.capital),
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99, 102, 241, 0.2)',
          tension: 0.3,
        },
      ],
    };
  }, [snapshots]);

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
        <Typography variant="h5">DRIP Projection</Typography>
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

      {snapshots.length === 0 ? (
        <Typography color="text.secondary">
          No DRIP snapshots yet. The monthly Celery task will populate this.
        </Typography>
      ) : (
        <Line data={chartData} />
      )}
    </Box>
  );
}

export default DRIPChart;
