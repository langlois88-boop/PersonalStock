import { useEffect, useState } from 'react';
import {
  Box,
  CircularProgress,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Typography,
} from '@mui/material';

import api from '../api/api';

function Backtest() {
  const [portfolios, setPortfolios] = useState([]);
  const [selected, setSelected] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

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
    setLoading(true);
    api
      .get('backtest/', { params: { portfolio_id: selected, days: 180 } })
      .then((res) => setResult(res.data))
      .catch(() => setResult(null))
      .finally(() => setLoading(false));
  }, [selected]);

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
        <Typography variant="h5">Backtest (paper)</Typography>
        <FormControl size="small" sx={{ minWidth: 220 }}>
          <InputLabel>Portfolio</InputLabel>
          <Select value={selected} label="Portfolio" onChange={(e) => setSelected(e.target.value)}>
            {portfolios.map((p) => (
              <MenuItem key={p.id} value={p.id}>
                {p.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>

      {loading ? (
        <Box display="flex" justifyContent="center" py={2}>
          <CircularProgress size={22} />
        </Box>
      ) : !result ? (
        <Typography color="text.secondary">No backtest data yet.</Typography>
      ) : (
        <Stack spacing={1}>
          <Typography color="text.secondary">
            Period: {result.days} days
          </Typography>
          <Typography>Total return: {(result.total_return * 100).toFixed(2)}%</Typography>
          <Typography>Annualized return: {(result.annualized_return * 100).toFixed(2)}%</Typography>
          <Typography>Volatility: {(result.volatility * 100).toFixed(2)}%</Typography>
          <Typography>Max drawdown: {(result.max_drawdown * 100).toFixed(2)}%</Typography>
        </Stack>
      )}
    </Box>
  );
}

export default Backtest;
