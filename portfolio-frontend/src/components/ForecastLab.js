import { useEffect, useState } from 'react';
import {
  Box,
  CircularProgress,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';

import api from '../api/api';

function ForecastLab() {
  const [portfolios, setPortfolios] = useState([]);
  const [selected, setSelected] = useState('');
  const [rows, setRows] = useState([]);
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
      .get('forecast/', { params: { portfolio_id: selected, include_universe: true } })
      .then((res) => setRows(res.data.results || []))
      .catch(() => setRows([]))
      .finally(() => setLoading(false));
  }, [selected]);

  const formatPct = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`;
  const formatPrice = (value) => `$${Number(value || 0).toFixed(2)}`;
  const formatNumber = (value) => {
    if (value === null || value === undefined) return '—';
    const num = Number(value);
    if (Number.isNaN(num)) return '—';
    return num.toFixed(2);
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
        <Typography variant="h5">Forecast lab (paper)</Typography>
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

      <Typography variant="caption" color="text.secondary">
        Paper forecasts only. Targets are statistical estimates, not advice.
      </Typography>

      <Table size="small" sx={{ mt: 2 }}>
        <TableHead>
          <TableRow>
            <TableCell>Symbol</TableCell>
            <TableCell align="right">Last</TableCell>
            <TableCell align="right">7d ↑%</TableCell>
            <TableCell align="right">7d Target</TableCell>
            <TableCell align="right">30d ↑%</TableCell>
            <TableCell align="right">30d Target</TableCell>
            <TableCell align="right">90d ↑%</TableCell>
            <TableCell align="right">90d Target</TableCell>
            <TableCell align="right">30d Hit</TableCell>
            <TableCell align="right">30d MAE</TableCell>
            <TableCell align="right">Sector Trend</TableCell>
            <TableCell align="right">Peer 30d</TableCell>
            <TableCell align="right">SPY/QQQ 30d</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {loading ? (
            <TableRow>
              <TableCell colSpan={13} align="center">
                <CircularProgress size={22} />
              </TableCell>
            </TableRow>
          ) : rows.length === 0 ? (
            <TableRow>
              <TableCell colSpan={13}>
                <Typography color="text.secondary">No forecast data available.</Typography>
              </TableCell>
            </TableRow>
          ) : (
            rows.map((r) => (
              <TableRow key={r.symbol}>
                <TableCell>{r.symbol}</TableCell>
                <TableCell align="right">{formatPrice(r.last_price)}</TableCell>
                <TableCell align="right">{formatPct(r.forecast?.['7']?.prob_up)}</TableCell>
                <TableCell align="right">{formatPrice(r.forecast?.['7']?.target_price)}</TableCell>
                <TableCell align="right">{formatPct(r.forecast?.['30']?.prob_up)}</TableCell>
                <TableCell align="right">{formatPrice(r.forecast?.['30']?.target_price)}</TableCell>
                <TableCell align="right">{formatPct(r.forecast?.['90']?.prob_up)}</TableCell>
                <TableCell align="right">{formatPrice(r.forecast?.['90']?.target_price)}</TableCell>
                <TableCell align="right">{formatPct(r.scorecard_30d?.hit_rate)}</TableCell>
                <TableCell align="right">{formatPct(r.scorecard_30d?.mae)}</TableCell>
                <TableCell align="right">{formatNumber(r.sector_context?.sentiment_trend)}</TableCell>
                <TableCell align="right">{formatPct(r.sector_context?.peer_median_return_30d)}</TableCell>
                <TableCell align="right">
                  {`${formatPct(r.market_context?.spy_30d)} / ${formatPct(r.market_context?.qqq_30d)}`}
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </Box>
  );
}

export default ForecastLab;
