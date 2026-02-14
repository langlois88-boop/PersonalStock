import { useEffect, useState } from 'react';
import {
  Box,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Typography,
} from '@mui/material';

import api from '../api/api';

function AnalystInsights() {
  const [stocks, setStocks] = useState([]);
  const [selected, setSelected] = useState('');
  const [data, setData] = useState(null);

  useEffect(() => {
    api
      .get('stocks/')
      .then((res) => {
        setStocks(res.data);
        if (res.data.length) {
          setSelected(res.data[0].symbol);
        }
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (!selected) return;
    api
      .get('analyst/', { params: { symbol: selected } })
      .then((res) => setData(res.data))
      .catch((err) => {
        console.error(err);
        setData(null);
      });
  }, [selected]);

  const rec = data?.recommendation;
  const target = data?.price_target;

  const getMeanLabel = (mean) => {
    if (mean === null || mean === undefined) return { label: 'N/A', color: '#64748b' };
    if (mean <= 1.5) return { label: 'BUY', color: '#16a34a' };
    if (mean <= 2.5) return { label: 'HOLD', color: '#f59e0b' };
    return { label: 'SELL', color: '#dc2626' };
  };

  const actionPill = (label, value) => {
    const colors = {
      'STRONG BUY': '#14532d',
      BUY: '#16a34a',
      HOLD: '#64748b',
      SELL: '#dc2626',
      'STRONG SELL': '#7f1d1d',
    };
    return (
      <Box
        key={label}
        component="span"
        sx={{
          px: 1.1,
          py: 0.35,
          borderRadius: 999,
          backgroundColor: colors[label] || '#64748b',
          color: '#fff',
          fontSize: '0.75rem',
          fontWeight: 600,
          letterSpacing: '0.04em',
          display: 'inline-flex',
          alignItems: 'center',
          gap: 0.4,
        }}
      >
        {label}: {value ?? 0}
      </Box>
    );
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
        <Typography variant="h5">Analyst insights</Typography>
        <FormControl size="small" sx={{ minWidth: 220 }}>
          <InputLabel>Stock</InputLabel>
          <Select value={selected} label="Stock" onChange={(e) => setSelected(e.target.value)}>
            {stocks.map((s) => (
              <MenuItem key={s.id} value={s.symbol}>
                {s.symbol} — {s.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>

      {!data ? (
        <Typography color="text.secondary">No analyst data available.</Typography>
      ) : (
        <Stack spacing={2}>
          <Box>
            <Typography variant="subtitle2" color="text.secondary">
              Recommendation (latest period)
            </Typography>
            {rec ? (
              rec.strong_buy != null || rec.buy != null || rec.hold != null || rec.sell != null ? (
                <Box display="flex" flexWrap="wrap" gap={1}>
                  {actionPill('STRONG BUY', rec.strong_buy)}
                  {actionPill('BUY', rec.buy)}
                  {actionPill('HOLD', rec.hold)}
                  {actionPill('SELL', rec.sell)}
                  {actionPill('STRONG SELL', rec.strong_sell)}
                </Box>
              ) : (
                <Box display="flex" alignItems="center" gap={1} flexWrap="wrap">
                  <Typography>Mean rating: {rec.mean ?? '—'}</Typography>
                  <Box
                    component="span"
                    sx={{
                      px: 1.2,
                      py: 0.3,
                      borderRadius: 999,
                      backgroundColor: getMeanLabel(rec.mean).color,
                      color: '#fff',
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      letterSpacing: '0.04em',
                    }}
                  >
                    {getMeanLabel(rec.mean).label}
                  </Box>
                  <Typography color="text.secondary">
                    Analysts: {rec.analyst_count ?? '—'}
                  </Typography>
                </Box>
              )
            ) : (
              <Typography color="text.secondary">No recommendation data.</Typography>
            )}
          </Box>

          <Box>
            <Typography variant="subtitle2" color="text.secondary">
              Target price (12m)
            </Typography>
            {target ? (
              <Typography>
                Long target (high): {target.high ?? '—'} · Mean: {target.mean ?? '—'} · Short target (low):{' '}
                {target.low ?? '—'}
              </Typography>
            ) : (
              <Typography color="text.secondary">No target price data.</Typography>
            )}
            {target?.last_updated ? (
              <Typography variant="caption" color="text.secondary">
                Last updated: {new Date(target.last_updated).toLocaleDateString()}
              </Typography>
            ) : null}
          </Box>
        </Stack>
      )}
    </Box>
  );
}

export default AnalystInsights;
