import { useEffect, useMemo, useState } from 'react';
import {
  Box,
  Chip,
  CircularProgress,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';

import api from '../api/api';

const formatPct = (value) => {
  if (value === null || value === undefined) return '—';
  const pct = Number(value) * 100;
  if (Number.isNaN(pct)) return '—';
  return `${pct.toFixed(2)}%`;
};

const formatNumber = (value) => {
  if (value === null || value === undefined) return '—';
  const num = Number(value);
  if (Number.isNaN(num)) return '—';
  return num.toFixed(2);
};

function SectorTrendCompare() {
  const [portfolios, setPortfolios] = useState([]);
  const [selectedPortfolio, setSelectedPortfolio] = useState('');
  const [symbols, setSymbols] = useState([]);
  const [selectedSymbol, setSelectedSymbol] = useState('');
  const [days, setDays] = useState(90);
  const [peerLimit, setPeerLimit] = useState(8);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api
      .get('portfolios/')
      .then((res) => {
        setPortfolios(res.data || []);
        if (res.data?.length) {
          setSelectedPortfolio(res.data[0].id);
        }
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (!selectedPortfolio) return;
    const portfolio = portfolios.find((p) => p.id === selectedPortfolio);
    const list = portfolio?.holdings?.map((h) => h.stock) || [];
    setSymbols(list);
    if (list.length && !list.includes(selectedSymbol)) {
      setSelectedSymbol(list[0]);
    }
  }, [portfolios, selectedPortfolio, selectedSymbol]);

  useEffect(() => {
    if (!selectedSymbol) return;
    setLoading(true);
    api
      .get('sector-compare/', {
        params: {
          symbol: selectedSymbol,
          days,
          peer_limit: peerLimit,
        },
      })
      .then((res) => setData(res.data))
      .catch((err) => {
        console.error(err);
        setData(null);
      })
      .finally(() => setLoading(false));
  }, [selectedSymbol, days, peerLimit]);

  const predictionLabel = useMemo(() => data?.prediction?.label || '—', [data]);
  const predictionProb = useMemo(() => data?.prediction?.prob_up, [data]);
  const predictionDown = useMemo(
    () => (predictionProb === null || predictionProb === undefined ? null : 1 - predictionProb),
    [predictionProb]
  );

  const chipStyle = { color: 'var(--text)', borderColor: 'var(--border)' };
  const primaryText = { color: 'var(--text)' };
  const mutedText = { color: 'var(--text-muted)' };
  const labelChipStyle = (label) => {
    const val = (label || '').toUpperCase();
    if (val === 'UP') return { color: 'var(--text)', borderColor: '#22c55e' };
    if (val === 'DOWN') return { color: 'var(--text)', borderColor: '#ef4444' };
    return { color: 'var(--text)', borderColor: '#6b7280' };
  };
  const probChipStyle = (prob) => {
    if (prob >= 0.55) return { color: 'var(--text)', borderColor: '#22c55e' };
    if (prob <= 0.45) return { color: 'var(--text)', borderColor: '#ef4444' };
    return { color: 'var(--text)', borderColor: '#6b7280' };
  };

  const reasonChipStyle = (reason) => {
    const text = (reason || '').toLowerCase();
    if (text.includes('positive') || text.includes('up')) {
      return { color: 'var(--text)', borderColor: '#22c55e' };
    }
    if (text.includes('negative') || text.includes('down') || text.includes('weakening')) {
      return { color: 'var(--text)', borderColor: '#ef4444' };
    }
    return { color: 'var(--text)', borderColor: '#6b7280' };
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Sector trend comparison
      </Typography>
      <Stack spacing={2}>
        <Box display="flex" flexWrap="wrap" gap={2} alignItems="center">
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel>Portfolio</InputLabel>
            <Select
              value={selectedPortfolio}
              label="Portfolio"
              onChange={(e) => setSelectedPortfolio(e.target.value)}
            >
              {portfolios.map((p) => (
                <MenuItem key={p.id} value={p.id}>
                  {p.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl size="small" sx={{ minWidth: 160 }}>
            <InputLabel>Symbol</InputLabel>
            <Select
              value={selectedSymbol}
              label="Symbol"
              onChange={(e) => setSelectedSymbol(e.target.value)}
            >
              {symbols.map((s) => (
                <MenuItem key={s} value={s}>
                  {s}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Days</InputLabel>
            <Select value={days} label="Days" onChange={(e) => setDays(e.target.value)}>
              {[60, 90, 120, 180].map((d) => (
                <MenuItem key={d} value={d}>
                  {d}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl size="small" sx={{ minWidth: 140 }}>
            <InputLabel>Peers</InputLabel>
            <Select value={peerLimit} label="Peers" onChange={(e) => setPeerLimit(e.target.value)}>
              {[6, 8, 10, 12].map((d) => (
                <MenuItem key={d} value={d}>
                  {d}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>

        {loading ? (
          <Box display="flex" justifyContent="center" py={2}>
            <CircularProgress size={22} />
          </Box>
        ) : data ? (
          <Stack spacing={2}>
            <Box display="flex" flexWrap="wrap" gap={2} alignItems="center">
              <Typography variant="subtitle1" sx={primaryText}>
                Sector: <strong>{data.sector || 'Unknown'}</strong>
              </Typography>
              <Chip
                label={`Prediction: ${predictionLabel}`}
                color={predictionLabel === 'UP' ? 'success' : predictionLabel === 'DOWN' ? 'error' : 'default'}
                variant="outlined"
                sx={labelChipStyle(predictionLabel)}
              />
              <Chip
                label={`Prob ↑ ${formatPct(predictionProb)}`}
                color={predictionProb >= 0.55 ? 'success' : predictionProb <= 0.45 ? 'error' : 'warning'}
                variant="outlined"
                sx={probChipStyle(predictionProb)}
              />
              <Chip
                label={`Prob ↓ ${formatPct(predictionDown)}`}
                color={predictionDown >= 0.55 ? 'error' : predictionDown <= 0.45 ? 'success' : 'warning'}
                variant="outlined"
                sx={probChipStyle(1 - predictionDown)}
              />
              <Chip
                label={`Sector ${data.sector_sentiment?.label || 'flat'}`}
                variant="outlined"
                sx={chipStyle}
              />
            </Box>

            <Box display="grid" gridTemplateColumns={{ xs: '1fr', md: 'repeat(3, 1fr)' }} gap={2}>
              <Box>
                <Typography variant="subtitle2" sx={primaryText}>Stock trend</Typography>
                <Typography sx={mutedText}>30d: {formatPct(data.stock?.return_30d)}</Typography>
                <Typography sx={mutedText}>90d: {formatPct(data.stock?.return_90d)}</Typography>
                <Typography sx={mutedText}>Sentiment: {formatNumber(data.stock?.sentiment)}</Typography>
              </Box>
              <Box>
                <Typography variant="subtitle2" sx={primaryText}>Sector sentiment</Typography>
                <Typography sx={mutedText}>Recent: {formatNumber(data.sector_sentiment?.recent)}</Typography>
                <Typography sx={mutedText}>Prior: {formatNumber(data.sector_sentiment?.prior)}</Typography>
                <Typography sx={mutedText}>Trend: {formatNumber(data.sector_sentiment?.trend)}</Typography>
              </Box>
              <Box>
                <Typography variant="subtitle2" sx={primaryText}>Peer summary</Typography>
                <Typography sx={mutedText}>Peers: {data.peer_summary?.count || 0}</Typography>
                <Typography sx={mutedText}>
                  Verified: {data.peer_summary?.verified_count || 0}
                </Typography>
                <Typography sx={mutedText}>
                  Median 30d: {formatPct(data.peer_summary?.median_return_30d)}
                </Typography>
              </Box>
              {data.sector_etf ? (
                <Box>
                  <Typography variant="subtitle2" sx={primaryText}>Sector ETF</Typography>
                  <Typography sx={mutedText}>Symbol: {data.sector_etf?.symbol || '—'}</Typography>
                  <Typography sx={mutedText}>30d: {formatPct(data.sector_etf?.return_30d)}</Typography>
                  <Typography sx={mutedText}>90d: {formatPct(data.sector_etf?.return_90d)}</Typography>
                  <Typography sx={mutedText}>Sentiment: {formatNumber(data.sector_etf?.sentiment)}</Typography>
                </Box>
              ) : null}
            </Box>

            {data.prediction?.reasons?.length ? (
              <Box>
                <Typography variant="subtitle2" sx={primaryText}>Reasons</Typography>
                <Stack direction="row" flexWrap="wrap" gap={1}>
                  {data.prediction.reasons.map((r) => (
                    <Chip
                      key={r}
                      label={r}
                      size="small"
                      variant="outlined"
                      sx={reasonChipStyle(r)}
                    />
                  ))}
                </Stack>
              </Box>
            ) : null}

            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Verified peer comparison
              </Typography>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Symbol</TableCell>
                    <TableCell>Verified</TableCell>
                    <TableCell align="right">30d</TableCell>
                    <TableCell align="right">90d</TableCell>
                    <TableCell align="right">Sentiment</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {data.peers?.length ? (
                    data.peers.map((p) => (
                      <TableRow key={p.symbol}>
                        <TableCell>{p.symbol}</TableCell>
                        <TableCell>{p.verified ? 'Yes' : 'No'}</TableCell>
                        <TableCell align="right">{formatPct(p.return_30d)}</TableCell>
                        <TableCell align="right">{formatPct(p.return_90d)}</TableCell>
                        <TableCell align="right">{formatNumber(p.sentiment)}</TableCell>
                      </TableRow>
                    ))
                  ) : (
                    <TableRow>
                      <TableCell colSpan={5}>
                        <Typography color="text.secondary">No peers found.</Typography>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </Box>
          </Stack>
        ) : (
          <Typography color="text.secondary">Select a symbol to compare.</Typography>
        )}
      </Stack>
    </Box>
  );
}

export default SectorTrendCompare;
