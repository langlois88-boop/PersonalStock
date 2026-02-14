import { useCallback, useEffect, useState } from 'react';
import {
  Box,
  Button,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Typography,
} from '@mui/material';

import api from '../api/api';

function AIScout() {
  const [stocks, setStocks] = useState([]);
  const [portfolios, setPortfolios] = useState([]);
  const [selectedPortfolio, setSelectedPortfolio] = useState('');
  const [selected, setSelected] = useState('');
  const [summary, setSummary] = useState('');
  const [batchSummaries, setBatchSummaries] = useState([]);
  const [loading, setLoading] = useState(false);
  const [batchLoading, setBatchLoading] = useState(false);

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

    api
      .get('portfolios/')
      .then((res) => {
        setPortfolios(res.data);
        if (res.data.length) {
          setSelectedPortfolio(res.data[0].id);
        }
      })
      .catch(console.error);
  }, []);

  const fetchSummary = useCallback(async () => {
    if (!selected) return;
    setLoading(true);
    try {
      const res = await api.get('ai/scout/', { params: { symbol: selected } });
      setSummary(res.data.summary || 'No summary available.');
    } catch (err) {
      console.error(err);
      setSummary('No summary available.');
    } finally {
      setLoading(false);
    }
  }, [selected]);

  const fetchBatchSummaries = useCallback(async () => {
    if (!selectedPortfolio) return;
    setBatchLoading(true);
    try {
      const res = await api.post('ai/scout/batch/', {
        portfolio_id: selectedPortfolio,
        limit: 6,
      });
      setBatchSummaries(res.data.results || []);
    } catch (err) {
      console.error(err);
      setBatchSummaries([]);
    } finally {
      setBatchLoading(false);
    }
  }, [selectedPortfolio]);

  useEffect(() => {
    if (selected) {
      fetchSummary();
    }
  }, [fetchSummary, selected]);

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" flexWrap="wrap" gap={2}>
        <Typography variant="h5">AI Scout</Typography>
        <Box display="flex" gap={2} flexWrap="wrap" alignItems="center">
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
          <Button variant="outlined" onClick={fetchSummary} disabled={loading}>
            {loading ? 'Generating…' : 'Refresh'}
          </Button>
          <FormControl size="small" sx={{ minWidth: 220 }}>
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
          <Button variant="contained" onClick={fetchBatchSummaries} disabled={batchLoading}>
            {batchLoading ? 'Generating…' : 'Generate for holdings'}
          </Button>
        </Box>
      </Box>

      <Stack spacing={1.2} mt={2}>
        {summary
          .split('\n')
          .filter(Boolean)
          .map((line, idx) => (
            <Typography key={idx} color="text.secondary">
              {line}
            </Typography>
          ))}
      </Stack>

      {batchSummaries.length > 0 && (
        <Stack spacing={2} mt={3}>
          <Typography variant="subtitle2" color="text.secondary">
            Holdings summaries
          </Typography>
          {batchSummaries.map((item, idx) => (
            <Box key={`${item.symbol}-${idx}`}>
              <Typography variant="subtitle1">{item.symbol}</Typography>
              {item.error ? (
                <Typography color="error">{item.error}</Typography>
              ) : (
                item.summary
                  .split('\n')
                  .filter(Boolean)
                  .map((line, lineIdx) => (
                    <Typography key={lineIdx} color="text.secondary">
                      {line}
                    </Typography>
                  ))
              )}
            </Box>
          ))}
        </Stack>
      )}
    </Box>
  );
}

export default AIScout;
