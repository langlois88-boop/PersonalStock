import { useEffect, useMemo, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
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

const TO_SUFFIX_SYMBOLS = new Set(
  (process.env.REACT_APP_FORCE_TO_SUFFIX_SYMBOLS || 'TEC')
    .split(',')
    .map((item) => item.trim().toUpperCase())
    .filter(Boolean)
);

const formatTicker = (symbol) => {
  const clean = String(symbol || '').trim().toUpperCase();
  if (!clean) return '';
  if (clean.includes('.')) return clean;
  if (TO_SUFFIX_SYMBOLS.has(clean)) return `${clean}.TO`;
  return clean;
};

function PortfolioInsights() {
  const [portfolios, setPortfolios] = useState([]);
  const [selected, setSelected] = useState('');
  const [insights, setInsights] = useState(null);

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
      .get(`portfolios/${selected}/insights/`)
      .then((res) => setInsights(res.data))
      .catch(console.error);
  }, [selected]);

  const riskRows = useMemo(() => {
    if (!insights?.risk) return [];
    return [
      { label: 'Volatility', value: insights.risk.volatility },
      { label: 'Max Drawdown', value: insights.risk.max_drawdown },
      { label: 'VaR (95%)', value: insights.risk.var_95 },
      { label: 'Beta vs SPY', value: insights.risk.beta_spy },
    ];
  }, [insights]);

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
        <Typography variant="h5">Portfolio Insights</Typography>
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

      {!insights ? (
        <Typography color="text.secondary">No insights available yet.</Typography>
      ) : (
        <Box display="grid" gap={2}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Rebalancing Suggestions
              </Typography>
              {insights.rebalancing?.length ? (
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Symbol</TableCell>
                      <TableCell align="right">Actual</TableCell>
                      <TableCell align="right">Target</TableCell>
                      <TableCell align="right">Action</TableCell>
                      <TableCell align="right">Amount</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {insights.rebalancing.map((r) => (
                      <TableRow key={r.symbol}>
                        <TableCell>
                          {(() => {
                            const displaySymbol = formatTicker(r.symbol);
                            return (
                          <Typography
                            component="a"
                            href={`https://finance.yahoo.com/quote/${displaySymbol}`}
                            target="_blank"
                            rel="noreferrer"
                            sx={{ textDecoration: 'none', color: 'inherit', fontWeight: 600 }}
                          >
                            {displaySymbol}
                          </Typography>
                            );
                          })()}
                        </TableCell>
                        <TableCell align="right">
                          {(r.actual_weight * 100).toFixed(2)}%
                        </TableCell>
                        <TableCell align="right">
                          {(r.target_weight * 100).toFixed(2)}%
                        </TableCell>
                        <TableCell align="right">{r.action}</TableCell>
                        <TableCell align="right">
                          ${Number(r.amount || 0).toLocaleString()}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : (
                <Typography color="text.secondary">
                  No rebalancing recommendations at the current threshold.
                </Typography>
              )}
            </CardContent>
          </Card>

          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Risk Metrics
              </Typography>
              <Table size="small">
                <TableBody>
                  {riskRows.map((r) => (
                    <TableRow key={r.label}>
                      <TableCell>{r.label}</TableCell>
                      <TableCell align="right">
                        {r.value === null || r.value === undefined
                          ? 'N/A'
                          : Number(r.value).toFixed(4)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </Box>
      )}
    </Box>
  );
}

export default PortfolioInsights;
