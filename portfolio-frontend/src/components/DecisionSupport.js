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
  TableSortLabel,
  Typography,
} from '@mui/material';

import api from '../api/api';

function DecisionSupport() {
  const [portfolios, setPortfolios] = useState([]);
  const [selected, setSelected] = useState('');
  const [data, setData] = useState(null);
  const [scoreOrder, setScoreOrder] = useState('desc');

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
      .get(`portfolios/${selected}/decision_support/`)
      .then((res) => setData(res.data))
      .catch(console.error);
  }, [selected]);

  const sortedRecommendations = useMemo(() => {
    if (!data?.recommendations) return [];
    const copy = [...data.recommendations];
    copy.sort((a, b) => {
      const aScore = Number(a.score || 0);
      const bScore = Number(b.score || 0);
      return scoreOrder === 'asc' ? aScore - bScore : bScore - aScore;
    });
    return copy;
  }, [data, scoreOrder]);

  const actionColor = (action) => {
    const normalized = String(action || '').toUpperCase();
    if (normalized === 'BUY') return '#16a34a';
    if (normalized === 'SELL') return '#dc2626';
    return '#64748b';
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
        <Typography variant="h5">Decision Support</Typography>
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

      {!data ? (
        <Typography color="text.secondary">No decision data yet.</Typography>
      ) : (
        <Box display="grid" gap={2}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Weekly Recommendations
              </Typography>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Symbol</TableCell>
                    <TableCell>Action</TableCell>
                    <TableCell align="right">
                      <TableSortLabel
                        active
                        direction={scoreOrder}
                        onClick={() =>
                          setScoreOrder((prev) => (prev === 'asc' ? 'desc' : 'asc'))
                        }
                      >
                        Score
                      </TableSortLabel>
                    </TableCell>
                    <TableCell>Reasons</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {sortedRecommendations.map((r) => (
                    <TableRow key={r.symbol}>
                      <TableCell>
                        <Typography
                          component="a"
                          href={`https://finance.yahoo.com/quote/${r.symbol}`}
                          target="_blank"
                          rel="noreferrer"
                          sx={{ textDecoration: 'none', color: 'inherit', fontWeight: 600 }}
                        >
                          {r.symbol}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box
                          component="span"
                          sx={{
                            color: actionColor(r.action),
                            fontWeight: 600,
                            letterSpacing: '0.02em',
                          }}
                        >
                          {r.action}
                        </Box>
                      </TableCell>
                      <TableCell align="right">{Number(r.score).toFixed(2)}</TableCell>
                      <TableCell>{r.reasons.join(', ') || '—'}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Scenario Simulation
              </Typography>
              <Typography color="text.secondary">
                Crash: {(data.scenario.crash_pct * 100).toFixed(0)}% → Value: ${Number(
                  data.scenario.post_crash_value || 0
                ).toLocaleString()}
              </Typography>
              <Typography color="text.secondary">
                Monthly dividends after crash: ${Number(
                  data.scenario.estimated_monthly_dividend || 0
                ).toLocaleString()}
              </Typography>
              <Typography color="text.secondary">
                Mitigation: {data.scenario.mitigation.join('; ')}
              </Typography>
            </CardContent>
          </Card>

          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Tax Planning (Heuristic)
              </Typography>
              <Typography color="text.secondary">
                Withdrawal order: {data.tax_plan.withdrawal_order.join(' → ')}
              </Typography>
              <Typography color="text.secondary">
                {data.tax_plan.dividend_reinvest}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {data.tax_plan.note}
              </Typography>
            </CardContent>
          </Card>
        </Box>
      )}
    </Box>
  );
}

export default DecisionSupport;
