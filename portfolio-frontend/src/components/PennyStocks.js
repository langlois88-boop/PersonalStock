import { useEffect, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';

import api from '../api/api';

function PennyStocks() {
  const [signals, setSignals] = useState([]);
  const [scout, setScout] = useState([]);

  useEffect(() => {
    api.get('penny-signals/').then((res) => setSignals(res.data)).catch(console.error);
    api.get('penny-scout/', { params: { limit: 24 } }).then((res) => setScout(res.data)).catch(console.error);
  }, []);

  const scoreColor = (value) => {
    const score = Number(value || 0);
    if (score >= 75) return 'success';
    if (score >= 55) return 'warning';
    return 'error';
  };

  const sentimentColor = (label) => {
    if (label === 'Positive') return 'success';
    if (label === 'Negative') return 'error';
    return 'default';
  };

  const calcUpDown = (score, sentimentScore) => {
    const base = Math.min(0.8, Math.max(0.2, Number(score || 0) / 100));
    const tilt = sentimentScore ? Math.min(0.15, Math.max(-0.15, sentimentScore * 0.25)) : 0;
    const up = Math.min(0.9, Math.max(0.1, base + tilt));
    const down = 1 - up;
    return { up, down };
  };

  const calcProfit30d = (score) => {
    const normalized = (Number(score || 0) / 100) - 0.5;
    const expected = normalized * 0.3;
    return expected * 100;
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Penny Stock Watchlist
      </Typography>

      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Penny Stock Scout
          </Typography>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Symbol</TableCell>
                <TableCell align="right">AI Score</TableCell>
                <TableCell align="right">Pattern</TableCell>
                <TableCell align="right">Sentiment</TableCell>
                <TableCell align="right">Hype</TableCell>
                <TableCell align="right">Price</TableCell>
                <TableCell align="right">Mentions</TableCell>
                <TableCell align="right">Up %</TableCell>
                <TableCell align="right">Down %</TableCell>
                <TableCell align="right">30d Profit %</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {scout.map((item) => {
                const sentimentScore = item.latest_sentiment?.score ?? 0;
                const { up, down } = calcUpDown(item.ai_score?.score, sentimentScore);
                const profit30d = calcProfit30d(item.ai_score?.score);
                return (
                  <TableRow key={item.ticker}>
                    <TableCell>{item.ticker}</TableCell>
                    <TableCell align="right">
                      <Chip
                        size="small"
                        label={Number(item.ai_score?.score || 0).toFixed(1)}
                        color={scoreColor(item.ai_score?.score)}
                      />
                    </TableCell>
                    <TableCell align="right">{item.ai_score?.trend || 'N/A'}</TableCell>
                    <TableCell align="right">
                      <Chip
                        size="small"
                        variant="outlined"
                        color={sentimentColor(item.latest_sentiment?.label)}
                        label={item.latest_sentiment?.label || 'Neutral'}
                      />
                    </TableCell>
                    <TableCell align="right">{item.latest_sentiment?.buzz ?? 0}</TableCell>
                    <TableCell align="right">${Number(item.latest_price || 0).toFixed(4)}</TableCell>
                    <TableCell align="right">{item.latest_sentiment?.buzz ?? 0}</TableCell>
                    <TableCell align="right">{(up * 100).toFixed(1)}%</TableCell>
                    <TableCell align="right">{(down * 100).toFixed(1)}%</TableCell>
                    <TableCell align="right" style={{ color: profit30d >= 0 ? '#22c55e' : '#ef4444' }}>
                      {profit30d.toFixed(1)}%
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
          {scout.length === 0 && (
            <Typography color="text.secondary" sx={{ mt: 1 }}>
              No scout data yet. Run the nightly task.
            </Typography>
          )}
        </CardContent>
      </Card>

      <Card variant="outlined">
        <CardContent>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Symbol</TableCell>
                <TableCell align="right">Score</TableCell>
                <TableCell align="right">Pattern</TableCell>
                <TableCell align="right">Sentiment</TableCell>
                <TableCell align="right">Hype</TableCell>
                <TableCell align="right">Price</TableCell>
                <TableCell align="right">Mentions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {signals.map((s) => (
                <TableRow key={`${s.symbol}-${s.as_of}`}>
                  <TableCell>{s.symbol}</TableCell>
                  <TableCell align="right">{Number(s.combined_score).toFixed(2)}</TableCell>
                  <TableCell align="right">{Number(s.pattern_score).toFixed(2)}</TableCell>
                  <TableCell align="right">{Number(s.sentiment_score).toFixed(2)}</TableCell>
                  <TableCell align="right">{Number(s.hype_score).toFixed(2)}</TableCell>
                  <TableCell align="right">${Number(s.last_price || 0).toFixed(3)}</TableCell>
                  <TableCell align="right">{s.mentions}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          {signals.length === 0 && (
            <Typography color="text.secondary" sx={{ mt: 1 }}>
              No penny signals yet. Run the daily task or command.
            </Typography>
          )}
        </CardContent>
      </Card>
    </Box>
  );
}

export default PennyStocks;
