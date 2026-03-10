import { useEffect, useMemo, useState } from 'react';
import {
  Box,
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

function StockTable() {
  const [stocks, setStocks] = useState([]);
  const [sortBy, setSortBy] = useState('symbol');
  const [portfolios, setPortfolios] = useState([]);
  const [selectedPortfolio, setSelectedPortfolio] = useState('');
  const [holdings, setHoldings] = useState([]);

  useEffect(() => {
    api.get('stocks/').then((res) => setStocks(res.data)).catch(console.error);
  }, []);

  useEffect(() => {
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

  useEffect(() => {
    if (!selectedPortfolio) return;
    api
      .get('holdings/', { params: { portfolio_id: selectedPortfolio } })
      .then((res) => setHoldings(res.data))
      .catch(console.error);
  }, [selectedPortfolio]);

  const weightMap = useMemo(() => {
    const map = new Map();
    holdings.forEach((h) => {
      map.set(h.stock_symbol, Number(h.shares || 0));
    });
    return map;
  }, [holdings]);

  const totalValue = useMemo(() => {
    return stocks.reduce((sum, s) => {
      const shares = weightMap.get(s.symbol) || 0;
      const price = Number(s.latest_price || 0);
      return sum + shares * price;
    }, 0);
  }, [stocks, weightMap]);

  const totalShares = useMemo(() => {
    return stocks.reduce((sum, s) => sum + (weightMap.get(s.symbol) || 0), 0);
  }, [stocks, weightMap]);

  const getWeightPercent = (stock) => {
    if (totalValue > 0) {
      const value = (weightMap.get(stock.symbol) || 0) * Number(stock.latest_price || 0);
      return `${((value / totalValue) * 100).toFixed(2)}%`;
    }
    if (totalShares > 0) {
      const sharePct = ((weightMap.get(stock.symbol) || 0) / totalShares) * 100;
      return `${sharePct.toFixed(2)}%`;
    }
    return '0.00%';
  };

  const sorted = useMemo(() => {
    const copy = [...stocks];
    copy.sort((a, b) => {
      if (sortBy === 'dividend') {
        return (b.dividend_yield || 0) - (a.dividend_yield || 0);
      }
      if (sortBy === 'weight') {
        const aShares = weightMap.get(a.symbol) || 0;
        const bShares = weightMap.get(b.symbol) || 0;
        const aValue = aShares * Number(a.latest_price || 0);
        const bValue = bShares * Number(b.latest_price || 0);
        if (totalValue > 0) {
          return bValue - aValue;
        }
        return bShares - aShares;
      }
      if (sortBy === 'sector') {
        return (a.sector || '').localeCompare(b.sector || '');
      }
      return (a.symbol || '').localeCompare(b.symbol || '');
    });
    return copy;
  }, [stocks, sortBy, weightMap, totalValue]);

  return (
    <Box>
      <Box display="flex" flexWrap="wrap" gap={2} justifyContent="space-between" alignItems="center" mb={1}>
        <Typography variant="h5">Stocks</Typography>
        <Box display="flex" gap={2} flexWrap="wrap">
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
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel>Sort by</InputLabel>
            <Select
              value={sortBy}
              label="Sort by"
              onChange={(e) => setSortBy(e.target.value)}
            >
              <MenuItem value="symbol">Symbol</MenuItem>
              <MenuItem value="weight">Weight</MenuItem>
              <MenuItem value="dividend">Dividend yield</MenuItem>
              <MenuItem value="sector">Sector</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </Box>

      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Symbol</TableCell>
            <TableCell>Name</TableCell>
            <TableCell>Sector</TableCell>
            <TableCell align="right">Weight</TableCell>
            <TableCell align="right">Dividend yield</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {sorted.map((s) => (
            <TableRow key={s.id}>
              <TableCell>
                {(() => {
                  const displaySymbol = formatTicker(s.symbol);
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
              <TableCell>{s.name}</TableCell>
              <TableCell>{s.sector}</TableCell>
              <TableCell align="right">
                {getWeightPercent(s)}
              </TableCell>
              <TableCell align="right">{Number(s.dividend_yield || 0).toFixed(2)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Box>
  );
}

export default StockTable;
