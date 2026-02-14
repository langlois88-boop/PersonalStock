import { useEffect, useMemo, useState } from 'react';
import {
  Box,
  FormControl,
  InputLabel,
  List,
  ListItem,
  ListItemText,
  MenuItem,
  Select,
  Stack,
  Typography,
} from '@mui/material';

import api from '../api/api';

function StockNews() {
  const [stocks, setStocks] = useState([]);
  const [selected, setSelected] = useState('');
  const [latestNews, setLatestNews] = useState([]);
  const [stockNews, setStockNews] = useState([]);

  useEffect(() => {
    api.get('stocks/').then((res) => {
      setStocks(res.data);
      if (res.data.length) {
        setSelected(res.data[0].symbol);
      }
    });
  }, []);

  useEffect(() => {
    api
      .get('stocknews/', { params: { days: 5 } })
      .then((res) => setLatestNews(res.data))
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (!selected) return;
    api
      .get('stocknews/', { params: { symbol: selected } })
      .then((res) => setStockNews(res.data))
      .catch(console.error);
  }, [selected]);

  const latestFive = useMemo(() => latestNews.slice(0, 5), [latestNews]);

  const stripHtml = (value) => {
    if (!value) return '';
    return String(value)
      .replace(/&nbsp;/g, ' ')
      .replace(/<[^>]*>/g, '')
      .replace(/\s+/g, ' ')
      .trim();
  };

  const sentimentColor = (sentiment) => {
    const value = Number(sentiment || 0);
    if (value > 0.2) return '#16a34a';
    if (value < -0.2) return '#dc2626';
    return '#64748b';
  };

  const formatHeadline = (item) => {
    const symbol = item.stock_symbol ? `${item.stock_symbol} ` : '';
    return `${symbol}${stripHtml(item.headline)}`.trim();
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
        <Typography variant="h5">News Feed</Typography>
        <FormControl size="small" sx={{ minWidth: 220 }}>
          <InputLabel>Stock</InputLabel>
          <Select
            value={selected}
            label="Stock"
            onChange={(e) => setSelected(e.target.value)}
          >
            {stocks.map((s) => (
              <MenuItem key={s.id} value={s.symbol}>
                {s.symbol} — {s.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>

      <Stack spacing={2}>
        <Box>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Latest 5 days
          </Typography>
          <List dense sx={{ maxWidth: '100%' }}>
            {latestFive.map((n) => (
              <ListItem key={n.id} component="a" href={n.url} target="_blank" rel="noreferrer">
                <ListItemText
                  primary={formatHeadline(n)}
                  primaryTypographyProps={{ sx: { wordBreak: 'break-word' } }}
                  secondary={
                    <Box component="span">
                      {`${stripHtml(n.summary) || n.source || ''} • ${
                        n.published_at ? new Date(n.published_at).toLocaleDateString() : 'Recent'
                      } • `}
                      <Box
                        component="span"
                        sx={{
                          color: sentimentColor(n.sentiment),
                          fontWeight: 600,
                        }}
                      >
                        Sentiment: {Number(n.sentiment || 0).toFixed(2)}
                      </Box>
                    </Box>
                  }
                  secondaryTypographyProps={{ sx: { wordBreak: 'break-word' } }}
                />
              </ListItem>
            ))}
          </List>
        </Box>

        <Box>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            News by stock
          </Typography>
          <List dense sx={{ maxWidth: '100%' }}>
            {stockNews.map((n) => (
              <ListItem key={n.id} component="a" href={n.url} target="_blank" rel="noreferrer">
                <ListItemText
                  primary={formatHeadline(n)}
                  primaryTypographyProps={{ sx: { wordBreak: 'break-word' } }}
                  secondary={
                    <Box component="span">
                      {`${stripHtml(n.summary) || n.source || ''} • ${
                        n.published_at ? new Date(n.published_at).toLocaleDateString() : 'Recent'
                      } • `}
                      <Box
                        component="span"
                        sx={{
                          color: sentimentColor(n.sentiment),
                          fontWeight: 600,
                        }}
                      >
                        Sentiment: {Number(n.sentiment || 0).toFixed(2)}
                      </Box>
                    </Box>
                  }
                  secondaryTypographyProps={{ sx: { wordBreak: 'break-word' } }}
                />
              </ListItem>
            ))}
          </List>
        </Box>
      </Stack>
    </Box>
  );
}

export default StockNews;
