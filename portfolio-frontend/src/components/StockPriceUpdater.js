import { useEffect, useState } from 'react';
import { Box, List, ListItem, ListItemText, Typography } from '@mui/material';

import api from '../api/api';

const ONE_HOUR = 60 * 60 * 1000;

function StockPriceUpdater() {
  const [stocks, setStocks] = useState([]);

  const formatPrice = (value) => {
    if (value === null || value === undefined) {
      return 'n/a';
    }
    const num = Number(value);
    if (Number.isNaN(num)) {
      return 'n/a';
    }
    if (num !== 0 && Math.abs(num) < 0.01) {
      return `$${num.toFixed(5)}`;
    }
    if (Math.abs(num) < 1) {
      return `$${num.toFixed(4)}`;
    }
    return `$${num.toFixed(2)}`;
  };

  useEffect(() => {
    let timer;

    const fetchStocks = () => {
      api
        .get('stocks/')
        .then((res) => setStocks(res.data))
        .catch(console.error);
    };

    fetchStocks();
    timer = setInterval(fetchStocks, ONE_HOUR);

    return () => clearInterval(timer);
  }, []);

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Latest Prices
      </Typography>
      <List dense>
        {stocks.map((s) => (
          <ListItem key={s.id}>
            <ListItemText
              primary={`${s.symbol} — ${s.name}`}
              secondary={`${formatPrice(s.latest_price)} • Low ${formatPrice(s.day_low)} • High ${formatPrice(s.day_high)}`}
            />
          </ListItem>
        ))}
      </List>
    </Box>
  );
}

export default StockPriceUpdater;
