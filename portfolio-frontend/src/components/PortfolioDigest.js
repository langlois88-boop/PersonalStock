import { useEffect, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Typography,
} from '@mui/material';

import api from '../api/api';

function PortfolioDigest() {
  const [portfolios, setPortfolios] = useState([]);
  const [selected, setSelected] = useState('');
  const [digests, setDigests] = useState([]);

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
      .get('digests/', { params: { portfolio_id: selected } })
      .then((res) => setDigests(res.data))
      .catch(console.error);
  }, [selected]);

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
        <Typography variant="h5">Portfolio Digest</Typography>
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

      {digests.length === 0 ? (
        <Typography color="text.secondary">
          No digests yet. The weekly task will generate one.
        </Typography>
      ) : (
        digests.map((d) => (
          <Card variant="outlined" key={d.id} sx={{ mb: 1 }}>
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary">
                {d.period_start} → {d.period_end}
              </Typography>
              <Typography>{d.summary}</Typography>
            </CardContent>
          </Card>
        ))
      )}
    </Box>
  );
}

export default PortfolioDigest;
