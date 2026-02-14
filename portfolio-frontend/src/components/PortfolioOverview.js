import { useEffect, useRef, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  CircularProgress,
  IconButton,
  Tooltip,
  Typography,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

import api from '../api/api';

function PortfolioOverview() {
  const [portfolios, setPortfolios] = useState([]);
  const [summaries, setSummaries] = useState({});
  const [loading, setLoading] = useState(true);
  const [confirmId, setConfirmId] = useState(null);
  const confirmTimer = useRef(null);

  const fetchPortfolios = () => {
    let mounted = true;

    api
      .get('portfolios/')
      .then((res) => {
        if (!mounted) return;
        setPortfolios(res.data);
        return Promise.all(res.data.map((p) => api.get(`portfolios/${p.id}/summary/`)));
      })
      .then((responses) => {
        if (!mounted || !responses) return;
        const map = {};
        responses.forEach((r) => {
          map[r.data.portfolio_id] = r.data;
        });
        setSummaries(map);
      })
      .catch((err) => console.error(err))
      .finally(() => mounted && setLoading(false));

    return () => {
      mounted = false;
    };
  };

  useEffect(() => {
    const cleanup = fetchPortfolios();
    return () => cleanup && cleanup();
  }, []);

  useEffect(() => {
    return () => {
      if (confirmTimer.current) {
        clearTimeout(confirmTimer.current);
      }
    };
  }, []);

  const handleDelete = async (portfolioId) => {
    if (confirmId !== portfolioId) {
      setConfirmId(portfolioId);
      if (confirmTimer.current) {
        clearTimeout(confirmTimer.current);
      }
      confirmTimer.current = setTimeout(() => setConfirmId(null), 2500);
      return;
    }

    try {
      await api.delete(`portfolios/${portfolioId}/`);
      setConfirmId(null);
      setLoading(true);
      fetchPortfolios();
    } catch (err) {
      console.error(err);
    }
  };

  if (loading) {
    return (
      <Box display="flex" alignItems="center" gap={2}>
        <CircularProgress size={20} />
        <Typography>Loading portfolios...</Typography>
      </Box>
    );
  }

  if (portfolios.length === 0) {
    return <Typography>No portfolios available.</Typography>;
  }

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Portfolio Overview
      </Typography>
      <Box
        display="grid"
        gap={2}
        sx={{ gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' } }}
      >
        {portfolios.map((p) => {
          const summary = summaries[p.id];
          return (
            <Card variant="outlined" key={p.id}>
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center" gap={1}>
                  <Typography variant="h6">{p.name}</Typography>
                  <Tooltip
                    title={
                      confirmId === p.id
                        ? 'Click again to confirm delete'
                        : 'Delete portfolio'
                    }
                  >
                    <IconButton
                      size="small"
                      onClick={() => handleDelete(p.id)}
                      className="delete-portfolio"
                    >
                      <CloseIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Box>
                <Typography color="text.secondary">
                  Capital: ${Number(p.capital || 0).toLocaleString()}
                </Typography>
                {summary && (
                  <>
                    <Typography color="text.secondary">
                      Dividends accrued: ${Number(
                        summary.total_dividends_accrued || 0
                      ).toLocaleString()}
                    </Typography>
                    <Typography color="text.secondary">
                      DRIP yield: {(summary.drip_projection?.yield_rate || 0) * 100}%
                    </Typography>
                  </>
                )}
              </CardContent>
            </Card>
          );
        })}
      </Box>
    </Box>
  );
}

export default PortfolioOverview;
