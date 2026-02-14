import { useState } from 'react';
import { Box, Button, Card, CardContent, TextField, Typography } from '@mui/material';

import api from '../api/api';

function ImportPortfolioCsv() {
  const [file, setFile] = useState(null);
  const [portfolioName, setPortfolioName] = useState('');
  const [initialCapital, setInitialCapital] = useState('');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const submit = (event) => {
    event.preventDefault();
    setMessage('');
    setError('');

    if (!file) {
      setError('Please choose a CSV file.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    if (portfolioName) formData.append('portfolio_name', portfolioName);
    if (initialCapital) formData.append('initial_capital', initialCapital);

    api
      .post('import/portfolio/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      .then((res) => {
        setMessage(
          `Imported ${res.data.created_stocks} stocks and ${res.data.created_transactions} transactions.`
        );
      })
      .catch((err) => {
        console.error(err);
        setError(err?.response?.data?.error || 'Import failed.');
      });
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Import Portfolio (CSV)
      </Typography>
      <Card variant="outlined">
        <CardContent>
          <Box component="form" onSubmit={submit} display="grid" gap={2}>
            <TextField
              label="Portfolio name"
              value={portfolioName}
              onChange={(e) => setPortfolioName(e.target.value)}
            />
            <TextField
              label="Initial capital"
              type="number"
              value={initialCapital}
              onChange={(e) => setInitialCapital(e.target.value)}
            />
            <Button variant="outlined" component="label">
              Choose CSV
              <input
                type="file"
                accept=".csv"
                hidden
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
            </Button>
            {file && <Typography>{file.name}</Typography>}
            <Button type="submit" variant="contained">
              Import
            </Button>
            {message && <Typography color="success.main">{message}</Typography>}
            {error && <Typography color="error.main">{error}</Typography>}
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}

export default ImportPortfolioCsv;
