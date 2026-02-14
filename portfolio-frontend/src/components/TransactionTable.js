import { useEffect, useState } from 'react';
import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';

import api from '../api/api';

function TransactionTable() {
  const [transactions, setTransactions] = useState([]);

  useEffect(() => {
    api.get('transactions/').then((res) => setTransactions(res.data)).catch(console.error);
  }, []);

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Transactions
      </Typography>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Date</TableCell>
            <TableCell>Type</TableCell>
            <TableCell>Stock</TableCell>
            <TableCell align="right">Shares</TableCell>
            <TableCell align="right">Price</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {transactions.map((t) => (
            <TableRow key={t.id}>
              <TableCell>{t.date}</TableCell>
              <TableCell>{t.transaction_type}</TableCell>
              <TableCell>{t.stock_details?.symbol || t.stock}</TableCell>
              <TableCell align="right">{t.shares}</TableCell>
              <TableCell align="right">{t.price_per_share}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Box>
  );
}

export default TransactionTable;
