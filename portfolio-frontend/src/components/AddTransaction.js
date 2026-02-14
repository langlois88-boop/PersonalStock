import { Box, Typography } from '@mui/material';

import TransactionForm from './TransactionForm';

function AddTransaction() {
  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Add Transaction
      </Typography>
      <TransactionForm />
    </Box>
  );
}

export default AddTransaction;
