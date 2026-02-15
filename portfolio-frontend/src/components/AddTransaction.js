import { Box, Typography } from '@mui/material';

import TransactionForm from './TransactionForm';
import TransactionEditor from './TransactionEditor';

function AddTransaction() {
  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Add Transaction
      </Typography>
      <TransactionForm />
      <TransactionEditor />
    </Box>
  );
}

export default AddTransaction;
