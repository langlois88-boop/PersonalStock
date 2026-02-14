import { useEffect, useState } from 'react';
import { Alert, AlertTitle, Box, Typography } from '@mui/material';

import api from '../api/api';

function Alerts() {
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    api.get('alerts/').then((res) => setAlerts(res.data)).catch(console.error);
  }, []);

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Alerts
      </Typography>
      {alerts.length === 0 && (
        <Typography color="text.secondary">No alerts yet.</Typography>
      )}
      {alerts.map((a) => (
        <Alert key={a.id} severity="warning" sx={{ mb: 1 }}>
          <AlertTitle>{a.category}</AlertTitle>
          {a.message}
        </Alert>
      ))}
    </Box>
  );
}

export default Alerts;
