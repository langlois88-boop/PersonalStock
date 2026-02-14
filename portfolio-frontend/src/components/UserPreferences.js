import { useEffect, useState } from 'react';
import { Box, Card, CardContent, Typography } from '@mui/material';

import api from '../api/api';

function UserPreferences() {
  const [prefs, setPrefs] = useState([]);

  useEffect(() => {
    api.get('preferences/').then((res) => setPrefs(res.data)).catch(console.error);
  }, []);

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Adaptive Preferences
      </Typography>
      {prefs.length === 0 ? (
        <Typography color="text.secondary">No preferences yet.</Typography>
      ) : (
        prefs.map((p) => (
          <Card variant="outlined" key={p.id} sx={{ mb: 1 }}>
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary">
                User {p.user}
              </Typography>
              <Typography>Risk score: {p.risk_score}</Typography>
              <Typography>
                Preferred sectors: {Object.keys(p.preferred_sectors || {}).join(', ') || '—'}
              </Typography>
              {p.last_feedback && (
                <Typography>Last feedback: {p.last_feedback}</Typography>
              )}
            </CardContent>
          </Card>
        ))
      )}
    </Box>
  );
}

export default UserPreferences;
