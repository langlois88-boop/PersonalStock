import { render, screen, waitFor } from '@testing-library/react';
import { act } from 'react';
import UnifiedAlerts from '../UnifiedAlerts';
import { cachedGet } from '../../api/cachedApi';
import { pushApiError } from '../../api/errorStore';

jest.mock('../../api/cachedApi', () => ({
  cachedGet: jest.fn(),
}));

const mockHealth = {
  tasks: {
    compute_continuous_evaluation_daily: { status: 'FAILED', error: 'Eval failed' },
    auto_retrain_on_drift_daily: { status: 'SUCCESS' },
    auto_rollback_models_daily: { status: 'UNKNOWN' },
  },
};

const mockMonitoring = {
  results: [
    { model_name: 'BLUECHIP', sandbox: 'WATCHLIST', drift: { psi: 0.25 } },
  ],
};

const mockAlerts = {
  results: [
    { id: 1, category: 'CAPITAL_THRESHOLD', message: 'Capital low', created_at: '2025-01-01T00:00:00Z' },
  ],
};

beforeEach(() => {
  cachedGet.mockImplementation((url) => {
    if (url === 'health/') return Promise.resolve(mockHealth);
    if (url === 'models/monitoring/') return Promise.resolve(mockMonitoring);
    if (url === 'alerts/') return Promise.resolve(mockAlerts);
    return Promise.resolve({});
  });
});

afterEach(() => {
  cachedGet.mockClear();
});

test('renders unified alerts from drift, tasks, and API errors', async () => {
  render(<UnifiedAlerts />);

  await waitFor(() => {
    expect(screen.getByText(/Alertes unifiées/i)).toBeInTheDocument();
    expect(screen.getByText(/Drift détecté/i)).toBeInTheDocument();
    expect(screen.getByText(/Évaluation continue/i)).toBeInTheDocument();
    expect(screen.getByText(/CAPITAL_THRESHOLD/i)).toBeInTheDocument();
  });

  await act(async () => {
    pushApiError({ message: 'Network error', status: 500, url: '/api/test' });
  });

  await waitFor(() => {
    expect(screen.getByText(/API 500/i)).toBeInTheDocument();
  });
});
