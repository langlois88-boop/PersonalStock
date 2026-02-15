import { useEffect, useMemo, useState } from 'react';
import { cachedGet } from '../api/cachedApi';
import { subscribeApiErrors } from '../api/errorStore';

const severityStyle = (level) => {
  if (level === 'critical') return 'border-rose-500/40 bg-rose-500/10 text-rose-200';
  if (level === 'warning') return 'border-amber-500/40 bg-amber-500/10 text-amber-200';
  return 'border-slate-700 bg-slate-900/60 text-slate-200';
};

const formatTime = (value) => {
  if (!value) return '—';
  try {
    return new Date(value).toLocaleString();
  } catch (err) {
    return String(value);
  }
};

function UnifiedAlerts() {
  const [health, setHealth] = useState(null);
  const [monitoring, setMonitoring] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [apiErrors, setApiErrors] = useState([]);

  useEffect(() => {
    let active = true;

    const load = async () => {
      try {
        const [healthData, monitoringData, alertsData] = await Promise.all([
          cachedGet('health/', {}, 30000),
          cachedGet('models/monitoring/', {}, 60000),
          cachedGet('alerts/', { page_size: 6 }, 20000),
        ]);
        if (!active) return;
        setHealth(healthData || null);
        setMonitoring(monitoringData?.results || []);
        const list = Array.isArray(alertsData?.results) ? alertsData.results : alertsData || [];
        setAlerts(list.slice(0, 6));
      } catch (err) {
        if (!active) return;
        setHealth(null);
        setMonitoring([]);
        setAlerts([]);
      }
    };

    load();
    const interval = setInterval(load, 60000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => subscribeApiErrors(setApiErrors), []);

  const driftAlerts = useMemo(() => {
    const threshold = 0.2;
    return (monitoring || [])
      .filter((entry) => entry?.drift?.psi !== undefined && entry.drift.psi !== null)
      .filter((entry) => Number(entry.drift.psi) >= threshold)
      .map((entry) => ({
        id: `${entry.model_name}-${entry.sandbox}-drift`,
        title: `Drift détecté · ${entry.model_name} · ${entry.sandbox}`,
        message: `PSI ${Number(entry.drift.psi).toFixed(3)} (seuil ${threshold})`,
        level: 'warning',
      }));
  }, [monitoring]);

  const taskAlerts = useMemo(() => {
    const tasks = health?.tasks || {};
    const tracked = [
      { key: 'compute_continuous_evaluation_daily', label: 'Évaluation continue' },
      { key: 'auto_retrain_on_drift_daily', label: 'Retrain drift' },
      { key: 'auto_rollback_models_daily', label: 'Rollback modèle' },
    ];
    return tracked
      .map((task) => {
        const status = tasks?.[task.key]?.status || 'UNKNOWN';
        if (status === 'SUCCESS') return null;
        return {
          id: `task-${task.key}`,
          title: `${task.label} · ${status}`,
          message: tasks?.[task.key]?.error || 'Dernière exécution non OK.',
          level: status === 'FAILED' ? 'critical' : 'warning',
        };
      })
      .filter(Boolean);
  }, [health]);

  const apiAlertItems = useMemo(() => {
    return (apiErrors || []).slice(0, 5).map((err, index) => ({
      id: `api-${index}-${err?.timestamp || ''}`,
      title: `API ${err?.status || 'ERR'} ${err?.method || ''}`.trim(),
      message: `${err?.message || 'Erreur API'}${err?.url ? ` · ${err.url}` : ''}`,
      level: 'critical',
      timestamp: err?.timestamp,
    }));
  }, [apiErrors]);

  const alertEvents = (alerts || []).map((entry) => ({
    id: `alert-${entry.id}`,
    title: entry.category || 'Alert',
    message: entry.message || '—',
    level: 'warning',
    timestamp: entry.created_at,
  }));

  const merged = [...apiAlertItems, ...taskAlerts, ...driftAlerts, ...alertEvents];

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4">
      <div className="flex items-center justify-between">
        <p className="text-white font-semibold">Alertes unifiées</p>
        <span className="text-xs text-slate-400">API · Drift · Retrain · Rollback</span>
      </div>
      <div className="mt-4 space-y-3">
        {merged.length === 0 ? (
          <p className="text-sm text-slate-400">Aucune alerte active.</p>
        ) : (
          merged.map((item) => (
            <div
              key={item.id}
              className={`rounded-xl border px-3 py-2 text-sm ${severityStyle(item.level)}`}
            >
              <div className="flex items-center justify-between">
                <span className="font-semibold">{item.title}</span>
                <span className="text-xs opacity-70">{formatTime(item.timestamp)}</span>
              </div>
              <p className="text-xs mt-1 opacity-90">{item.message}</p>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default UnifiedAlerts;
