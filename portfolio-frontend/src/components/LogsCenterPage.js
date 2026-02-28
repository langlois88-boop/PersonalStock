import { useCallback, useEffect, useMemo, useState } from 'react';
import { RefreshCw, AlertTriangle, CheckCircle2, XCircle, ListFilter } from 'lucide-react';
import api from '../api/api';

const categories = [
  { value: '', label: 'Toutes les catégories' },
  { value: 'AI_PENNY', label: 'AI Penny' },
  { value: 'AI_BLUECHIP', label: 'AI Bluechip' },
  { value: 'AI_CRYPTO', label: 'AI Crypto' },
  { value: 'PAPER_TRADE', label: 'Paper Trade' },
  { value: 'SYSTEM', label: 'System' },
  { value: 'TELEGRAM', label: 'Telegram' },
];

const levels = [
  { value: '', label: 'Tous les niveaux' },
  { value: 'INFO', label: 'INFO' },
  { value: 'SUCCESS', label: 'SUCCESS' },
  { value: 'WARNING', label: 'WARNING' },
  { value: 'ERROR', label: 'ERROR' },
];

const taskStatuses = [
  { value: '', label: 'Tous les statuts' },
  { value: 'SUCCESS', label: 'SUCCESS' },
  { value: 'FAILED', label: 'FAILED' },
];

const levelBadge = (level) => {
  if (level === 'SUCCESS') return 'bg-emerald-500/15 text-emerald-200 border-emerald-500/30';
  if (level === 'WARNING') return 'bg-amber-500/15 text-amber-200 border-amber-500/30';
  if (level === 'ERROR') return 'bg-rose-500/15 text-rose-200 border-rose-500/30';
  return 'bg-slate-700/30 text-slate-200 border-slate-600/40';
};

const statusBadge = (status) => {
  if (status === 'SUCCESS') return 'bg-emerald-500/15 text-emerald-200 border-emerald-500/30';
  if (status === 'FAILED') return 'bg-rose-500/15 text-rose-200 border-rose-500/30';
  return 'bg-slate-700/30 text-slate-200 border-slate-600/40';
};

const formatTime = (value) => {
  if (!value) return '—';
  try {
    return new Date(value).toLocaleString();
  } catch (err) {
    return String(value);
  }
};

function LogsCenterPage() {
  const [activeTab, setActiveTab] = useState('system');
  const [systemLogs, setSystemLogs] = useState([]);
  const [taskLogs, setTaskLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const [category, setCategory] = useState('');
  const [level, setLevel] = useState('');
  const [symbol, setSymbol] = useState('');
  const [errorsOnly, setErrorsOnly] = useState(false);
  const [limit, setLimit] = useState(200);

  const [task, setTask] = useState('');
  const [taskStatus, setTaskStatus] = useState('');
  const [taskErrorsOnly, setTaskErrorsOnly] = useState(false);
  const [taskLimit, setTaskLimit] = useState(200);

  const fetchSystemLogs = useCallback(async () => {
    setLoading(true);
    try {
      const response = await api.get('logs/data/', {
        params: {
          category: category || undefined,
          level: level || undefined,
          symbol: symbol || undefined,
          errors_only: errorsOnly ? '1' : undefined,
          limit,
        },
      });
      setSystemLogs(response?.data?.results || []);
      setLastUpdated(new Date().toISOString());
    } catch (err) {
      setSystemLogs([]);
    } finally {
      setLoading(false);
    }
  }, [category, level, symbol, errorsOnly, limit]);

  const fetchTaskLogs = useCallback(async () => {
    setLoading(true);
    try {
      const response = await api.get('logs/tasks/', {
        params: {
          task: task || undefined,
          status: taskStatus || undefined,
          errors_only: taskErrorsOnly ? '1' : undefined,
          limit: taskLimit,
        },
      });
      setTaskLogs(response?.data?.results || []);
      setLastUpdated(new Date().toISOString());
    } catch (err) {
      setTaskLogs([]);
    } finally {
      setLoading(false);
    }
  }, [task, taskStatus, taskErrorsOnly, taskLimit]);

  useEffect(() => {
    if (activeTab === 'system') {
      fetchSystemLogs();
    } else {
      fetchTaskLogs();
    }
  }, [activeTab, fetchSystemLogs, fetchTaskLogs]);

  useEffect(() => {
    if (!autoRefresh) return undefined;
    const interval = setInterval(() => {
      if (activeTab === 'system') {
        fetchSystemLogs();
      } else {
        fetchTaskLogs();
      }
    }, 15000);
    return () => clearInterval(interval);
  }, [activeTab, autoRefresh, fetchSystemLogs, fetchTaskLogs]);

  useEffect(() => {
    if (activeTab !== 'system') return;
    fetchSystemLogs();
  }, [category, level, symbol, errorsOnly, limit, activeTab, fetchSystemLogs]);

  useEffect(() => {
    if (activeTab !== 'tasks') return;
    fetchTaskLogs();
  }, [task, taskStatus, taskErrorsOnly, taskLimit, activeTab, fetchTaskLogs]);

  const systemHeader = useMemo(() => (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-6 gap-3">
      <select value={category} onChange={(e) => setCategory(e.target.value)} className="bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm">
        {categories.map((item) => (
          <option key={item.value || 'all'} value={item.value}>{item.label}</option>
        ))}
      </select>
      <select value={level} onChange={(e) => setLevel(e.target.value)} className="bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm">
        {levels.map((item) => (
          <option key={item.value || 'all'} value={item.value}>{item.label}</option>
        ))}
      </select>
      <input value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="Ticker" className="bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm" />
      <label className="flex items-center gap-2 text-sm text-slate-300">
        <input type="checkbox" checked={errorsOnly} onChange={(e) => setErrorsOnly(e.target.checked)} />
        Erreurs seulement
      </label>
      <input type="number" min="50" max="1000" value={limit} onChange={(e) => setLimit(Number(e.target.value || 200))} className="bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm" />
      <button type="button" onClick={fetchSystemLogs} className="flex items-center justify-center gap-2 bg-indigo-500/20 border border-indigo-400/30 text-indigo-200 px-3 py-2 rounded-lg text-sm">
        <RefreshCw size={16} /> Actualiser
      </button>
    </div>
  ), [category, level, symbol, errorsOnly, limit, fetchSystemLogs]);

  const taskHeader = useMemo(() => (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-3">
      <input value={task} onChange={(e) => setTask(e.target.value)} placeholder="Nom de tâche" className="bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm" />
      <select value={taskStatus} onChange={(e) => setTaskStatus(e.target.value)} className="bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm">
        {taskStatuses.map((item) => (
          <option key={item.value || 'all'} value={item.value}>{item.label}</option>
        ))}
      </select>
      <label className="flex items-center gap-2 text-sm text-slate-300">
        <input type="checkbox" checked={taskErrorsOnly} onChange={(e) => setTaskErrorsOnly(e.target.checked)} />
        Échecs seulement
      </label>
      <input type="number" min="50" max="1000" value={taskLimit} onChange={(e) => setTaskLimit(Number(e.target.value || 200))} className="bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm" />
      <button type="button" onClick={fetchTaskLogs} className="flex items-center justify-center gap-2 bg-indigo-500/20 border border-indigo-400/30 text-indigo-200 px-3 py-2 rounded-lg text-sm">
        <RefreshCw size={16} /> Actualiser
      </button>
    </div>
  ), [task, taskStatus, taskErrorsOnly, taskLimit, fetchTaskLogs]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <p className="text-slate-400 text-sm">Centre de logs par section</p>
          <h2 className="text-2xl font-semibold text-white">Logs Center</h2>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <ListFilter size={14} /> Dernière mise à jour: {lastUpdated ? new Date(lastUpdated).toLocaleTimeString() : '—'}
          </div>
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} />
            Auto refresh 15s
          </label>
        </div>
      </div>

      <div className="flex gap-2">
        <button type="button" onClick={() => setActiveTab('system')} className={`px-4 py-2 rounded-lg text-sm ${activeTab === 'system' ? 'bg-indigo-500/20 border border-indigo-400/30 text-indigo-200' : 'bg-slate-900 border border-slate-800 text-slate-400'}`}>
          Logs Système
        </button>
        <button type="button" onClick={() => setActiveTab('tasks')} className={`px-4 py-2 rounded-lg text-sm ${activeTab === 'tasks' ? 'bg-indigo-500/20 border border-indigo-400/30 text-indigo-200' : 'bg-slate-900 border border-slate-800 text-slate-400'}`}>
          Logs Tâches
        </button>
      </div>

      <div className="bg-slate-950/70 border border-slate-800 rounded-2xl p-4 space-y-4">
        {activeTab === 'system' ? systemHeader : taskHeader}

        <div className="text-xs text-slate-400 flex items-center gap-2">
          {loading ? (
            <AlertTriangle size={14} className="text-amber-300" />
          ) : (
            <CheckCircle2 size={14} className="text-emerald-300" />
          )}
          {loading ? 'Chargement en cours…' : 'Logs prêts'}
        </div>

        {activeTab === 'system' ? (
          <div className="overflow-auto rounded-xl border border-slate-800">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-900 text-slate-300">
                <tr>
                  <th className="text-left px-3 py-2">Heure</th>
                  <th className="text-left px-3 py-2">Catégorie</th>
                  <th className="text-left px-3 py-2">Niveau</th>
                  <th className="text-left px-3 py-2">Ticker</th>
                  <th className="text-left px-3 py-2">Message</th>
                  <th className="text-left px-3 py-2">Meta</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-900">
                {systemLogs.map((log) => (
                  <tr key={log.id} className="hover:bg-slate-900/40">
                    <td className="px-3 py-2 text-slate-400 whitespace-nowrap">{formatTime(log.timestamp)}</td>
                    <td className="px-3 py-2 text-slate-200">{log.category}</td>
                    <td className="px-3 py-2">
                      <span className={`inline-flex items-center px-2 py-1 rounded-full border text-xs ${levelBadge(log.level)}`}>
                        {log.level}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-slate-300">{log.symbol || '—'}</td>
                    <td className="px-3 py-2 text-slate-100 max-w-xl">
                      <div className="whitespace-pre-wrap break-words">{log.message}</div>
                    </td>
                    <td className="px-3 py-2 text-slate-400 text-xs">
                      {log.metadata && Object.keys(log.metadata).length > 0 ? JSON.stringify(log.metadata) : '—'}
                    </td>
                  </tr>
                ))}
                {systemLogs.length === 0 && (
                  <tr>
                    <td colSpan="6" className="px-3 py-6 text-center text-slate-500">Aucun log système trouvé.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="overflow-auto rounded-xl border border-slate-800">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-900 text-slate-300">
                <tr>
                  <th className="text-left px-3 py-2">Heure</th>
                  <th className="text-left px-3 py-2">Tâche</th>
                  <th className="text-left px-3 py-2">Statut</th>
                  <th className="text-left px-3 py-2">Durée</th>
                  <th className="text-left px-3 py-2">Erreur</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-900">
                {taskLogs.map((log) => (
                  <tr key={log.id} className="hover:bg-slate-900/40">
                    <td className="px-3 py-2 text-slate-400 whitespace-nowrap">{formatTime(log.started_at)}</td>
                    <td className="px-3 py-2 text-slate-200">{log.task_name}</td>
                    <td className="px-3 py-2">
                      <span className={`inline-flex items-center px-2 py-1 rounded-full border text-xs ${statusBadge(log.status)}`}>
                        {log.status}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-slate-300">{log.duration_ms ? `${log.duration_ms} ms` : '—'}</td>
                    <td className="px-3 py-2 text-slate-100">
                      <div className="whitespace-pre-wrap break-words max-w-xl">
                        {log.error || '—'}
                      </div>
                    </td>
                  </tr>
                ))}
                {taskLogs.length === 0 && (
                  <tr>
                    <td colSpan="5" className="px-3 py-6 text-center text-slate-500">Aucun log de tâche trouvé.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 text-xs text-slate-500">
        {activeTab === 'system' ? <XCircle size={14} /> : <AlertTriangle size={14} />}
        Les logs système proviennent de `SystemLog` et les logs tâches de `TaskRunLog`.
      </div>
    </div>
  );
}

export default LogsCenterPage;
