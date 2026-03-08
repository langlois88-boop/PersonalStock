import { useEffect, useState } from 'react';
import api from '../api/api';
import LivePaperTrading from './LivePaperTrading';

const fallbackModels = [
  {
    model_name: 'BLUECHIP',
    model_version: 'local',
    status: 'ACTIVE',
    backtest_win_rate: 58.2,
    backtest_sharpe: 1.12,
    paper_win_rate: 54.1,
    paper_trades: 120,
    trained_at: null,
  },
  {
    model_name: 'PENNY',
    model_version: 'local',
    status: 'CANDIDATE',
    backtest_win_rate: 52.4,
    backtest_sharpe: 0.86,
    paper_win_rate: 50.3,
    paper_trades: 64,
    trained_at: null,
  },
];

const fallbackSandbox = [
  {
    sandbox: 'WATCHLIST',
    broker: 'ALL',
    trades: 0,
    win_rate: 0,
    total_return_pct: 0,
    sharpe_ratio: 0,
    max_drawdown: 0,
    final_balance: 0,
  },
  {
    sandbox: 'AI_BLUECHIP',
    broker: 'ALL',
    trades: 0,
    win_rate: 0,
    total_return_pct: 0,
    sharpe_ratio: 0,
    max_drawdown: 0,
    final_balance: 0,
  },
  {
    sandbox: 'AI_PENNY',
    broker: 'ALL',
    trades: 0,
    win_rate: 0,
    total_return_pct: 0,
    sharpe_ratio: 0,
    max_drawdown: 0,
    final_balance: 0,
  },
  {
    sandbox: 'AI_CRYPTO',
    broker: 'ALL',
    trades: 0,
    win_rate: 0,
    total_return_pct: 0,
    sharpe_ratio: 0,
    max_drawdown: 0,
    final_balance: 0,
  },
];

const formatPct = (value, digits = 1) => {
  if (!Number.isFinite(Number(value))) return '—';
  return `${Number(value).toFixed(digits)}%`;
};

const formatNumber = (value, digits = 2) => {
  if (!Number.isFinite(Number(value))) return '—';
  return Number(value).toFixed(digits);
};

const formatDate = (value) => {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  return date.toLocaleString();
};

function WarRoomPage() {
  const [modelRegistry, setModelRegistry] = useState(fallbackModels);
  const [sandboxPerformance, setSandboxPerformance] = useState(fallbackSandbox);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const load = async () => {
      setIsLoading(true);
      try {
        const [modelsRes, sandboxRes] = await Promise.all([
          api.get('models/registry/'),
          api.get('paper-trades/performance/'),
        ]);
        if (!isMounted) return;
        const models = modelsRes?.data?.results || [];
        const sandboxes = sandboxRes?.data?.results || [];
        setModelRegistry(models.length ? models : fallbackModels);
        setSandboxPerformance(sandboxes.length ? sandboxes : fallbackSandbox);
      } catch (err) {
        if (isMounted) {
          setModelRegistry(fallbackModels);
          setSandboxPerformance(fallbackSandbox);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    load();
    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div className="app-container war-room">
      <div className="app-hero">
        <div className="hero-eyebrow">ML Ops</div>
        <div className="hero-title">Machine Learning & Paper Trading</div>
        <div className="hero-subtitle">
          Supervision des modèles, sandboxes et exécution paper trading sur un seul écran.
        </div>
      </div>

      <div className="war-room-grid">
        <section className="lux-card war-room-panel">
          <div className="card-header">
            <div className="card-title">Registry des modèles</div>
            <div className="card-subtitle">Derniers modèles validés / candidats</div>
          </div>
          <div className="war-room-list">
            {modelRegistry.map((model, index) => (
              <div key={`${model.model_name}-${model.model_version}-${index}`} className="war-room-item">
                <div>
                  <div className="war-room-ticker">{model.model_name}</div>
                  <div className="war-room-price">{model.model_version}</div>
                  <div className="text-xs text-slate-400">{formatDate(model.trained_at)}</div>
                </div>
                <div className="text-right text-sm text-slate-200">
                  <div className="text-xs text-slate-400">{model.status}</div>
                  <div>BT Win {formatPct(model.backtest_win_rate, 1)}</div>
                  <div>Sharpe {formatNumber(model.backtest_sharpe, 2)}</div>
                  <div>Paper Win {formatPct(model.paper_win_rate, 1)}</div>
                </div>
              </div>
            ))}
            {!modelRegistry.length && !isLoading && (
              <div className="war-room-empty">Aucun modèle disponible.</div>
            )}
            {isLoading && (
              <div className="war-room-empty">Chargement des modèles...</div>
            )}
          </div>
        </section>

        <section className="lux-card war-room-panel">
          <div className="card-header">
            <div className="card-title">Sandbox performance</div>
            <div className="card-subtitle">SIM + Alpaca regroupés</div>
          </div>
          <div className="war-room-list">
            {sandboxPerformance.map((row, index) => (
              <div key={`${row.sandbox}-${row.broker}-${index}`} className="war-room-item">
                <div>
                  <div className="war-room-ticker">{row.sandbox}</div>
                  <div className="war-room-price">Trades {row.trades}</div>
                </div>
                <div className="text-right text-sm text-slate-200">
                  <div>Win {formatPct(row.win_rate, 1)}</div>
                  <div>Return {formatPct(row.total_return_pct, 2)}</div>
                  <div>DD {formatPct(row.max_drawdown, 2)}</div>
                  <div>Sharpe {formatNumber(row.sharpe_ratio, 2)}</div>
                </div>
              </div>
            ))}
            {!sandboxPerformance.length && !isLoading && (
              <div className="war-room-empty">Aucune performance sandbox.</div>
            )}
            {isLoading && (
              <div className="war-room-empty">Chargement des sandboxes...</div>
            )}
          </div>
        </section>
      </div>

      <section className="lux-card war-room-panel war-room-panel--full">
        <div className="card-header">
          <div className="card-title">Live Paper Trading</div>
          <div className="card-subtitle">Exécution et suivi des positions</div>
        </div>
        <LivePaperTrading />
      </section>
    </div>
  );
}

export default WarRoomPage;
