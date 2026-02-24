import { useEffect, useMemo, useState } from 'react';
import { Gauge, ShieldCheck, TrendingDown } from 'lucide-react';
import api from '../api/api';

const STORAGE_KEY = 'risk-control-settings';

const defaultSettings = {
  confidenceThreshold: 85,
  maxDrawdown: 8,
  defaultSize: 150,
  maxSize: 300,
};

function RiskControlCenter() {
  const [settings, setSettings] = useState(defaultSettings);
  const [summary, setSummary] = useState(null);
  const [recentTrades, setRecentTrades] = useState([]);

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        setSettings({ ...defaultSettings, ...JSON.parse(stored) });
      } catch (err) {
        setSettings(defaultSettings);
      }
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  }, [settings]);

  useEffect(() => {
    api
      .get('paper-trades/summary/')
      .then((res) => setSummary(res.data))
      .catch(() => setSummary(null));

    api
      .get('paper-trades/', { params: { page_size: 10 } })
      .then((res) => {
        const data = res.data?.results || res.data || [];
        setRecentTrades(Array.isArray(data) ? data.slice(0, 10) : []);
      })
      .catch(() => setRecentTrades([]));
  }, []);

  const exposurePct = useMemo(() => {
    if (!summary) return 0;
    const openValue = Number(summary.open_value || 0);
    const available = Number(summary.available_capital || 0);
    const total = openValue + available;
    if (!total) return 0;
    return Math.min(100, Math.round((openValue / total) * 100));
  }, [summary]);

  const riskStatus = exposurePct > 70 ? 'Élevé' : exposurePct > 40 ? 'Modéré' : 'Contrôlé';

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <p className="text-xs uppercase tracking-[0.3em] text-slate-500">Risk Manager</p>
        <h2 className="text-2xl font-semibold text-white">Risk Control Center</h2>
        <p className="text-sm text-slate-400">Ajuste les paramètres de sécurité et surveille l'exposition en temps réel.</p>
      </header>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-2xl bg-slate-900/70 p-5 border border-slate-800">
          <div className="flex items-center gap-2 text-slate-200">
            <Gauge size={18} />
            <span className="text-sm font-semibold">Risque global</span>
          </div>
          <p className="mt-3 text-3xl font-semibold text-white">{riskStatus}</p>
          <p className="text-xs text-slate-400">Exposition actuelle : {exposurePct}%</p>
        </div>
        <div className="rounded-2xl bg-slate-900/70 p-5 border border-slate-800">
          <div className="flex items-center gap-2 text-slate-200">
            <ShieldCheck size={18} />
            <span className="text-sm font-semibold">Cash disponible</span>
          </div>
          <p className="mt-3 text-3xl font-semibold text-emerald-300">${summary?.available_capital || 0}</p>
          <p className="text-xs text-slate-400">Capital engagé : ${summary?.open_value || 0}</p>
        </div>
        <div className="rounded-2xl bg-slate-900/70 p-5 border border-slate-800">
          <div className="flex items-center gap-2 text-slate-200">
            <TrendingDown size={18} />
            <span className="text-sm font-semibold">Daily stop-loss</span>
          </div>
          <p className="mt-3 text-3xl font-semibold text-red-300">{settings.maxDrawdown}%</p>
          <p className="text-xs text-slate-400">Objectif de perte max journalier.</p>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="rounded-2xl bg-slate-900/70 p-5 border border-slate-800 space-y-4">
          <h3 className="text-sm font-semibold text-slate-200">Paramètres du Risk Manager</h3>
          {[
            {
              label: 'Seuil de confiance (%)',
              key: 'confidenceThreshold',
              min: 60,
              max: 95,
            },
            {
              label: 'Max Drawdown (%)',
              key: 'maxDrawdown',
              min: 3,
              max: 15,
            },
            {
              label: 'Mise standard ($)',
              key: 'defaultSize',
              min: 50,
              max: 500,
            },
            {
              label: 'Mise max ($)',
              key: 'maxSize',
              min: 150,
              max: 1000,
            },
          ].map((slider) => (
            <div key={slider.key} className="space-y-2">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>{slider.label}</span>
                <span className="text-slate-200 font-semibold">{settings[slider.key]}</span>
              </div>
              <input
                type="range"
                min={slider.min}
                max={slider.max}
                value={settings[slider.key]}
                onChange={(event) =>
                  setSettings((prev) => ({
                    ...prev,
                    [slider.key]: Number(event.target.value),
                  }))
                }
                className="w-full accent-indigo-500"
              />
            </div>
          ))}
        </section>

        <section className="rounded-2xl bg-slate-900/70 p-5 border border-slate-800 space-y-4">
          <h3 className="text-sm font-semibold text-slate-200">Historique confidence vs P&L</h3>
          {recentTrades.length === 0 ? (
            <p className="text-sm text-slate-400">Pas encore de trades à afficher.</p>
          ) : (
            <div className="space-y-3">
              {recentTrades.map((trade) => (
                <div
                  key={trade.id || `${trade.ticker}-${trade.entry_date}`}
                  className="flex items-center justify-between rounded-xl bg-slate-950/70 px-4 py-3 text-sm"
                >
                  <div>
                    <p className="text-slate-200 font-semibold">{trade.ticker}</p>
                    <p className="text-xs text-slate-500">Confidence: {(Number(trade.entry_signal || 0) * 100).toFixed(1)}%</p>
                  </div>
                  <span className={`font-semibold ${Number(trade.pnl || 0) >= 0 ? 'text-emerald-300' : 'text-red-300'}`}>
                    {Number(trade.pnl || 0).toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

export default RiskControlCenter;
