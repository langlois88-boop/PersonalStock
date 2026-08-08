import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { BarChart3, Activity, TrendingUp, Shield, ClipboardList } from 'lucide-react';
import { cachedGet } from '../api/cachedApi';

const sections = [
  {
    to: '/lab',
    icon: BarChart3,
    title: 'Analytics & ML Lab',
    desc: 'Résultats de backtests, performance des modèles et diagnostics du moteur ML.',
  },
  {
    to: '/paper',
    icon: Activity,
    title: 'Live Paper Trading',
    desc: 'Positions simulées en cours et historique des trades, avec les explications de chaque décision.',
  },
  {
    to: '/intraday',
    icon: TrendingUp,
    title: 'Intraday AI Guide',
    desc: 'Signaux temps réel sur un titre : pattern détecté, probabilité, stop-loss et target suggérés.',
  },
  {
    to: '/risk',
    icon: Shield,
    title: 'Risk Control Center',
    desc: 'Paramètres de gestion du risque et limite de perte journalière.',
  },
  {
    to: '/logs',
    icon: ClipboardList,
    title: 'Logs Center',
    desc: 'Historique des tâches automatiques et des événements système.',
  },
];

function DashboardPage() {
  const [health, setHealth] = useState(null);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      cachedGet('health/', {}, 30000).catch(() => null),
      cachedGet('paper-trades/summary/', {}, 30000).catch(() => null),
    ]).then(([h, s]) => {
      if (cancelled) return;
      setHealth(h);
      setSummary(s);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const pnl = summary?.closed_pnl;
  const pnlPositive = pnl != null && Number(pnl) >= 0;
  const statusOk = health?.status === 'ok';

  return (
    <div className="space-y-8">
      <div>
        <p className="text-xs uppercase tracking-[0.3em] text-slate-500">Dashboard Home</p>
        <h2 className="text-2xl font-semibold text-white mt-1">Bienvenue</h2>
        <p className="mt-2 text-sm text-slate-400 max-w-2xl">
          Ce bot analyse des actions avec des modèles de machine learning, teste ses décisions en
          « paper trading » (argent fictif, aucun risque réel), et affine ses stratégies au fil du
          temps. Les pages ci-dessous couvrent chaque étape : résultats du modèle, trades simulés
          en direct, signaux intraday, gestion du risque, et journal des tâches automatiques.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <div className="rounded-2xl bg-slate-900/70 p-5 border border-slate-800">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Statut système</p>
          {loading ? (
            <p className="mt-3 text-lg text-slate-400">Chargement…</p>
          ) : (
            <p className={`mt-3 text-2xl font-semibold ${statusOk ? 'text-emerald-300' : 'text-red-300'}`}>
              {statusOk ? 'OK' : 'Attention'}
            </p>
          )}
          <p className="text-xs text-slate-500 mt-1">
            {health?.stale_prices != null ? `${health.stale_prices} prix périmé(s)` : '—'}
          </p>
        </div>
        <div className="rounded-2xl bg-slate-900/70 p-5 border border-slate-800">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">P&L réalisé (paper)</p>
          {loading ? (
            <p className="mt-3 text-lg text-slate-400">Chargement…</p>
          ) : (
            <p className={`mt-3 text-2xl font-semibold ${pnlPositive ? 'text-emerald-300' : 'text-red-300'}`}>
              {pnl != null ? `$${pnl}` : '—'}
            </p>
          )}
          <p className="text-xs text-slate-500 mt-1">Trading simulé, aucun argent réel engagé.</p>
        </div>
        <div className="rounded-2xl bg-slate-900/70 p-5 border border-slate-800">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Capital disponible (paper)</p>
          {loading ? (
            <p className="mt-3 text-lg text-slate-400">Chargement…</p>
          ) : (
            <p className="mt-3 text-2xl font-semibold text-white">
              {summary?.available_capital != null ? `$${summary.available_capital}` : '—'}
            </p>
          )}
        </div>
      </div>

      <div>
        <h3 className="text-sm uppercase tracking-[0.2em] text-slate-500 mb-3">Explorer</h3>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {sections.map((s) => {
            const Icon = s.icon;
            return (
              <Link
                key={s.to}
                to={s.to}
                className="rounded-2xl bg-slate-900/70 p-5 border border-slate-800 hover:border-indigo-500/50 hover:bg-slate-900 transition block"
              >
                <Icon size={20} className="text-indigo-300" />
                <p className="mt-3 text-white font-semibold">{s.title}</p>
                <p className="mt-1 text-xs text-slate-400">{s.desc}</p>
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default DashboardPage;
