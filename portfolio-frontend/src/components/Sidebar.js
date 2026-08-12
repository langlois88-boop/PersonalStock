import { NavLink } from 'react-router-dom';
import { LayoutDashboard, BarChart3, Activity, TrendingUp, Shield, ClipboardList, LineChart, Microscope } from 'lucide-react';

const backendBase = (() => {
  const host = window.location.hostname;
  const isLocal = host === 'localhost' || host === '127.0.0.1';
  if (isLocal) {
    return `${window.location.protocol}//${host}:8001`;
  }
  return window.location.origin;
})();

const navItems = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard, to: '/' },
  { id: 'lab', label: 'Analytics & ML Lab', icon: BarChart3, to: '/lab' },
  { id: 'analysis', label: 'Screener fondamental', icon: Microscope, to: '/analysis' },
  { id: 'paper', label: 'Live Paper Trading', icon: Activity, to: '/paper' },
  { id: 'intraday', label: 'Intraday AI Guide', icon: TrendingUp, to: '/intraday' },
  { id: 'risk', label: 'Risk Control Center', icon: Shield, to: '/risk' },
  { id: 'logs', label: 'Logs Center', icon: ClipboardList, to: '/logs' },
  {
    id: 'ml-performance',
    label: 'ML Performance',
    icon: LineChart,
    to: `${backendBase}/models/performance/`,
    external: true,
  },
];

function Sidebar() {
  return (
    // Mobile (< lg) : barre horizontale compacte, icônes seules, défilement
    // horizontal si ça déborde -- avant ce fix, les 8 items s'empilaient
    // verticalement avec icône+texte, sans limite de hauteur, et pouvaient
    // occuper 400-500px sur un écran de téléphone (~700px de haut),
    // laissant à peine de la place pour la page elle-même (bug rapporté :
    // "on arrive pas à voir la page, on peut seulement scroller").
    // Desktop (lg+) : comportement inchangé, menu vertical complet avec
    // libellés.
    <aside className="bg-slate-950 text-slate-200 w-full lg:w-64 flex-shrink-0 border-r border-slate-900">
      <div className="hidden lg:block p-6">
        <p className="text-xs uppercase tracking-[0.3em] text-slate-500">ML Trading Lab</p>
        <h1 className="text-xl font-semibold text-white">Command Center</h1>
      </div>
      <nav className="flex lg:flex-col gap-2 overflow-x-auto lg:overflow-visible p-3 lg:p-6 lg:pt-0">
        {navItems.map((item) => {
          const Icon = item.icon;
          const content = (
            <>
              <Icon size={18} className="flex-shrink-0" />
              <span className="hidden lg:inline">{item.label}</span>
            </>
          );
          if (item.external) {
            return (
              <a
                key={item.id}
                href={item.to}
                target="_blank"
                rel="noreferrer"
                className="flex items-center gap-3 px-3 lg:px-4 py-2 lg:py-3 rounded-xl text-sm transition text-slate-400 hover:text-white hover:bg-slate-900 flex-shrink-0 whitespace-nowrap"
              >
                {content}
              </a>
            );
          }
          return (
            <NavLink
              key={item.id}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) => `flex items-center gap-3 px-3 lg:px-4 py-2 lg:py-3 rounded-xl text-sm transition flex-shrink-0 whitespace-nowrap ${
                isActive
                  ? 'bg-indigo-500/15 text-indigo-200 border border-indigo-500/30'
                  : 'text-slate-400 hover:text-white hover:bg-slate-900'
              }`}
            >
              {content}
            </NavLink>
          );
        })}
      </nav>
    </aside>
  );
}

export default Sidebar;
