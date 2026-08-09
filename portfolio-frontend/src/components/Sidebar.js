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
    <aside className="bg-slate-950 text-slate-200 w-full lg:w-64 p-6 flex lg:flex-col gap-6 border-r border-slate-900">
      <div>
        <p className="text-xs uppercase tracking-[0.3em] text-slate-500">ML Trading Lab</p>
        <h1 className="text-xl font-semibold text-white">Command Center</h1>
      </div>
      <nav className="flex flex-col gap-2">
        {navItems.map((item) => {
          const Icon = item.icon;
          if (item.external) {
            return (
              <a
                key={item.id}
                href={item.to}
                target="_blank"
                rel="noreferrer"
                className="flex items-center gap-3 px-4 py-3 rounded-xl text-sm transition text-slate-400 hover:text-white hover:bg-slate-900"
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </a>
            );
          }
          return (
            <NavLink
              key={item.id}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) => `flex items-center gap-3 px-4 py-3 rounded-xl text-sm transition ${
                isActive
                  ? 'bg-indigo-500/15 text-indigo-200 border border-indigo-500/30'
                  : 'text-slate-400 hover:text-white hover:bg-slate-900'
              }`}
            >
              <Icon size={18} />
              <span>{item.label}</span>
            </NavLink>
          );
        })}
      </nav>
    </aside>
  );
}

export default Sidebar;
