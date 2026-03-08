import { NavLink } from 'react-router-dom';
import { BarChart3, Brain, Eye, LayoutDashboard, Activity, UploadCloud, TrendingUp, Shield, ClipboardList, MessageCircle } from 'lucide-react';

const navItems = [
  { id: 'home', label: 'Dashboard Home', icon: LayoutDashboard, to: '/' },
  { id: 'manage', label: 'Import & Transactions', icon: UploadCloud, to: '/manage' },
  { id: 'optimizer', label: 'AI Portfolio Optimizer', icon: Brain, to: '/optimizer' },
  { id: 'ai-center', label: 'AI Center', icon: Brain, to: '/ai-center' },
  { id: 'chat', label: 'Ask the Quant', icon: MessageCircle, to: '/ask-quant' },
  { id: 'scout', label: 'The Scout Eye', icon: Eye, to: '/scout' },
  { id: 'lab', label: 'Analytics & ML Lab', icon: BarChart3, to: '/lab' },
  { id: 'paper', label: 'Live Paper Trading', icon: Activity, to: '/paper' },
  { id: 'intraday', label: 'Intraday AI Guide', icon: TrendingUp, to: '/intraday' },
  { id: 'risk', label: 'Risk Control Center', icon: Shield, to: '/risk' },
  { id: 'logs', label: 'Logs Center', icon: ClipboardList, to: '/logs' },
];

function Sidebar() {
  return (
    <aside className="bg-slate-950 text-slate-200 w-full lg:w-64 p-6 flex lg:flex-col gap-6 border-r border-slate-900">
      <div>
        <p className="text-xs uppercase tracking-[0.3em] text-slate-500">Scout AI</p>
        <h1 className="text-xl font-semibold text-white">Command Center</h1>
      </div>
      <nav className="flex flex-col gap-2">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.id}
              to={item.to}
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
