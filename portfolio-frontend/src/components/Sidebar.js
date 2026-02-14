import { BarChart3, Brain, Eye, LayoutDashboard, Activity, UploadCloud } from 'lucide-react';

const navItems = [
  { id: 'home', label: 'Dashboard Home', icon: LayoutDashboard },
  { id: 'manage', label: 'Import & Transactions', icon: UploadCloud },
  { id: 'optimizer', label: 'AI Portfolio Optimizer', icon: Brain },
  { id: 'scout', label: 'The Scout Eye', icon: Eye },
  { id: 'lab', label: 'Analytics & ML Lab', icon: BarChart3 },
  { id: 'paper', label: 'Live Paper Trading', icon: Activity },
];

function Sidebar({ active, onSelect }) {
  return (
    <aside className="bg-slate-950 text-slate-200 w-full lg:w-64 p-6 flex lg:flex-col gap-6 border-r border-slate-900">
      <div>
        <p className="text-xs uppercase tracking-[0.3em] text-slate-500">Scout AI</p>
        <h1 className="text-xl font-semibold text-white">Command Center</h1>
      </div>
      <nav className="flex flex-col gap-2">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = active === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelect(item.id)}
              className={`flex items-center gap-3 px-4 py-3 rounded-xl text-sm transition ${
                isActive
                  ? 'bg-indigo-500/15 text-indigo-200 border border-indigo-500/30'
                  : 'text-slate-400 hover:text-white hover:bg-slate-900'
              }`}
            >
              <Icon size={18} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>
    </aside>
  );
}

export default Sidebar;
