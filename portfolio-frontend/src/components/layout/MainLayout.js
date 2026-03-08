import Sidebar from '../Sidebar';
import QuickSearchBar from '../QuickSearchBar';

function MainLayout({ children }) {
  return (
    <div className="min-h-screen bg-[#0b0f14] text-slate-100 flex flex-col lg:flex-row">
      <Sidebar />
      <main className="flex-1 flex flex-col min-w-0">
        <header className="sticky top-0 z-40 bg-[#0b0f14]/80 backdrop-blur-md p-4 border-b border-white/5">
          <QuickSearchBar />
        </header>
        <div className="p-6 lg:p-10 overflow-y-auto">
          {children}
        </div>
      </main>
    </div>
  );
}

export default MainLayout;
