import { useState } from 'react';
import './App.css';
import Sidebar from './components/Sidebar';
import GlobalPortfolio from './components/GlobalPortfolio';
import OptimizerPage from './components/OptimizerPage';
import ScoutEyePage from './components/ScoutEyePage';
import AnalyticsLabPage from './components/AnalyticsLabPage';
import LivePaperTrading from './components/LivePaperTrading';
import ManagePortfolioPage from './components/ManagePortfolioPage';
import IntradayAI from './components/IntradayAI';
import QuickSearchBar from './components/QuickSearchBar';
import RiskControlCenter from './components/RiskControlCenter';

function App() {
  const [page, setPage] = useState('home');

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col lg:flex-row">
      <Sidebar active={page} onSelect={setPage} />
      <main className="flex-1 p-6 lg:p-10 bg-slate-950">
        <div className="mb-6">
          <QuickSearchBar />
        </div>
        {page === 'home' && <GlobalPortfolio />}
        {page === 'manage' && <ManagePortfolioPage />}
        {page === 'optimizer' && <OptimizerPage />}
        {page === 'scout' && <ScoutEyePage />}
        {page === 'lab' && <AnalyticsLabPage />}
        {page === 'paper' && <LivePaperTrading />}
        {page === 'intraday' && <IntradayAI />}
        {page === 'risk' && <RiskControlCenter />}
      </main>
    </div>
  );
}

export default App;
