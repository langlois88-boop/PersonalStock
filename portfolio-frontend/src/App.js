import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import './App.css';
import GlobalPortfolio from './components/GlobalPortfolio';
import OptimizerPage from './components/OptimizerPage';
import ScoutEyePage from './components/ScoutEyePage';
import AnalyticsLabPage from './components/AnalyticsLabPage';
import LivePaperTrading from './components/LivePaperTrading';
import ManagePortfolioPage from './components/ManagePortfolioPage';
import IntradayAI from './components/IntradayAI';
import RiskControlCenter from './components/RiskControlCenter';
import LogsCenterPage from './components/LogsCenterPage';
import AskQuantPage from './components/AskQuantPage';
import AICenterPage from './components/AICenterPage';
import WarRoomPage from './components/WarRoomPage';
import MainLayout from './components/layout/MainLayout';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <MainLayout>
          <Routes>
            <Route path="/" element={<WarRoomPage />} />
            <Route path="/war-room" element={<WarRoomPage />} />
            <Route path="/portfolio" element={<GlobalPortfolio />} />
            <Route path="/manage" element={<ManagePortfolioPage />} />
            <Route path="/optimizer" element={<OptimizerPage />} />
            <Route path="/ai-center" element={<AICenterPage />} />
            <Route path="/scout" element={<ScoutEyePage />} />
            <Route path="/lab" element={<AnalyticsLabPage />} />
            <Route path="/paper" element={<LivePaperTrading />} />
            <Route path="/intraday" element={<IntradayAI />} />
            <Route path="/risk" element={<RiskControlCenter />} />
            <Route path="/logs" element={<LogsCenterPage />} />
            <Route path="/ask-quant" element={<AskQuantPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </MainLayout>
      </Router>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

export default App;
