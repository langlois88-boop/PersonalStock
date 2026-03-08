import { useEffect, useMemo, useState } from 'react';
import { Search } from 'lucide-react';
import api from '../api/api';
import AnalysisModal from './AnalysisModal';

const loadingMessages = [
  'Lecture des news...',
  'Calcul des probabilités...',
  'Consultation de Gemini...',
  'Évaluation du risque...',
];

function QuickSearchBar() {
  const [ticker, setTicker] = useState('');
  const [loading, setLoading] = useState(false);
  const [messageIndex, setMessageIndex] = useState(0);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);
  const [open, setOpen] = useState(false);

  const loadingMessage = useMemo(() => loadingMessages[messageIndex], [messageIndex]);

  useEffect(() => {
    if (!loading) return undefined;
    const interval = setInterval(() => {
      setMessageIndex((prev) => (prev + 1) % loadingMessages.length);
    }, 1200);
    return () => clearInterval(interval);
  }, [loading]);

  const pollAnalysis = async (symbol, attempt = 0) => {
    try {
      const response = await api.get('ai/analysis/', { params: { ticker: symbol } });
      if (response?.data?.status === 'processing') {
        if (attempt >= 10) {
          setError('Analyse en cours. Réessaie dans quelques instants.');
          setLoading(false);
          return;
        }
        setTimeout(() => pollAnalysis(symbol, attempt + 1), 2000);
        return;
      }
      setAnalysis(response.data);
      setLoading(false);
    } catch (err) {
      const message = err?.response?.data?.error || 'Analyse indisponible.';
      setError(message);
      setLoading(false);
    }
  };

  const runAnalysis = async () => {
    const symbol = ticker.trim().toUpperCase();
    if (!symbol) return;
    setLoading(true);
    setError(null);
    setAnalysis(null);
    setOpen(true);
    pollAnalysis(symbol);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    runAnalysis();
  };

  return (
    <div className="w-full">
      <form onSubmit={handleSubmit} className="flex flex-wrap items-center gap-3">
        <div className="flex-1 min-w-[220px] flex items-center gap-2 bg-slate-900/80 border border-slate-800 rounded-2xl px-4 py-3">
          <Search size={18} className="text-slate-400" />
          <input
            value={ticker}
            onChange={(event) => setTicker(event.target.value)}
            placeholder="Analyse express (ex: HIVE, TSLA)"
            className="w-full bg-transparent outline-none text-sm text-slate-100 placeholder:text-slate-500"
          />
        </div>
        <button
          type="submit"
          className="px-5 py-3 rounded-2xl bg-indigo-500/80 text-white text-sm font-semibold hover:bg-indigo-500"
        >
          Analyser
        </button>
        {loading && (
          <span className="text-xs text-slate-400">{loadingMessage}</span>
        )}
      </form>

      <AnalysisModal
        open={open}
        loading={loading}
        error={error}
        data={analysis}
        onClose={() => setOpen(false)}
      />
    </div>
  );
}

export default QuickSearchBar;
