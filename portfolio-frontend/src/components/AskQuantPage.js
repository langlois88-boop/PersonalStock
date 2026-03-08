import { useCallback, useState } from 'react';
import api from '../api/api';

function AskQuantPage() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const pushMessage = (role, content, context) => {
    setMessages((prev) => [...prev, { role, content, context }]);
  };

  const submitQuestion = useCallback(async (overrideQuestion) => {
    const text = String(overrideQuestion || question || '').trim();
    if (!text) {
      setError('Question requise.');
      return;
    }
    setError('');
    setLoading(true);
    pushMessage('user', text);
    if (!overrideQuestion) setQuestion('');
    const streamFromEndpoint = async (endpoint) => {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (!response.ok || !response.body) {
        throw new Error('stream_unavailable');
      }
      const assistantId = `${Date.now()}-assistant`;
      setMessages((prev) => [...prev, { id: assistantId, role: 'assistant', content: '' }]);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let done = false;
      const hardStopAt = Date.now() + 45000;
      while (!done) {
        const result = await reader.read();
        done = result.done;
        buffer += decoder.decode(result.value || new Uint8Array(), { stream: !done });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';
        parts.forEach((part) => {
          const line = part.trim();
          if (!line.startsWith('data:')) return;
          const payloadText = line.replace(/^data:\s*/, '');
          let payload;
          try {
            payload = JSON.parse(payloadText);
          } catch (err) {
            payload = { text: payloadText };
          }
          const chunk = payload?.text || '';
          if (chunk) {
            setMessages((prev) => prev.map((msg) => (
              msg.id === assistantId ? { ...msg, content: `${msg.content}${chunk}` } : msg
            )));
          }
          if (payload?.done) {
            done = true;
          }
        });
        if (!done && Date.now() > hardStopAt) {
          controller.abort();
          break;
        }
      }
    };
    try {
      const baseUrl = api?.defaults?.baseURL || '';
      await streamFromEndpoint(`${baseUrl}ask-the-quant-stream/`);
    } catch (err) {
      try {
        const fallbackStreamUrl = `${window.location.protocol}//${window.location.hostname}:8001/api/ask-the-quant-stream/`;
        await streamFromEndpoint(fallbackStreamUrl);
      } catch (fallbackErr) {
        try {
          const res = await api.post('ai/consult/', { question: text });
          const answer = res?.data?.answer || 'Réponse indisponible.';
          const context = res?.data?.context || null;
          pushMessage('assistant', answer, context);
        } catch (finalErr) {
          setError('Impossible de contacter le bot.');
        }
      }
    } finally {
      setLoading(false);
    }
  }, [question]);

  const renderContext = (context) => {
    if (!context) return null;
    const indicators = context?.indicators || {};
    const backtest = context?.backtest || null;
    const sandboxes = context?.sandbox_performance || [];
    const brokerSync = context?.broker_sync || null;
    return (
      <div className="text-xs text-slate-400 space-y-1">
        {context.latest_price ? <p>Prix: {context.latest_price}</p> : null}
        {indicators.rsi14 ? <p>RSI14: {Number(indicators.rsi14).toFixed(2)}</p> : null}
        {backtest ? (
          <p>
            Backtest {backtest.lookback_days}j · Win {Number(backtest.win_rate || 0).toFixed(2)}% · Sharpe{' '}
            {Number(backtest.sharpe_ratio || 0).toFixed(2)}
          </p>
        ) : null}
        {sandboxes.length ? (
          <div>
            <p>Sandboxes:</p>
            <ul className="list-disc list-inside">
              {sandboxes.map((row) => (
                <li key={row.id}>
                  {row.id}: SIM {row.sim?.win_rate ?? 0}% / Sharpe {row.sim?.sharpe ?? 0} · ALPACA {row.alpaca?.win_rate ?? 0}% / Sharpe {row.alpaca?.sharpe ?? 0}
                </li>
              ))}
            </ul>
          </div>
        ) : null}
        {brokerSync?.alpaca_status ? (
          <p>Broker sync: {brokerSync.alpaca_status} (diff {brokerSync.diff_internal_external})</p>
        ) : null}
      </div>
    );
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="flex flex-col gap-2">
        <h2 className="text-2xl font-semibold text-white">Ask the Quant</h2>
        <p className="text-sm text-slate-400">
          Chat d’analyse connecté à tes données (news, backtests, sandboxes, portefeuille).
        </p>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => submitQuestion('Analyse mes sandboxes et compare SIM vs Alpaca.')}
            disabled={loading}
            className="px-3 py-2 rounded-lg text-xs font-semibold bg-slate-800 text-slate-200 border border-slate-700"
          >
            Analyser mes Sandboxes
          </button>
          <button
            type="button"
            onClick={() => submitQuestion('Quel est le contexte actuel pour RY et ATD.TO ?')}
            disabled={loading}
            className="px-3 py-2 rounded-lg text-xs font-semibold bg-slate-800 text-slate-200 border border-slate-700"
          >
            Snapshot RY + ATD.TO
          </button>
        </div>

        <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 h-[420px] overflow-y-auto space-y-4">
          {messages.length === 0 ? (
            <p className="text-sm text-slate-400">
              Pose une question. Exemple: “HIVE est à 2.80$, on achète le dip ?”
            </p>
          ) : (
            messages.map((msg, index) => (
              <div key={`${msg.role}-${index}`} className="space-y-2">
                <p className="text-[11px] uppercase tracking-[0.2em] text-slate-500">
                  {msg.role === 'user' ? 'Question' : 'Réponse'}
                </p>
                <div
                  className={
                    msg.role === 'user'
                      ? 'rounded-xl bg-slate-800/60 p-3 text-slate-100'
                      : 'rounded-xl bg-indigo-500/10 border border-indigo-500/20 p-3 text-slate-100'
                  }
                >
                  <p className="whitespace-pre-line text-sm">{msg.content}</p>
                  {msg.role === 'assistant' ? renderContext(msg.context) : null}
                </div>
              </div>
            ))
          )}
        </div>

        {error ? <p className="text-xs text-rose-300">{error}</p> : null}

        <div className="flex flex-col gap-3">
          <textarea
            rows={3}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Écris ta question…"
            className="w-full rounded-lg bg-slate-950 border border-slate-800 text-sm text-slate-200 p-3 focus:outline-none focus:ring-1 focus:ring-indigo-500/60"
          />
          <div className="flex items-center justify-between">
            <p className="text-xs text-slate-500">Réponse basée sur tes données + DeepSeek.</p>
            <button
              type="button"
              onClick={() => submitQuestion()}
              disabled={loading}
              className="px-4 py-2 rounded-lg text-xs font-semibold bg-indigo-500/20 text-indigo-200 border border-indigo-400/40"
            >
              {loading ? 'Analyse…' : 'Envoyer'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AskQuantPage;
