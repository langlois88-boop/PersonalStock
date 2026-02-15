import { useEffect, useState } from 'react';

import api from '../api/api';

function TransactionForm() {
  const [accounts, setAccounts] = useState([]);
  const [stocks, setStocks] = useState([]);
  const [stockQuery, setStockQuery] = useState('');
  const [stockOptions, setStockOptions] = useState([]);
  const [adding, setAdding] = useState(false);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState('');
  const [form, setForm] = useState({
    account: '',
    stock: '',
    shares: '',
    price_per_share: '',
    date: '',
    transaction_type: 'BUY',
  });
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    api.get('accounts/').then((res) => {
      setAccounts(res.data);
      if (res.data.length) {
        setForm((prev) => ({ ...prev, account: res.data[0].id }));
      }
    });
    api.get('stocks/').then((res) => {
      setStocks(res.data);
      if (res.data.length) {
        setForm((prev) => ({ ...prev, stock: res.data[0].id }));
        setStockQuery(formatStockLabel(res.data[0].symbol, res.data[0].name));
      }
    });
  }, []);

  useEffect(() => {
    if (!stockQuery) {
      setStockOptions([]);
      setSearchError('');
      return undefined;
    }

    const timeout = setTimeout(() => {
      setSearching(true);
      setSearchError('');
      api
        .get('stocks/search/', { params: { q: stockQuery } })
        .then((res) => {
          const list = res.data?.results || [];
          setStockOptions(list);
        })
        .catch(() => {
          setStockOptions([]);
          setSearchError('Search unavailable. Check backend connection.');
        })
        .finally(() => setSearching(false));
    }, 250);

    return () => clearTimeout(timeout);
  }, [stockQuery]);

  const updateForm = (key, value) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const formatStockLabel = (symbol, name) => {
    const safeSymbol = (symbol || '').trim();
    const safeName = (name || '').trim();
    if (!safeName || safeName.toUpperCase() === safeSymbol.toUpperCase()) {
      return safeSymbol;
    }
    return `${safeSymbol} — ${safeName}`;
  };

  const submit = async (event) => {
    event.preventDefault();
    setError('');
    setMessage('');

    let stockId = form.stock;
    if (!stockId && stockQuery) {
      const symbol = parseSymbol(stockQuery);
      if (symbol) {
        const existing = findStockBySymbol(symbol);
        if (existing) {
          stockId = existing.id;
        } else {
          try {
            const res = await api.post('stocks/create_from_symbol/', { symbol });
            const created = res.data;
            if (created?.id) {
              setStocks((prev) => [...prev, created]);
              stockId = created.id;
            }
          } catch (err) {
            console.error(err);
            setError(err?.response?.data || 'Failed to add stock.');
            return;
          }
        }
      }
    }

    api
      .post('account-transactions/', {
        account: form.account,
        stock: stockId,
        date: form.date,
        type: form.transaction_type,
        quantity: Number(form.shares),
        price: Number(form.price_per_share),
      })
      .then(() => {
        setMessage('Transaction saved.');
        setForm((prev) => ({ ...prev, shares: '', price_per_share: '' }));
      })
      .catch((err) => {
        console.error(err);
        setError(err?.response?.data || 'Failed to save transaction.');
      });
  };

  const findStockBySymbol = (symbol) =>
    stocks.find((s) => s.symbol?.toUpperCase() === symbol.toUpperCase());

  const parseSymbol = (value) => {
    if (!value) return '';
    const parts = value.split('—');
    return parts[0].trim().toUpperCase();
  };

  const handleStockSelect = async (value) => {
    const symbol = parseSymbol(value);
    if (!symbol) return;

    const existing = findStockBySymbol(symbol);
    if (existing) {
      updateForm('stock', existing.id);
      setStockQuery(formatStockLabel(existing.symbol, existing.name));
      return;
    }

    await createFromSymbol(symbol);
  };

  const createFromSymbol = async (symbol) => {
    try {
      setAdding(true);
      const res = await api.post('stocks/create_from_symbol/', { symbol });
      const created = res.data;
      if (created?.id) {
        setStocks((prev) => [...prev, created]);
        updateForm('stock', created.id);
        setStockQuery(formatStockLabel(created.symbol, created.name));
      }
    } catch (err) {
      console.error(err);
      setError(err?.response?.data || 'Failed to add stock.');
    } finally {
      setAdding(false);
    }
  };

  const syncStockSelection = async (value) => {
    setStockQuery(value);
    if (!value) {
      updateForm('stock', '');
      return;
    }

    if (value.includes('—')) {
      await handleStockSelect(value);
      return;
    }
  };

  return (
    <div className="portfolio-card">
      <form className="transaction-form" onSubmit={submit}>
        <label>
          Account
          <select
            value={form.account}
            onChange={(e) => updateForm('account', Number(e.target.value))}
          >
            {accounts.length === 0 ? (
              <option value="">No account available</option>
            ) : (
              accounts.map((account) => (
                <option key={account.id} value={account.id}>
                  {account.name} ({account.account_type})
                </option>
              ))
            )}
          </select>
        </label>

        <label>
          Stock
          <input
            list="stock-options"
            value={stockQuery}
            onChange={(e) => syncStockSelection(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                handleStockSelect(stockQuery);
              }
            }}
            placeholder="Type a symbol (e.g., F, FLT)"
            required
          />
          <div style={{ marginTop: 6, color: 'var(--text-muted)', fontSize: 12 }}>
            Type a symbol or name, then pick a result below.
          </div>
          {searching ? (
            <div style={{ marginTop: 6, color: 'var(--text-muted)', fontSize: 12 }}>
              Searching…
            </div>
          ) : null}
          {searchError ? (
            <div style={{ marginTop: 6, color: '#ef4444', fontSize: 12 }}>{searchError}</div>
          ) : null}
          {!searching && stockQuery.length >= 2 && stockOptions.length === 0 && !searchError ? (
            <div style={{ marginTop: 6, color: 'var(--text-muted)', fontSize: 12 }}>
              No results. Try a symbol (AAPL) or company name (Apple).
            </div>
          ) : null}
          <datalist id="stock-options">
            {stocks.map((s) => (
              <option key={`local-${s.id}`} value={formatStockLabel(s.symbol, s.name)} />
            ))}
            {stockOptions.map((s) => (
              <option
                key={`remote-${s.symbol}`}
                value={formatStockLabel(s.symbol, s.name || '')}
              />
            ))}
          </datalist>
        </label>

        <label>
          Type
          <select
            value={form.transaction_type}
            onChange={(e) => updateForm('transaction_type', e.target.value)}
          >
            <option value="BUY">BUY</option>
            <option value="SELL">SELL</option>
          </select>
        </label>

        <label>
          Shares
          <input
            type="number"
            min="0"
            step="0.01"
            value={form.shares}
            onChange={(e) => updateForm('shares', e.target.value)}
            required
          />
        </label>

        <label>
          Price per share
          <input
            type="number"
            min="0"
            step="0.01"
            value={form.price_per_share}
            onChange={(e) => updateForm('price_per_share', e.target.value)}
            required
          />
        </label>

        <label>
          Date
          <input
            type="date"
            value={form.date}
            onChange={(e) => updateForm('date', e.target.value)}
            required
          />
        </label>

        <button type="submit">Save Transaction</button>
      </form>
      {message && <p className="success">{message}</p>}
      {error && <p className="error">{typeof error === 'string' ? error : JSON.stringify(error)}</p>}
    </div>
  );
}

export default TransactionForm;
