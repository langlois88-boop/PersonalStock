import { useEffect, useMemo, useState } from 'react';

import api from '../api/api';

const toNumber = (value) => {
  const num = Number(value);
  return Number.isFinite(num) ? num : 0;
};

function TransactionEditor() {
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [editingId, setEditingId] = useState(null);
  const [editForm, setEditForm] = useState({
    quantity: '',
    price: '',
    date: '',
    type: 'BUY',
  });
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');

  const fetchTransactions = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await api.get('account-transactions/');
      const list = Array.isArray(res.data) ? res.data : [];
      setTransactions(list);
    } catch (err) {
      console.error(err);
      setError('Impossible de charger les transactions.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTransactions();
  }, []);

  const beginEdit = (tx) => {
    setMessage('');
    setError('');
    setEditingId(tx.id);
    setEditForm({
      quantity: tx.quantity ?? '',
      price: tx.price ?? '',
      date: tx.date ?? '',
      type: tx.type || 'BUY',
    });
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditForm({ quantity: '', price: '', date: '', type: 'BUY' });
  };

  const updateField = (key, value) => {
    setEditForm((prev) => ({ ...prev, [key]: value }));
  };

  const saveEdit = async (id) => {
    setMessage('');
    setError('');
    try {
      const payload = {
        quantity: toNumber(editForm.quantity),
        price: toNumber(editForm.price),
        date: editForm.date,
        type: editForm.type,
      };
      const res = await api.patch(`account-transactions/${id}/`, payload);
      const updated = res.data;
      setTransactions((prev) => prev.map((tx) => (tx.id === id ? updated : tx)));
      setMessage('Transaction mise à jour.');
      cancelEdit();
    } catch (err) {
      console.error(err);
      setError('Échec de la mise à jour.');
    }
  };

  const deleteTx = async (id) => {
    setMessage('');
    setError('');
    try {
      await api.delete(`account-transactions/${id}/`);
      setTransactions((prev) => prev.filter((tx) => tx.id !== id));
      if (editingId === id) {
        cancelEdit();
      }
      setMessage('Transaction supprimée.');
    } catch (err) {
      console.error(err);
      setError('Échec de la suppression.');
    }
  };

  const sortedTransactions = useMemo(() => {
    return [...transactions].sort((a, b) => {
      const dateA = a.date ? new Date(a.date).getTime() : 0;
      const dateB = b.date ? new Date(b.date).getTime() : 0;
      return dateB - dateA;
    });
  }, [transactions]);

  return (
    <div className="portfolio-card" style={{ marginTop: 24 }}>
      <div className="flex items-center justify-between" style={{ marginBottom: 12 }}>
        <h3 style={{ margin: 0 }}>Éditer les transactions</h3>
        <button type="button" onClick={fetchTransactions}>
          Rafraîchir
        </button>
      </div>

      {loading ? <p>Chargement…</p> : null}
      {error ? <p className="error">{error}</p> : null}
      {message ? <p className="success">{message}</p> : null}

      {!loading && sortedTransactions.length === 0 ? (
        <p style={{ color: 'var(--text-muted)' }}>Aucune transaction disponible.</p>
      ) : null}

      {!loading && sortedTransactions.length > 0 ? (
        <div style={{ overflowX: 'auto' }}>
          <table className="transaction-table" style={{ width: '100%', minWidth: 720 }}>
            <thead>
              <tr>
                <th>Date</th>
                <th>Compte</th>
                <th>Stock</th>
                <th>Type</th>
                <th>Quantité</th>
                <th>Prix / action</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {sortedTransactions.map((tx) => {
                const isEditing = editingId === tx.id;
                return (
                  <tr key={tx.id}>
                    <td>
                      {isEditing ? (
                        <input
                          type="date"
                          value={editForm.date}
                          onChange={(e) => updateField('date', e.target.value)}
                        />
                      ) : (
                        tx.date
                      )}
                    </td>
                    <td>{tx.account_name || tx.account}</td>
                    <td>{tx.stock_symbol || tx.stock}</td>
                    <td>
                      {isEditing ? (
                        <select
                          value={editForm.type}
                          onChange={(e) => updateField('type', e.target.value)}
                        >
                          <option value="BUY">BUY</option>
                          <option value="SELL">SELL</option>
                          <option value="DIVIDEND">DIVIDEND</option>
                        </select>
                      ) : (
                        tx.type
                      )}
                    </td>
                    <td>
                      {isEditing ? (
                        <input
                          type="number"
                          min="0"
                          step="0.000001"
                          value={editForm.quantity}
                          onChange={(e) => updateField('quantity', e.target.value)}
                        />
                      ) : (
                        tx.quantity
                      )}
                    </td>
                    <td>
                      {isEditing ? (
                        <input
                          type="number"
                          min="0"
                          step="0.000001"
                          value={editForm.price}
                          onChange={(e) => updateField('price', e.target.value)}
                        />
                      ) : (
                        tx.price
                      )}
                    </td>
                    <td style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                      {isEditing ? (
                        <>
                          <button type="button" onClick={() => saveEdit(tx.id)}>
                            Sauvegarder
                          </button>
                          <button type="button" onClick={cancelEdit} style={{ marginLeft: 8 }}>
                            Annuler
                          </button>
                        </>
                      ) : (
                        <>
                          <button type="button" onClick={() => beginEdit(tx)}>
                            Modifier
                          </button>
                          <button
                            type="button"
                            onClick={() => deleteTx(tx.id)}
                            style={{ marginLeft: 8 }}
                          >
                            Supprimer
                          </button>
                        </>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}

export default TransactionEditor;
