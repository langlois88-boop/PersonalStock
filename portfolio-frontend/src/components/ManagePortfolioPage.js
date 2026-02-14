import { useEffect, useState } from 'react';
import ImportPortfolioCsv from './ImportPortfolioCsv';
import AddTransaction from './AddTransaction';
import api from '../api/api';

function ManagePortfolioPage() {
  const [accounts, setAccounts] = useState([]);
  const [accountName, setAccountName] = useState('');
  const [accountType, setAccountType] = useState('CASH');
  const [loadingAccounts, setLoadingAccounts] = useState(true);
  const [accountError, setAccountError] = useState('');

  const loadAccounts = () => {
    setLoadingAccounts(true);
    setAccountError('');
    api
      .get('accounts/')
      .then((res) => setAccounts(res.data || []))
      .catch(() => setAccountError('Impossible de charger les comptes.'))
      .finally(() => setLoadingAccounts(false));
  };

  useEffect(() => {
    loadAccounts();
  }, []);

  const createAccount = (event) => {
    event.preventDefault();
    if (!accountName.trim()) {
      setAccountError('Le nom du compte est requis.');
      return;
    }
    api
      .post('accounts/', { name: accountName.trim(), account_type: accountType })
      .then(() => {
        setAccountName('');
        setAccountType('CASH');
        loadAccounts();
      })
      .catch((err) => setAccountError(err?.response?.data || 'Création du compte échouée.'));
  };

  const deleteAccount = (id) => {
    api
      .delete(`accounts/${id}/`)
      .then(() => loadAccounts())
      .catch(() => setAccountError('Suppression du compte échouée.'));
  };

  return (
    <div className="space-y-6">
      <div>
        <p className="text-xs uppercase tracking-[0.3em] text-slate-500">Gestion</p>
        <h2 className="text-2xl font-semibold text-white">Portfolios & Transactions</h2>
        <p className="text-sm text-slate-400">Importe un CSV ou ajoute des transactions manuellement.</p>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">Comptes</h3>
          <span className="text-xs text-slate-500">Créer / Supprimer</span>
        </div>

        <form onSubmit={createAccount} className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <input
            className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm text-slate-100"
            placeholder="Nom du compte"
            value={accountName}
            onChange={(e) => setAccountName(e.target.value)}
          />
          <select
            className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm text-slate-100"
            value={accountType}
            onChange={(e) => setAccountType(e.target.value)}
          >
            <option value="CASH">CASH</option>
            <option value="TFSA">TFSA</option>
            <option value="CRI">CRI</option>
          </select>
          <button
            type="submit"
            className="bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl px-4 py-2 text-sm"
          >
            Ajouter un compte
          </button>
        </form>

        {accountError && <p className="text-sm text-rose-400 mt-3">{accountError}</p>}

        <div className="mt-4 space-y-2">
          {loadingAccounts ? (
            <p className="text-sm text-slate-400">Chargement…</p>
          ) : accounts.length === 0 ? (
            <p className="text-sm text-slate-400">Aucun compte enregistré.</p>
          ) : (
            accounts.map((account) => (
              <div key={account.id} className="flex items-center justify-between bg-slate-950/60 border border-slate-800 rounded-xl px-4 py-3">
                <div>
                  <p className="text-slate-100 font-semibold">{account.name}</p>
                  <p className="text-xs text-slate-500">{account.account_type}</p>
                </div>
                <button
                  type="button"
                  onClick={() => deleteAccount(account.id)}
                  className="text-xs text-rose-300 border border-rose-500/30 px-3 py-1 rounded-lg hover:bg-rose-500/10"
                >
                  Supprimer
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
          <ImportPortfolioCsv />
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
          <AddTransaction />
        </div>
      </div>
    </div>
  );
}

export default ManagePortfolioPage;
