import axios from 'axios';
import { pushApiError } from './errorStore';

const normalizeApiBase = (base) => {
	if (!base) return base;
	const trimmed = String(base).replace(/\/+$/, '');
	if (trimmed.endsWith('/api')) {
		return `${trimmed}/`;
	}
	return `${trimmed}/api/`;
};

const defaultApiBase = (() => {
	const host = window.location.hostname;
	const isLocal = host === 'localhost' || host === '127.0.0.1';
	if (isLocal) {
		return `${window.location.protocol}//${host}:8001`;
	}
	return window.location.origin;
})();

const apiBaseUrl = normalizeApiBase(
	process.env.REACT_APP_API_BASE_URL || defaultApiBase
);

const api = axios.create({ baseURL: apiBaseUrl, timeout: 30000 });

api.interceptors.response.use(
	(response) => response,
	(error) => {
		const status = error?.response?.status;
		const url = error?.config?.url;
		const method = error?.config?.method;
		const message = error?.response?.data?.error || error?.message || 'Request failed';
		pushApiError({
			status,
			url,
			method,
			message,
			timestamp: new Date().toISOString(),
		});
		return Promise.reject(error);
	}
);

export default api;
