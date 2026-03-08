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

const getCookie = (name) => {
	if (!document?.cookie) return null;
	const cookie = document.cookie
		.split(';')
		.map((item) => item.trim())
		.find((item) => item.startsWith(`${name}=`));
	if (!cookie) return null;
	return decodeURIComponent(cookie.split('=')[1]);
};

const api = axios.create({
	baseURL: apiBaseUrl,
	timeout: 30000,
	withCredentials: true,
	xsrfCookieName: 'csrftoken',
	xsrfHeaderName: 'X-CSRFToken',
});

api.interceptors.request.use((config) => {
	const token = getCookie('csrftoken');
	if (token) {
		config.headers = {
			...config.headers,
			'X-CSRFToken': token,
		};
	}
	return config;
});

api.interceptors.response.use(
	(response) => response,
	(error) => {
		const suppress = error?.config?.meta?.suppressErrorReport;
		const status = error?.response?.status;
		const url = error?.config?.url;
		const method = error?.config?.method;
		const message = error?.response?.data?.error || error?.message || 'Request failed';
		if (!suppress) {
			pushApiError({
				status,
				url,
				method,
				message,
				timestamp: new Date().toISOString(),
			});
		}
		return Promise.reject(error);
	}
);

export default api;
