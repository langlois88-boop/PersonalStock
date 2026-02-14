import axios from 'axios';

const apiBaseUrl =
	process.env.REACT_APP_API_BASE_URL ||
	`${window.location.protocol}//${window.location.hostname}:8001/api/`;

const api = axios.create({ baseURL: apiBaseUrl });

export default api;
