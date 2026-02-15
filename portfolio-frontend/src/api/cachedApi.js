import api from './api';

const cache = new Map();

const buildKey = (url, params) => `${url}::${JSON.stringify(params || {})}`;

export const cachedGet = async (url, params = {}, ttlMs = 60000) => {
  const key = buildKey(url, params);
  const now = Date.now();
  const cached = cache.get(key);
  if (cached && now - cached.timestamp < ttlMs) {
    return cached.data;
  }

  const response = await api.get(url, { params });
  cache.set(key, { data: response.data, timestamp: now });
  return response.data;
};

export const invalidateCache = (url, params = {}) => {
  const key = buildKey(url, params);
  cache.delete(key);
};
