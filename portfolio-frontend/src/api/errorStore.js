const listeners = new Set();
const errors = [];
const MAX_ERRORS = 20;

export const pushApiError = (entry) => {
  if (!entry) return;
  errors.unshift(entry);
  if (errors.length > MAX_ERRORS) errors.pop();
  listeners.forEach((listener) => listener([...errors]));
};

export const subscribeApiErrors = (listener) => {
  if (!listener) return () => {};
  listener([...errors]);
  listeners.add(listener);
  return () => listeners.delete(listener);
};

export const getApiErrors = () => [...errors];
