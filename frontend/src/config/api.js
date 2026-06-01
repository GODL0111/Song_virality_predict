const trimTrailingSlash = (value) => value.replace(/\/+$/, '')

export const API_BASE_URL = trimTrailingSlash(
  import.meta.env.VITE_API_URL ||
    (typeof window !== 'undefined' && window.BACKEND_URL) ||
    'http://localhost:5001'
)
