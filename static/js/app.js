/* NiyamSetu AI — Global JS placeholder */
/* Add your shared frontend JS here */

// JWT token helpers
const Auth = {
  getToken:  () => localStorage.getItem('ns_token'),
  getRole:   () => localStorage.getItem('ns_role'),
  getName:   () => localStorage.getItem('ns_name'),
  isLoggedIn:() => !!localStorage.getItem('ns_token'),
  clear:     () => {
    localStorage.removeItem('ns_token');
    localStorage.removeItem('ns_role');
    localStorage.removeItem('ns_name');
  },
  // Use this for all authenticated fetch calls
  fetch: (url, options = {}) => {
    const token = localStorage.getItem('ns_token');
    return fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(options.headers || {}),
      },
    });
  },
};
