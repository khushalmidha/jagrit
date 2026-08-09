import React, { createContext, useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || ${API_URL};

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const initializeAuth = async () => {
      if (token) {
        try {
          const res = await axios.get(${API_URL}, {
            headers: { Authorization: `Bearer ${token}` }
          });
          setUser(res.data);
        } catch (error) {
          setToken(null);
          localStorage.removeItem('token');
          await loginAsGuest();
        }
      } else {
        await loginAsGuest();
      }
      setLoading(false);
    };

    initializeAuth();
  }, [token]);

  const loginAsGuest = async () => {
    try {
      // Try logging in with the shared guest account
      const res = await axios.post(${API_URL}, { 
        email: 'guest@jagrit.com', 
        password: 'guestpassword123' 
      });
      setToken(res.data.token);
      setUser(res.data.user);
      localStorage.setItem('token', res.data.token);
    } catch (err) {
      // If it doesn't exist, register it with a cold-start MIND ID
      try {
        const res = await axios.post(${API_URL}, { 
          name: 'Guest Reader',
          email: 'guest@jagrit.com', 
          password: 'guestpassword123',
          mind_user_id: 'NEW_USER_999'
        });
        setToken(res.data.token);
        setUser(res.data.user);
        localStorage.setItem('token', res.data.token);
      } catch (regErr) {
        console.error("Guest authentication failed", regErr);
      }
    }
  };

  const login = async (email, password) => {
    const res = await axios.post(${API_URL}, { email, password });
    setToken(res.data.token);
    setUser(res.data.user);
    localStorage.setItem('token', res.data.token);
  };

  const register = async (name, email, password) => {
    const res = await axios.post(${API_URL}, { name, email, password });
    setToken(res.data.token);
    setUser(res.data.user);
    localStorage.setItem('token', res.data.token);
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem('token');
  };

  return (
    <AuthContext.Provider value={{ user, token, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};
