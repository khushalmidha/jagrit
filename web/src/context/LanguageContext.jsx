import React, { createContext, useState, useEffect, useContext } from 'react';
import { AuthContext } from './AuthContext';
import axios from 'axios';

export const LanguageContext = createContext();

export const LanguageProvider = ({ children }) => {
  const [lang, setLang] = useState('en');
  const { token, user } = useContext(AuthContext);

  useEffect(() => {
    if (user && token) {
      axios.get('http://localhost:5000/api/preferences', {
        headers: { Authorization: `Bearer ${token}` }
      }).then(res => {
        if (res.data.language) setLang(res.data.language);
      }).catch(console.error);
    }
  }, [user, token]);

  const toggleLanguage = async () => {
    const newLang = lang === 'en' ? 'hi' : 'en';
    setLang(newLang);
    
    if (user && token) {
      try {
        await axios.patch('http://localhost:5000/api/preferences', { language: newLang }, {
          headers: { Authorization: `Bearer ${token}` }
        });
      } catch (error) {
        console.error("Failed to update language preference");
      }
    }
  };

  // Simple i18n dictionary for UI elements
  const t = (key) => {
    const dict = {
      'en': {
        'For You': 'For You',
        'Top Stories': 'Top Stories',
        'Login': 'Login',
        'Logout': 'Logout',
        'Preferences': 'Preferences'
      },
      'hi': {
        'For You': 'आपके लिए',
        'Top Stories': 'मुख्य खबरें',
        'Login': 'लॉग इन',
        'Logout': 'लॉग आउट',
        'Preferences': 'प्राथमिकताएं'
      }
    };
    return dict[lang][key] || key;
  };

  return (
    <LanguageContext.Provider value={{ lang, toggleLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
};
