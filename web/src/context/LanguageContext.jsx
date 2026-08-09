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
        'Preferences': 'Preferences',
        'Personalized Edition • Powered by Jagrit AI': 'Personalized Edition • Powered by Jagrit AI',
        'TOP TOPICS': 'TOP TOPICS',
        'Politics': 'Politics',
        'World': 'World',
        'Business': 'Business',
        'Technology': 'Technology',
        'Sports': 'Sports',
        'Jagrit Newsletter': 'Jagrit Newsletter',
        'Get the top stories delivered to your inbox daily.': 'Get the top stories delivered to your inbox daily.',
        'Your Email': 'Your Email',
        'Subscribe': 'Subscribe'
      },
      'hi': {
        'For You': 'आपके लिए',
        'Top Stories': 'मुख्य खबरें',
        'Login': 'लॉग इन',
        'Logout': 'लॉग आउट',
        'Preferences': 'प्राथमिकताएं',
        'Personalized Edition • Powered by Jagrit AI': 'व्यक्तिगत संस्करण • जाग्रत AI द्वारा संचालित',
        'TOP TOPICS': 'शीर्ष विषय',
        'Politics': 'राजनीति',
        'World': 'दुनिया',
        'Business': 'व्यापार',
        'Technology': 'तकनीक',
        'Sports': 'खेल',
        'Jagrit Newsletter': 'जाग्रत न्यूज़लैटर',
        'Get the top stories delivered to your inbox daily.': 'हर दिन अपने इनबॉक्स में मुख्य खबरें पाएं।',
        'Your Email': 'आपका ईमेल',
        'Subscribe': 'सब्सक्राइब करें'
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
