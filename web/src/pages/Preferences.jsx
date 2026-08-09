import React, { useState, useEffect, useContext } from 'react';
import axios from 'axios';
import { AuthContext } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';

const API_URL = import.meta.env.VITE_API_URL || ${API_URL};

const Preferences = () => {
  const { token, user } = useContext(AuthContext);
  const navigate = useNavigate();
  const [categories, setCategories] = useState([]);
  const [language, setLanguage] = useState('en');
  const [saved, setSaved] = useState(false);

  const availableCategories = ['news', 'sports', 'finance', 'entertainment', 'lifestyle', 'health', 'travel', 'autos'];

  useEffect(() => {
    if (!token) return navigate('/login');
    
    axios.get(${API_URL}, {
      headers: { Authorization: `Bearer ${token}` }
    }).then(res => {
      setCategories(res.data.preferred_categories || []);
      setLanguage(res.data.language || 'en');
    });
  }, [token, navigate]);

  const toggleCategory = (cat) => {
    setCategories(prev => 
      prev.includes(cat) ? prev.filter(c => c !== cat) : [...prev, cat]
    );
  };

  const handleSave = async () => {
    try {
      await axios.patch(${API_URL}, {
        preferred_categories: categories,
        language
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-serif font-bold mb-8">Your Preferences</h1>
      
      <div className="bg-white p-6 border border-gray-200 mb-8">
        <h2 className="text-xl font-bold mb-4 font-serif">Preferred Categories</h2>
        <div className="flex flex-wrap gap-3">
          {availableCategories.map(cat => (
            <button
              key={cat}
              onClick={() => toggleCategory(cat)}
              className={`px-4 py-2 text-sm font-bold uppercase tracking-wider rounded-full transition-colors border ${
                categories.includes(cat) 
                  ? 'bg-black text-white border-black' 
                  : 'bg-white text-gray-700 border-gray-300 hover:border-gray-500'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>
      
      <div className="bg-white p-6 border border-gray-200 mb-8">
        <h2 className="text-xl font-bold mb-4 font-serif">Language Preference</h2>
        <div className="flex space-x-6">
          <label className="flex items-center space-x-2 cursor-pointer">
            <input 
              type="radio" 
              name="lang" 
              value="en" 
              checked={language === 'en'} 
              onChange={() => setLanguage('en')}
              className="text-black focus:ring-black"
            />
            <span>English</span>
          </label>
          <label className="flex items-center space-x-2 cursor-pointer">
            <input 
              type="radio" 
              name="lang" 
              value="hi" 
              checked={language === 'hi'} 
              onChange={() => setLanguage('hi')}
              className="text-black focus:ring-black"
            />
            <span>हिंदी (Hindi)</span>
          </label>
        </div>
        <p className="text-xs text-gray-500 mt-2 italic">Note: Hindi translations are powered by Gemini AI.</p>
      </div>
      
      <button 
        onClick={handleSave}
        className="bg-black text-white px-8 py-3 font-bold uppercase tracking-widest text-sm hover:bg-gray-800 transition"
      >
        Save Preferences
      </button>
      
      {saved && <span className="ml-4 text-green-600 font-bold">Saved successfully!</span>}
    </div>
  );
};

export default Preferences;
