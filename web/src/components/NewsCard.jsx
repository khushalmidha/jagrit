import React, { useState, useContext } from 'react';
import { Bookmark, BookmarkCheck, ChevronDown, ChevronUp } from 'lucide-react';
import axios from 'axios';
import { LanguageContext } from '../context/LanguageContext';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

// Fixed category-based reliable images as requested
const CATEGORY_IMAGES = {
  technology: 'https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=800&h=600&auto=format&fit=crop',
  sports: 'https://images.unsplash.com/photo-1461896836934-ffe607ba8211?q=80&w=800&h=600&auto=format&fit=crop',
  finance: 'https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=800&h=600&auto=format&fit=crop',
  business: 'https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?q=80&w=800&h=600&auto=format&fit=crop',
  politics: 'https://images.unsplash.com/photo-1540910419892-4a36d2c3266c?q=80&w=800&h=600&auto=format&fit=crop',
  education: 'https://images.unsplash.com/photo-1523050854058-8df90110c9f1?q=80&w=800&h=600&auto=format&fit=crop',
  world: 'https://images.unsplash.com/photo-1521295121783-8a321d551ad2?q=80&w=800&h=600&auto=format&fit=crop',
  entertainment: 'https://images.unsplash.com/photo-1603190287605-e6ade32fa852?q=80&w=800&h=600&auto=format&fit=crop',
  default: 'https://images.unsplash.com/photo-1504711434969-e33886168f5c?q=80&w=800&h=600&auto=format&fit=crop'
};

const NewsCard = ({ article, isHero = false, token }) => {
  const [showScore, setShowScore] = useState(false);
  const [saved, setSaved] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const { lang } = useContext(LanguageContext);

  // Return fixed image based on category
  const getImageUrl = (category) => {
    const cat = String(category || '').toLowerCase();
    return CATEGORY_IMAGES[cat] || CATEGORY_IMAGES.default;
  };

  const handleSave = async (e) => {
    e.preventDefault();
    if (!token) return alert('Please login to save articles');
    try {
      await axios.post(`${API_URL}/api/feed/save/${article.news_id}`, {}, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setSaved(true);
    } catch (error) {
      if (error.response?.status === 400) setSaved(true);
    }
  };

  const displayTitle = lang === 'hi' && article.title_hi ? article.title_hi : article.title;
  const displayAbstract = lang === 'hi' && article.abstract_hi ? article.abstract_hi : (article.abstract || 'Read the full story to know more about this breaking development and its implications on the global landscape.');

  return (
    <div className={`flex flex-col bg-white overflow-hidden ${isHero ? 'md:flex-row border-b-[4px] border-black pb-8' : 'border-b border-gray-300 pb-6'}`}>
      <div className={`relative ${isHero ? 'md:w-[60%] h-[400px]' : 'h-56 mb-4'}`}>
        <img src={getImageUrl(article.category)} alt={displayTitle} className="w-full h-full object-cover border border-gray-200" />
      </div>
      
      <div className={`flex flex-col justify-between ${isHero ? 'md:w-[40%] md:pl-8 py-2' : ''}`}>
        <div>
          <div className="flex justify-between items-center mb-3">
            <span className="text-[10px] font-bold text-red-600 uppercase tracking-widest border-b-2 border-red-600 pb-1">
              {article.category || 'News'}
            </span>
            <button onClick={handleSave} className="text-gray-400 hover:text-black transition-colors">
              {saved ? <BookmarkCheck size={20} /> : <Bookmark size={20} />}
            </button>
          </div>
          
          <h2 className={`font-serif font-bold text-gray-900 leading-[1.1] mb-4 hover:text-red-700 cursor-pointer transition-colors ${isHero ? 'text-4xl md:text-5xl' : 'text-2xl'}`}>
            {displayTitle}
          </h2>
          
          <div className="text-gray-600 font-serif leading-relaxed">
            <p className={`${expanded ? '' : (isHero ? 'line-clamp-4' : 'line-clamp-3')} text-sm md:text-base`}>
              {displayAbstract}
            </p>
            {displayAbstract.length > 100 && (
              <button 
                onClick={() => setExpanded(!expanded)} 
                className="mt-2 text-xs font-bold uppercase tracking-wider text-black hover:text-red-600 flex items-center transition"
              >
                {expanded ? 'Show Less' : 'Read More'}
                {expanded ? <ChevronUp size={14} className="ml-1" /> : <ChevronDown size={14} className="ml-1" />}
              </button>
            )}
          </div>
        </div>
        
        <div className="mt-6 pt-3 border-t border-gray-200 flex justify-between items-center text-xs text-gray-500 uppercase tracking-wider font-bold">
          <span>By Jagrit Desk</span>
          
          <div className="flex items-center space-x-2">
            <button 
              onClick={() => setShowScore(!showScore)}
              className="text-gray-400 hover:text-black border border-gray-200 px-2 py-1 transition"
              title="Toggle Dev Mode (Show Rank Score)"
            >
              [DEV]
            </button>
            {showScore && (
              <span className="bg-black text-white px-2 py-1 font-mono normal-case">
                #{article.rank} ({article.score.toFixed(3)})
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default NewsCard;
