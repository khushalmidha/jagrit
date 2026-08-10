import React, { useState, useContext } from 'react';
import { Bookmark, BookmarkCheck, ChevronDown, ChevronUp } from 'lucide-react';
import axios from 'axios';
import { LanguageContext } from '../context/LanguageContext';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

  // Return local fixed image based on category
  const getImageUrl = (category, newsId) => {
    const cat = String(category || '').toLowerCase();
    
    // Choose between image 1 and 2 based on newsId hash
    let hash = 0;
    const str = String(newsId || '');
    for (let i = 0; i < str.length; i++) {
      hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }
    const index = (Math.abs(hash) % 2) + 1; // either 1 or 2
    
    // Check if category has images, otherwise default
    const validCats = ['technology', 'sports', 'finance', 'business', 'politics', 'education', 'world', 'entertainment'];
    const selectedCat = validCats.includes(cat) ? cat : 'default';
    
    return `/images/categories/${selectedCat}-${index}.jpg`;
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
        <img 
          src={article.image_url || getImageUrl(article.category, article.news_id)} 
          onError={(e) => {
            if (e.target.src !== getImageUrl(article.category, article.news_id)) {
              e.target.src = getImageUrl(article.category, article.news_id);
            }
          }}
          alt={displayTitle} 
          className="w-full h-full object-cover border border-gray-200" 
        />
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
