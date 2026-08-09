import React, { useState, useContext } from 'react';
import { Bookmark, BookmarkCheck } from 'lucide-react';
import axios from 'axios';
import { LanguageContext } from '../context/LanguageContext';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

const NewsCard = ({ article, isHero = false, token }) => {
  const [showScore, setShowScore] = useState(false);
  const [saved, setSaved] = useState(false);
  const { lang } = useContext(LanguageContext);

  // Generate a consistent pseudo-random image from picsum based on news_id
  // MIND dataset doesn't have images.
  const getImageUrl = (id) => {
    let hash = 0;
    for (let i = 0; i < id.length; i++) {
      hash = id.charCodeAt(i) + ((hash << 5) - hash);
    }
    const seed = Math.abs(hash) % 1000;
    return `https://picsum.photos/seed/${seed}/800/600`;
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
        <img src={getImageUrl(article.news_id)} alt={displayTitle} className="w-full h-full object-cover border border-gray-200" />
      </div>
      
      <div className={`flex flex-col justify-between ${isHero ? 'md:w-[40%] md:pl-8 py-2' : ''}`}>
        <div>
          <div className="flex justify-between items-center mb-3">
            <span className="text-[10px] font-bold text-red-600 uppercase tracking-widest border-b-2 border-red-600 pb-1">
              {article.category}
            </span>
            <button onClick={handleSave} className="text-gray-400 hover:text-black transition-colors">
              {saved ? <BookmarkCheck size={20} /> : <Bookmark size={20} />}
            </button>
          </div>
          
          <h2 className={`font-serif font-bold text-gray-900 leading-[1.1] mb-4 hover:text-red-700 cursor-pointer transition-colors ${isHero ? 'text-4xl md:text-5xl' : 'text-2xl'}`}>
            {displayTitle}
          </h2>
          
          <p className={`text-gray-600 font-serif leading-relaxed ${isHero ? 'text-lg line-clamp-4' : 'text-sm line-clamp-3'}`}>
            {displayAbstract}
          </p>
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
