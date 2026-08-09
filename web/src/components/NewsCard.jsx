import React, { useState } from 'react';
import { Bookmark, BookmarkCheck } from 'lucide-react';
import axios from 'axios';

const NewsCard = ({ article, isHero = false, token }) => {
  const [showScore, setShowScore] = useState(false);
  const [saved, setSaved] = useState(false);

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
      await axios.post(`http://localhost:5000/api/feed/save/${article.news_id}`, {}, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setSaved(true);
    } catch (error) {
      if (error.response?.status === 400) setSaved(true);
    }
  };

  return (
    <div className={`flex flex-col bg-white overflow-hidden rounded-md border border-gray-100 hover:shadow-lg transition-shadow duration-300 ${isHero ? 'hero-card md:flex-row' : ''}`}>
      <div className={`relative ${isHero ? 'md:w-2/3 h-64 md:h-auto' : 'h-48'}`}>
        <img src={getImageUrl(article.news_id)} alt={article.title} className="w-full h-full object-cover" />
        <div className="absolute top-2 left-2 bg-black bg-opacity-70 text-white text-xs font-bold px-2 py-1 uppercase tracking-wider">
          {article.category}
        </div>
      </div>
      
      <div className={`p-5 flex flex-col justify-between flex-grow ${isHero ? 'md:w-1/3' : ''}`}>
        <div>
          <div className="flex justify-between items-start mb-2">
            <span className="text-xs font-bold text-red-600 uppercase tracking-wider">MIND News</span>
            <button onClick={handleSave} className="text-gray-400 hover:text-black">
              {saved ? <BookmarkCheck size={18} /> : <Bookmark size={18} />}
            </button>
          </div>
          
          <h2 className={`font-serif font-bold leading-tight mb-3 text-gray-900 ${isHero ? 'text-3xl' : 'text-xl'}`}>
            {article.title}
          </h2>
          
          <p className="text-gray-600 text-sm line-clamp-3 font-sans">
            {article.abstract || 'Read the full story to know more about this breaking development.'}
          </p>
        </div>
        
        <div className="mt-4 pt-4 border-t border-gray-100 flex justify-between items-center text-xs text-gray-500">
          <span>By Jagrit Editorial</span>
          
          <div className="flex items-center space-x-2">
            <button 
              onClick={() => setShowScore(!showScore)}
              className="text-blue-500 hover:underline text-[10px]"
              title="Toggle Dev Mode (Show Rank Score)"
            >
              [DEV]
            </button>
            {showScore && (
              <span className="bg-blue-50 text-blue-700 px-2 py-1 rounded font-mono">
                Rank #{article.rank} ({article.score.toFixed(3)})
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default NewsCard;
