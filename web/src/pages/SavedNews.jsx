import React, { useContext, useEffect, useState } from 'react';
import axios from 'axios';
import NewsCard from '../components/NewsCard';
import FeedSkeleton from '../components/FeedSkeleton';
import { AuthContext } from '../context/AuthContext';
import { LanguageContext } from '../context/LanguageContext';
import { Link } from 'react-router-dom';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

const SavedNews = () => {
  const { user, token } = useContext(AuthContext);
  const { lang, t } = useContext(LanguageContext);
  const [feed, setFeed] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!token) {
      setLoading(false);
      return;
    }

    setLoading(true);
    axios.get(`${API_URL}/api/feed/saved?lang=${lang}`, {
      headers: { Authorization: `Bearer ${token}` }
    })
    .then(res => {
      setFeed(res.data);
    })
    .catch(console.error)
    .finally(() => setLoading(false));
  }, [token, lang]);

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <FeedSkeleton />
      </div>
    );
  }

  if (!token) {
    return (
      <div className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 py-20 text-center">
        <h2 className="text-3xl font-serif font-black mb-4 uppercase">{t('Login Required')}</h2>
        <p className="text-gray-500 font-serif mb-8">{t('Please login to view your saved articles.')}</p>
        <Link to="/login" className="bg-black text-white px-8 py-3 font-bold uppercase hover:bg-gray-800 transition">
          {t('Login')}
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Newspaper style header */}
      <div className="mb-10 flex flex-col items-center border-b-[6px] border-black pb-6 relative">
        <h1 className="text-6xl font-serif font-black text-gray-900 tracking-tight uppercase">{t('Saved News')}</h1>
        <p className="mt-3 text-gray-500 font-serif italic text-lg border-t border-b border-gray-300 py-1 px-4">
          {t('Your Personal Archive')}
        </p>
      </div>

      <div className="lg:w-3/4 mx-auto">
        {feed.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-12">
            {feed.map((article) => (
              <NewsCard 
                key={article.news_id} 
                article={article} 
                isHero={false} 
                token={token}
              />
            ))}
          </div>
        ) : (
          <div className="text-center py-20 text-gray-500 font-serif">
            {t('You have not saved any articles yet.')}
          </div>
        )}
      </div>
    </div>
  );
};

export default SavedNews;
