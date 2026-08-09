import React, { useContext, useEffect, useState } from 'react';
import axios from 'axios';
import NewsCard from '../components/NewsCard';
import FeedSkeleton from '../components/FeedSkeleton';
import { AuthContext } from '../context/AuthContext';
import { LanguageContext } from '../context/LanguageContext';
import { Link } from 'react-router-dom';

const ForYou = () => {
  const { user, token } = useContext(AuthContext);
  const { lang, t } = useContext(LanguageContext);
  const [feed, setFeed] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isColdStart, setIsColdStart] = useState(false);

  useEffect(() => {
    if (!token) {
      setLoading(false);
      return;
    }

    setLoading(true);
    axios.get(`http://localhost:5000/api/feed?lang=${lang}`, {
      headers: { Authorization: `Bearer ${token}` }
    })
    .then(res => {
      setFeed(res.data.feed);
      setIsColdStart(res.data.is_cold_start);
    })
    .catch(console.error)
    .finally(() => setLoading(false));
  }, [token, lang]);

  if (!user) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 text-center">
        <h2 className="text-3xl font-serif mb-4">Personalized News Awaits</h2>
        <p className="text-gray-600 mb-8">Please login to see your ML-ranked feed based on the MIND dataset.</p>
        <Link to="/login" className="bg-black text-white px-6 py-2 font-bold uppercase tracking-wide text-sm">
          {t('Login')}
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8 border-b-4 border-black inline-block">
        <h1 className="text-4xl font-serif font-bold text-gray-900 pb-1">{t('For You')}</h1>
      </div>

      {isColdStart && !loading && (
        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-8">
          <div className="flex">
            <div className="ml-3">
              <p className="text-sm text-yellow-700">
                Welcome to Jagrit! It looks like you're new here. We are showing you top trending stories. 
                <Link to="/preferences" className="font-bold underline ml-1">Tell us what you like</Link> to get better recommendations.
              </p>
            </div>
          </div>
        </div>
      )}

      {loading ? (
        <FeedSkeleton />
      ) : (
        <div className="magazine-grid">
          {feed.map((article, index) => (
            <NewsCard 
              key={article.news_id} 
              article={article} 
              isHero={index === 0} 
              token={token}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default ForYou;
