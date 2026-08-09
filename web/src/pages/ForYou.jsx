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

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <FeedSkeleton />
      </div>
    );
  }

  // Fallback to mock data if the backend/ML service is down and feed is empty
  const mockFeed = [
    { 
      news_id: 'M1', 
      title: lang === 'hi' ? 'टेक शेयरों में उछाल के साथ वैश्विक बाजारों में तेजी' : 'Global Markets Rally as Tech Stocks Surge', 
      abstract: lang === 'hi' ? 'प्रमुख टेक कंपनियों की बेहतर आय रिपोर्ट के बाद आज प्रमुख सूचकांक नई ऊंचाइयों पर पहुंच गए।' : 'Major indices hit new highs today following better-than-expected earnings reports from leading tech companies.', 
      category: 'business', rank: 1, score: 0.95, isMock: true 
    },
    { 
      news_id: 'M2', 
      title: lang === 'hi' ? 'नवीकरणीय ऊर्जा तकनीक में नई सफलता' : 'New Breakthrough in Renewable Energy Tech', 
      abstract: lang === 'hi' ? 'वैज्ञानिकों ने एक नया सोलर पैनल डिजाइन विकसित किया है जो दक्षता को 30% तक बढ़ाता है।' : 'Scientists have developed a new solar panel design that increases efficiency by 30%, potentially revolutionizing the green energy sector.', 
      category: 'technology', rank: 2, score: 0.91, isMock: true 
    },
    { 
      news_id: 'M3', 
      title: lang === 'hi' ? 'विश्व कप फाइनल: एक अविस्मरणीय मैच' : 'World Cup Finals: An Unforgettable Match', 
      abstract: lang === 'hi' ? 'रोमांचक फाइनल अतिरिक्त समय तक गया, जिसका अंत एक शानदार गोल के साथ हुआ।' : 'The thrilling finale went to extra time, ending with a spectacular goal that will be remembered for decades.', 
      category: 'sports', rank: 3, score: 0.88, isMock: true 
    },
    { 
      news_id: 'M4', 
      title: lang === 'hi' ? 'चुनाव 2026: आपको क्या जानना चाहिए' : 'Elections 2026: What You Need to Know', 
      abstract: lang === 'hi' ? 'आगामी चुनावों, प्रमुख उम्मीदवारों और राष्ट्र को आकार देने वाली प्रमुख नीतिगत बहसों का व्यापक विवरण।' : 'A comprehensive breakdown of the upcoming elections, key candidates, and major policy debates shaping the nation.', 
      category: 'politics', rank: 4, score: 0.85, isMock: true 
    },
    { 
      news_id: 'M5', 
      title: lang === 'hi' ? 'स्वास्थ्य सेवा में AI का भविष्य' : 'The Future of AI in Healthcare', 
      abstract: lang === 'hi' ? 'कैसे कृत्रिम बुद्धिमत्ता दवा खोज को तेज कर रही है और दुनिया भर में रोगी देखभाल को व्यक्तिगत बना रही है।' : 'How artificial intelligence is accelerating drug discovery and personalizing patient care around the globe.', 
      category: 'technology', rank: 5, score: 0.82, isMock: true 
    }
  ];
  const displayFeed = feed.length > 0 ? feed : mockFeed;

  return (
    <div className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Newspaper style header for the section */}
      <div className="mb-10 flex flex-col items-center border-b-[6px] border-black pb-6">
        <h1 className="text-6xl font-serif font-black text-gray-900 tracking-tight uppercase">{t('For You')}</h1>
        <p className="mt-3 text-gray-500 font-serif italic text-lg border-t border-b border-gray-300 py-1 px-4">
          {t('Personalized Edition • Powered by Jagrit AI')}
        </p>
      </div>

      {isColdStart && (
        <div className="bg-white border border-black p-4 mb-10 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] flex items-center justify-between">
          <div>
            <span className="font-bold uppercase tracking-wider text-sm mr-2 text-red-600">Notice:</span>
            <span className="text-gray-800 font-serif">You are currently viewing trending stories. Log in or update preferences for ML personalization.</span>
          </div>
          <Link to="/login" className="bg-black text-white px-4 py-1 text-xs font-bold uppercase hover:bg-gray-800 transition">
            {t('Login')}
          </Link>
        </div>
      )}

      {feed.length === 0 && (
        <div className="bg-yellow-50 border border-yellow-400 text-yellow-800 p-4 mb-10 text-center font-bold uppercase tracking-wider text-xs">
          ⚠️ Backend ML Service Unreachable — Displaying Unverified Mock Data
        </div>
      )}

      {/* Editorial Grid Layout */}
      <div className="flex flex-col lg:flex-row gap-8">
        {/* Main Content Area */}
        <div className="lg:w-3/4">
          {displayFeed.length > 0 && (
            <div className="mb-10">
              <NewsCard article={displayFeed[0]} isHero={true} token={token} />
            </div>
          )}
          
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-x-8 gap-y-12 border-t border-gray-200 pt-8">
            {displayFeed.slice(1).map((article) => (
              <NewsCard 
                key={article.news_id} 
                article={article} 
                isHero={false} 
                token={token}
              />
            ))}
          </div>
        </div>

        {/* Sidebar */}
        <div className="lg:w-1/4 space-y-8">
          <div className="bg-gray-50 p-6 border-t-4 border-black">
            <h3 className="font-serif font-black text-xl mb-4 uppercase">{t('TOP TOPICS')}</h3>
            <ul className="space-y-3 font-serif">
              {['Politics', 'World', 'Business', 'Technology', 'Sports'].map(topic => (
                <li key={topic} className="flex justify-between items-center border-b border-gray-200 pb-2">
                  <a href="#" className="hover:text-red-600 transition">{t(topic)}</a>
                  <span className="text-gray-400 text-xs">»</span>
                </li>
              ))}
            </ul>
          </div>
          
          <div className="border border-gray-200 p-6 text-center">
            <h3 className="font-serif font-bold text-lg mb-2">{t('Jagrit Newsletter')}</h3>
            <p className="text-gray-500 text-sm mb-4">{t('Get the top stories delivered to your inbox daily.')}</p>
            <input type="email" placeholder={t('Your Email')} className="w-full border border-gray-300 px-3 py-2 text-sm mb-3 focus:outline-none focus:border-black" />
            <button className="w-full bg-black text-white py-2 text-xs font-bold uppercase tracking-wider hover:bg-gray-800 transition">{t('Subscribe')}</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ForYou;
