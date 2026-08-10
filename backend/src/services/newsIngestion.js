const axios = require('axios');
const Redis = require('ioredis');
const { translateFeed } = require('./translationService');
const crypto = require('crypto');

const redisOptions = process.env.REDIS_URL && process.env.REDIS_URL.startsWith('rediss://') 
  ? { tls: { rejectUnauthorized: false } } 
  : {};

const redis = process.env.REDIS_URL 
  ? new Redis(process.env.REDIS_URL, redisOptions) 
  : new Redis({ host: 'localhost', port: 6379 });

redis.on('error', (err) => {
  console.error('Redis connection error:', err.message);
});

const NEWS_API_KEY = process.env.NEWS_API_KEY;

/**
 * Fetch top headlines from NewsAPI and ingest them into Redis for the ML Service.
 */
const ingestLiveNews = async () => {
  if (!NEWS_API_KEY) {
    console.error("NEWS_API_KEY is not set. Cannot fetch live news.");
    return;
  }

  try {
    // Check cooldown to avoid hitting NewsAPI limits (2 minutes)
    const lastFetch = await redis.get('last_news_fetch');
    if (lastFetch) {
      console.log("News fetched recently. Skipping to avoid API limits.");
      return 0; // indicates skipped
    }

    console.log("Fetching live news from NewsAPI...");
    
    // Fetch US top headlines as a baseline
    const response = await axios.get(`https://newsapi.org/v2/top-headlines?country=us&apiKey=${NEWS_API_KEY}`);
    const articles = response.data.articles;
    
    if (!articles || articles.length === 0) {
      console.log("No articles found.");
      return;
    }

    console.log(`Fetched ${articles.length} articles. Processing...`);

    const formattedArticles = articles.filter(a => a.title && a.description).map(a => {
      // Create a deterministic hash for news_id
      const hash = crypto.createHash('md5').update(a.url).digest('hex').substring(0, 8);
      const newsId = `LIVE_${hash}`;
      
      // Attempt to guess category based on source or just default to general
      let category = 'news';
      if (a.url.includes('tech') || a.url.includes('verge')) category = 'technology';
      else if (a.url.includes('sport') || a.url.includes('espn')) category = 'sports';
      else if (a.url.includes('finance') || a.url.includes('bloomberg')) category = 'finance';

      return {
        news_id: newsId,
        title: a.title,
        abstract: a.description,
        category: category,
        url: a.url,
        source: a.source.name,
        image_url: a.urlToImage || ""
      };
    });

    // Translate to Hindi via Gemini (and cache in Mongo)
    console.log("Translating articles to Hindi...");
    const translatedArticles = await translateFeed(formattedArticles);

    // Push to Redis for the ML Candidate Generator
    const pipeline = redis.pipeline();
    
    for (const article of translatedArticles) {
      // Store article details in a Hash
      pipeline.hset(`article:${article.news_id}`, {
        title: article.title,
        abstract: article.abstract || "",
        category: article.category,
        url: article.url || "",
        source: article.source || "",
        image_url: article.image_url || "",
        title_hi: article.title_hi || "",
        abstract_hi: article.abstract_hi || "",
        impressions: 0,
        clicks: 0,
        timestamp: Date.now()
      });
      
      // Add to a sorted set of recent news (score is timestamp)
      pipeline.zadd('news:recent', Date.now(), article.news_id);
    }
    
    // Keep only the latest 1000 articles in the recent set to prevent bloat
    pipeline.zremrangebyrank('news:recent', 0, -1001);
    
    await pipeline.exec();
    
    // Set 2-minute cooldown (120 seconds)
    await redis.set('last_news_fetch', Date.now(), 'EX', 120);
    
    console.log("Live news successfully ingested into Redis!");
    
    return translatedArticles.length;
    
  } catch (error) {
    console.error("Error during news ingestion:", error.message);
  }
};

module.exports = {
  ingestLiveNews
};
