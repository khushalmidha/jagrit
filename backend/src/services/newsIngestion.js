const axios = require('axios');
const Redis = require('ioredis');
const Parser = require('rss-parser');
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
const parser = new Parser({
  customFields: {
    item: ['media:content', 'media:thumbnail', 'enclosure']
  }
});

// Reliable RSS Feeds for free, unlimited, long-description news
const RSS_FEEDS = [
  { url: 'https://timesofindia.indiatimes.com/rssfeedstopstories.cms', source: 'Times of India', category: 'news' },
  { url: 'https://www.thehindu.com/news/national/feeder/default.rss', source: 'The Hindu', category: 'politics' },
  { url: 'https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en', source: 'Google News India', category: 'news' },
  { url: 'http://feeds.bbci.co.uk/news/world/rss.xml', source: 'BBC World', category: 'world' },
  { url: 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664', source: 'CNBC', category: 'finance' },
  { url: 'https://feeds.a.dj.com/rss/RSSWSJD.xml', source: 'Wall Street Journal', category: 'technology' },
  { url: 'https://www.espncricinfo.com/rss/content/story/feeds/0.xml', source: 'ESPN Cricinfo', category: 'sports' }
];

/**
 * Fetch top headlines from NewsAPI & RSS Feeds and ingest them into Redis for the ML Service.
 */
const ingestLiveNews = async () => {
  try {
    // Check cooldown to avoid hitting limits (2 minutes)
    const lastFetch = await redis.get('last_news_fetch');
    if (lastFetch) {
      console.log("News fetched recently. Skipping to avoid API limits.");
      return 0; // indicates skipped
    }

    console.log("Fetching live news from multiple sources...");
    let allArticles = [];
    
    // 1. Fetch from NewsAPI (if key exists)
    if (NEWS_API_KEY) {
      console.log("Using NewsAPI key to fetch diverse categories...");
      const categoriesToFetch = ['general', 'technology', 'sports', 'business', 'entertainment'];
      try {
        const newsApiPromises = categoriesToFetch.map(cat => 
          axios.get(`https://newsapi.org/v2/top-headlines?country=in&category=${cat}&apiKey=${NEWS_API_KEY}`)
            .then(res => {
              if (!res.data || !res.data.articles) return [];
              return res.data.articles
                .filter(a => a.title && a.description)
                .map(a => ({
                  title: a.title,
                  abstract: a.description,
                  category: cat === 'general' ? 'news' : cat,
                  url: a.url,
                  source: a.source.name,
                  image_url: a.urlToImage || ""
                }));
            })
            .catch(err => {
              console.error(`NewsAPI fetch failed for ${cat}:`, err.message);
              return [];
            })
        );
        
        const results = await Promise.all(newsApiPromises);
        results.forEach(articles => {
          allArticles = [...allArticles, ...articles];
        });
        console.log(`Fetched articles from NewsAPI across ${categoriesToFetch.length} categories.`);
      } catch (e) {
        console.error("NewsAPI overall fetch failed:", e.message);
      }
    }

    // 2. Fetch from RSS Feeds
    const rssPromises = RSS_FEEDS.map(async (feedInfo) => {
      try {
        const feed = await parser.parseURL(feedInfo.url);
        // Take top 10 from each feed to prevent overload
        const items = feed.items.slice(0, 10).filter(item => item.title && (item.contentSnippet || item.content));
        
        return items.map(item => {
          let imageUrl = "";
          if (item.enclosure && item.enclosure.url) {
            imageUrl = item.enclosure.url;
          } else if (item['media:content'] && item['media:content']['$'] && item['media:content']['$'].url) {
            imageUrl = item['media:content']['$'].url;
          } else if (item['media:thumbnail'] && item['media:thumbnail']['$'] && item['media:thumbnail']['$'].url) {
            imageUrl = item['media:thumbnail']['$'].url;
          } else {
            // Regex to find first image tag in content
            const imgMatch = (item.content || '').match(/<img[^>]+src="([^">]+)"/);
            if (imgMatch) imageUrl = imgMatch[1];
          }

          return {
            title: item.title,
            abstract: item.contentSnippet || item.content || item.summary || "",
            category: feedInfo.category,
            url: item.link,
            source: feedInfo.source,
            image_url: imageUrl
          };
        });
      } catch (e) {
        console.error(`RSS fetch failed for ${feedInfo.source}:`, e.message);
        return [];
      }
    });

    const rssResults = await Promise.all(rssPromises);
    rssResults.forEach(rssArticles => {
      allArticles = [...allArticles, ...rssArticles];
    });

    console.log(`Fetched total ${allArticles.length} combined articles. Formatting...`);

    if (allArticles.length === 0) {
      console.log("No articles found from any source.");
      return 0;
    }

    // Format and deduplicate
    const uniqueUrls = new Set();
    const formattedArticles = [];
    
    for (const a of allArticles) {
      if (uniqueUrls.has(a.url)) continue;
      uniqueUrls.add(a.url);
      
      const hash = crypto.createHash('md5').update(a.url).digest('hex').substring(0, 8);
      formattedArticles.push({
        ...a,
        news_id: `LIVE_${hash}`
      });
    }

    // Translate to Hindi via Gemini (and cache in Mongo)
    console.log(`Translating ${formattedArticles.length} unique articles to Hindi...`);
    const translatedArticles = await translateFeed(formattedArticles);

    // Push to Redis for the ML Candidate Generator
    const pipeline = redis.pipeline();
    
    // Clear old recent news so only fresh news appears
    pipeline.del('news:recent');
    
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
      // Expire hash after 24 hours to prevent memory bloat
      pipeline.expire(`article:${article.news_id}`, 86400);
      
      // Add to a sorted set of recent news (score is timestamp)
      pipeline.zadd('news:recent', Date.now(), article.news_id);
    }
    
    await pipeline.exec();
    
    // Set 1-minute cooldown (60 seconds)
    await redis.set('last_news_fetch', Date.now(), 'EX', 60);
    
    console.log("Live news successfully ingested into Redis!");
    
    return translatedArticles.length;
    
  } catch (error) {
    console.error("Error during news ingestion:", error.message);
    return 0;
  }
};

module.exports = {
  ingestLiveNews
};
