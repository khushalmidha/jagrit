const axios = require('axios');
const Redis = require('ioredis');

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

const redisOptions = process.env.REDIS_URL && process.env.REDIS_URL.startsWith('rediss://') 
  ? { tls: { rejectUnauthorized: false } } 
  : {};

const redis = process.env.REDIS_URL 
  ? new Redis(process.env.REDIS_URL, redisOptions) 
  : new Redis({ host: 'localhost', port: 6379 });

redis.on('error', (err) => {
  console.error('Redis error in mlServiceClient:', err.message);
});

const getRecommendations = async (mindUserId, topK = 15) => {
  try {
    const response = await axios.post(`${ML_SERVICE_URL}/recommend`, {
      user_id: mindUserId,
      top_k: topK
    }, { timeout: 5000 });
    
    return response.data; // { recommendations, latency_ms, is_cold_start, model_version }
  } catch (error) {
    console.error(`ML Service Error: ${error.message}`);
    // Circuit Breaker Lite: Fallback to Live Redis or popular if ML service is down
    const fallbackRecs = await _getFallbackArticles(topK);
    return {
      recommendations: fallbackRecs,
      latency_ms: 0,
      is_cold_start: true,
      model_version: 'fallback-redis'
    };
  }
};

// Fallback logic reads from Redis live news so user still gets news if ML service is down
const _getFallbackArticles = async (topK) => {
  try {
    // Get recent news IDs from Redis
    const liveIds = await redis.zrevrange('news:recent', 0, topK - 1);
    
    if (liveIds && liveIds.length > 0) {
      const pipeline = redis.pipeline();
      liveIds.forEach(id => pipeline.hgetall(`article:${id}`));
      const results = await pipeline.exec();
      
      return results.map((result, idx) => {
        const data = result[1]; // ioredis pipeline returns [error, result]
        return {
          news_id: liveIds[idx],
          title: data.title || 'Live News',
          abstract: data.abstract || '',
          category: data.category || 'news',
          score: 1.0 / (idx + 1),
          rank: idx + 1
        };
      });
    }
  } catch (err) {
    console.error("Redis fallback error:", err.message);
  }

  // Absolute fallback if Redis is empty or errors
  const fallbacks = [];
  for (let i = 1; i <= topK; i++) {
    fallbacks.push({
      news_id: `N100${i}`,
      title: `Global News Update ${i}`,
      abstract: 'Read the latest updates from around the globe.',
      category: i % 2 === 0 ? 'technology' : 'world',
      score: 1.0 / i,
      rank: i
    });
  }
  return fallbacks;
};

module.exports = {
  getRecommendations
};
