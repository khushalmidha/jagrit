const axios = require('axios');

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

const getRecommendations = async (mindUserId, topK = 10) => {
  try {
    const response = await axios.post(`${ML_SERVICE_URL}/recommend`, {
      user_id: mindUserId,
      top_k: topK
    }, { timeout: 5000 });
    
    return response.data; // { recommendations, latency_ms, is_cold_start, model_version }
  } catch (error) {
    console.error(`ML Service Error: ${error.message}`);
    // Circuit Breaker Lite: Fallback to trending/popular if ML service is down
    return {
      recommendations: _getFallbackArticles(topK),
      latency_ms: 0,
      is_cold_start: true,
      model_version: 'fallback'
    };
  }
};

// Fallback logic returns some hardcoded or cached popular MIND articles 
// so the frontend doesn't break if ML service dies.
const _getFallbackArticles = (topK) => {
  const fallbacks = [
    { news_id: 'N1000', title: 'Top News: Breaking Development', category: 'news', score: 1, rank: 1 },
    { news_id: 'N1001', title: 'Sports: Major Victory', category: 'sports', score: 0.9, rank: 2 },
    { news_id: 'N1002', title: 'Finance: Market Rally', category: 'finance', score: 0.8, rank: 3 },
    { news_id: 'N1003', title: 'Entertainment: New Release', category: 'entertainment', score: 0.7, rank: 4 },
  ];
  return fallbacks.slice(0, topK);
};

module.exports = {
  getRecommendations
};
