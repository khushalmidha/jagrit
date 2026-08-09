const { GoogleGenerativeAI } = require('@google/generative-ai');
const ArticleCache = require('../models/ArticleCache');

const apiKey = process.env.GEMINI_API_KEY;
let genAI;
let model;

if (apiKey) {
  genAI = new GoogleGenerativeAI(apiKey);
  model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });
} else {
  console.warn("GEMINI_API_KEY is not set. Translation will be mocked or fail.");
}

const translateArticle = async (article) => {
  try {
    // 1. Check Cache
    let cache = await ArticleCache.findOne({ news_id: article.news_id });
    if (cache && cache.title_hi) {
      return {
        ...article,
        title: cache.title_hi,
        abstract: cache.abstract_hi || article.abstract
      };
    }
    
    // 2. Perform translation async (don't block heavily if not needed, but here we wait)
    if (!model) {
      throw new Error("Gemini not configured.");
    }
    
    const prompt = `Translate the following news headline and abstract from English to Hindi. Keep it professional.
    Headline: ${article.title}
    Abstract: ${article.abstract || ''}
    
    Return ONLY a JSON object with keys "title_hi" and "abstract_hi".`;
    
    const result = await model.generateContent(prompt);
    const responseText = result.response.text();
    
    // Clean up potential markdown formatting in Gemini response
    const jsonStr = responseText.replace(/```json/g, '').replace(/```/g, '').trim();
    const parsed = JSON.parse(jsonStr);
    
    // 3. Save to cache
    if (!cache) {
      cache = new ArticleCache({
        news_id: article.news_id,
        title_en: article.title,
        abstract_en: article.abstract,
        category: article.category
      });
    }
    
    cache.title_hi = parsed.title_hi;
    cache.abstract_hi = parsed.abstract_hi;
    cache.translated_at = new Date();
    await cache.save();
    
    return {
      ...article,
      title: parsed.title_hi,
      abstract: parsed.abstract_hi || article.abstract
    };
    
  } catch (error) {
    console.error(`Translation error for ${article.news_id}:`, error.message);
    // Return original English if translation fails
    return article;
  }
};

const translateFeed = async (articles) => {
  // Translate multiple articles. In a real system, we might batch this.
  // For demo, we map concurrently.
  const translated = await Promise.all(articles.map(a => translateArticle(a)));
  return translated;
};

module.exports = {
  translateFeed,
  translateArticle
};
