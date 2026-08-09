const mongoose = require('mongoose');

const articleCacheSchema = new mongoose.Schema({
  news_id: { type: String, required: true, unique: true },
  title_en: { type: String, required: true },
  title_hi: { type: String },
  abstract_en: { type: String },
  abstract_hi: { type: String },
  category: { type: String },
  translated_at: { type: Date }
}, { timestamps: true });

module.exports = mongoose.model('ArticleCache', articleCacheSchema);
