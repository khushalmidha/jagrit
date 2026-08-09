const mongoose = require('mongoose');

const savedArticleSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  news_id: { type: String, required: true },
  saved_at: { type: Date, default: Date.now }
}, { timestamps: true });

savedArticleSchema.index({ userId: 1, news_id: 1 }, { unique: true });

module.exports = mongoose.model('SavedArticle', savedArticleSchema);
