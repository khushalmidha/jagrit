const mongoose = require('mongoose');

const preferenceSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true, unique: true },
  preferred_categories: [{ type: String }],
  language: { type: String, enum: ['en', 'hi'], default: 'en' },
  reading_history: [{
    news_id: String,
    timestamp: { type: Date, default: Date.now }
  }]
}, { timestamps: true });

module.exports = mongoose.model('Preference', preferenceSchema);
