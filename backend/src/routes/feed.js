const express = require('express');
const auth = require('../middleware/auth');
const User = require('../models/User');
const Preference = require('../models/Preference');
const SavedArticle = require('../models/SavedArticle');
const mlServiceClient = require('../services/mlServiceClient');
const translationService = require('../services/translationService');

const router = new express.Router();

// Get Feed
router.get('/', auth, async (req, res) => {
  try {
    const user = await User.findById(req.userId);
    const prefs = await Preference.findOne({ userId: req.userId });
    
    // lang param overrides user pref
    const lang = req.query.lang || (prefs ? prefs.language : 'en');
    
    // Get recommendations from ML service
    const mlResponse = await mlServiceClient.getRecommendations(user.mind_user_id, 20);
    
    let articles = mlResponse.recommendations;
    
    // Translate if requested
    if (lang === 'hi') {
      articles = await translationService.translateFeed(articles);
    }
    
    res.json({
      feed: articles,
      is_cold_start: mlResponse.is_cold_start,
      model_version: mlResponse.model_version
    });
    
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Server error' });
  }
});

// Save Article
router.post('/save/:news_id', auth, async (req, res) => {
  try {
    const saved = new SavedArticle({
      userId: req.userId,
      news_id: req.params.news_id
    });
    await saved.save();
    res.status(201).json(saved);
  } catch (error) {
    if (error.code === 11000) {
      return res.status(400).json({ error: 'Article already saved' });
    }
    res.status(500).json({ error: 'Server error' });
  }
});

// Get Saved Articles
router.get('/saved', auth, async (req, res) => {
  try {
    const saved = await SavedArticle.find({ userId: req.userId }).sort({ saved_at: -1 });
    const newsIds = saved.map(s => s.news_id);
    
    let articles = await mlServiceClient.getArticleByIds(newsIds);
    
    const prefs = await Preference.findOne({ userId: req.userId });
    const lang = req.query.lang || (prefs ? prefs.language : 'en');
    
    if (lang === 'hi') {
      articles = await translationService.translateFeed(articles);
    }
    
    // Attach saved_at date to each article
    const response = articles.map((art, idx) => ({
      ...art,
      saved_at: saved[idx].saved_at
    }));
    
    res.json(response);
  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

module.exports = router;
