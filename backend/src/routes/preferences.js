const express = require('express');
const Preference = require('../models/Preference');
const auth = require('../middleware/auth');
const router = new express.Router();

// Get preferences
router.get('/', auth, async (req, res) => {
  try {
    const prefs = await Preference.findOne({ userId: req.userId });
    if (!prefs) {
      return res.status(404).json({ error: 'Preferences not found' });
    }
    res.json(prefs);
  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Update preferences (categories, language)
router.patch('/', auth, async (req, res) => {
  try {
    const { preferred_categories, language } = req.body;
    const updates = {};
    if (preferred_categories) updates.preferred_categories = preferred_categories;
    if (language) updates.language = language;
    
    const prefs = await Preference.findOneAndUpdate(
      { userId: req.userId },
      { $set: updates },
      { new: true, runValidators: true }
    );
    
    res.json(prefs);
  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

module.exports = router;
