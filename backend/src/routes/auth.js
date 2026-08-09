const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const User = require('../models/User');
const Preference = require('../models/Preference');
const auth = require('../middleware/auth');
const router = new express.Router();

// Register
router.post('/register', async (req, res) => {
  try {
    const { email, password, name, mind_user_id } = req.body;
    
    // Check if user exists
    let user = await User.findOne({ email });
    if (user) {
      return res.status(400).json({ error: 'User already exists' });
    }
    
    // Hash password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);
    
    // Create User
    user = new User({
      email,
      password: hashedPassword,
      name,
      mind_user_id: mind_user_id || 'NEW_USER_999' // Use provided MIND ID or default
    });
    await user.save();
    
    // Create empty preferences
    const prefs = new Preference({ userId: user._id });
    await prefs.save();
    
    // Generate Token
    const token = jwt.sign({ userId: user._id }, process.env.JWT_SECRET || 'fallback_secret_for_dev', { expiresIn: '7d' });
    
    res.status(201).json({ user: { id: user._id, email: user.email, name: user.name, mind_user_id: user.mind_user_id }, token });
  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Login
router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    
    // Find user
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(400).json({ error: 'Invalid credentials' });
    }
    
    // Verify password
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).json({ error: 'Invalid credentials' });
    }
    
    // Generate Token
    const token = jwt.sign({ userId: user._id }, process.env.JWT_SECRET || 'fallback_secret_for_dev', { expiresIn: '7d' });
    
    res.json({ user: { id: user._id, email: user.email, name: user.name, mind_user_id: user.mind_user_id }, token });
  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Get Me
router.get('/me', auth, async (req, res) => {
  try {
    const user = await User.findById(req.userId).select('-password');
    res.json(user);
  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

module.exports = router;
