const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  name: { type: String },
  // For the purpose of the portfolio demo, we map a new Jagrit user to a real MIND user_id.
  // This allows the ML service to have real historical interactions to rank on.
  // If no specific mind_user_id is provided, it defaults to a cold start ID.
  mind_user_id: { type: String, default: 'NEW_USER_999' } 
}, { timestamps: true });

module.exports = mongoose.model('User', userSchema);
