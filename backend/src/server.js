require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');

const authRouter = require('./routes/auth');
const preferencesRouter = require('./routes/preferences');
const feedRouter = require('./routes/feed');

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

app.use('/api/auth', authRouter);
app.use('/api/preferences', preferencesRouter);
app.use('/api/feed', feedRouter);

app.get('/health', (req, res) => {
  res.send({ status: 'healthy', db_state: mongoose.connection.readyState });
});

const start = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/jagrit');
    console.log('Connected to MongoDB');
    
    app.listen(PORT, () => {
      console.log(`Backend server listening on port ${PORT}`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
};

start();
