const express = require('express');
const { ingestLiveNews } = require('../services/newsIngestion');
const router = new express.Router();

// Trigger news ingestion manually (In production, secure this with an admin token)
router.post('/ingest-news', async (req, res) => {
  const secret = process.env.ADMIN_SECRET || 'demo-secret';
  if (req.headers['x-admin-secret'] !== secret) {
    return res.status(401).json({ error: 'Unauthorized: Invalid admin secret' });
  }
  
  try {
    const count = await ingestLiveNews();
    res.json({ message: 'Ingestion completed successfully', articles_processed: count });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
