const express = require('express');
const { ingestLiveNews } = require('../services/newsIngestion');
const router = new express.Router();

// Trigger news ingestion manually (In production, secure this with an admin token)
router.post('/ingest-news', async (req, res) => {
  try {
    const count = await ingestLiveNews();
    res.json({ message: 'Ingestion completed successfully', articles_processed: count });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
