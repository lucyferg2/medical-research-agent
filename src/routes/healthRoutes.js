const express = require('express');
const router = express.Router();

// Health check endpoint
router.get('/', (req, res) => {
  const healthcheck = {
    uptime: process.uptime(),
    message: 'Medical Research Agent API is healthy',
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV || 'development',
    version: process.env.npm_package_version || '1.0.0'
  };
  
  res.json(healthcheck);
});

// Detailed system status
router.get('/status', async (req, res) => {
  try {
    const status = {
      api: 'healthy',
      database: 'not_configured', // Update when you add actual database
      openai: process.env.OPENAI_API_KEY ? 'configured' : 'not_configured',
      pinecone: process.env.PINECONE_API_KEY ? 'configured' : 'not_configured',
      memory: {
        used: Math.round(process.memoryUsage().heapUsed / 1024 / 1024) + ' MB',
        total: Math.round(process.memoryUsage().heapTotal / 1024 / 1024) + ' MB'
      },
      uptime: Math.round(process.uptime()) + ' seconds'
    };
    
    res.json({
      success: true,
      status,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

module.exports = router;
