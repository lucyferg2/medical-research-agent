const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const morgan = require('morgan');
const rateLimit = require('express-rate-limit');
require('dotenv').config();

const { SequentialWorkflowOrchestrator } = require('./src/orchestrator/SequentialWorkflowOrchestrator');
const { validateInput, generateSessionId } = require('./src/utils/helpers');
const agentRoutes = require('./src/routes/agentRoutes');
const workflowRoutes = require('./src/routes/workflowRoutes');
const healthRoutes = require('./src/routes/healthRoutes');

const app = express();
const PORT = process.env.PORT || 3000;

// Security middleware
app.use(helmet());
app.use(compression());
app.use(morgan('combined'));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.'
});
app.use('/api/', limiter);

// CORS configuration
app.use(cors({
  origin: process.env.NODE_ENV === 'production' 
    ? ['https://chat.openai.com', 'https://chatgpt.com']
    : ['http://localhost:3000', 'https://chat.openai.com'],
  credentials: true
}));

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/api/health', healthRoutes);
app.use('/api/agents', agentRoutes);
app.use('/api/workflow', workflowRoutes);

// Main sequential workflow endpoint
app.post('/api/sequential-workflow', async (req, res) => {
  try {
    const { query, sessionId = generateSessionId(), reportType = 'comprehensive' } = req.body;
    
    // Input validation
    const validation = await validateInput(query);
    if (!validation.isValid) {
      return res.status(400).json({ 
        success: false, 
        error: 'Invalid input', 
        details: validation.errors 
      });
    }

    const orchestrator = new SequentialWorkflowOrchestrator();
    const result = await orchestrator.executeSequentialWorkflow(sessionId, query, reportType);
    
    if (result.success) {
      res.json({
        success: true,
        sessionId: result.sessionId,
        executionTime: result.executionTime,
        agentConsultations: result.workflow.agents.length,
        report: result.finalReport,
        workflow: {
          steps: result.workflow.agents.map(agent => ({
            agent: agent.agent,
            timestamp: agent.timestamp,
            keyFindings: agent.results.summary || agent.results.keyPoints || 'Processing completed'
          })),
          cumulativeContext: result.workflow.cumulativeContext
        }
      });
    } else {
      res.status(500).json({
        success: false,
        error: result.error,
        partialResults: result.partialWorkflow?.agents || []
      });
    }
    
  } catch (error) {
    console.error('Sequential workflow error:', error);
    res.status(500).json({ 
      success: false, 
      error: 'Internal server error',
      message: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong'
    });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    success: false,
    error: 'Something went wrong!',
    message: process.env.NODE_ENV === 'development' ? err.message : 'Internal server error'
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found'
  });
});

app.listen(PORT, () => {
  console.log(`Medical Research Agent server running on port ${PORT}`);
  console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
});

app.get('/', (req, res) => {
  res.json({
    message: 'Medical Research Agent API is running!',
    version: '1.0.0',
    endpoints: [
      '/api/health',
      '/api/agents/vector-search',
      '/api/agents/literature-analysis',
      '/api/agents/clinical-trials',
      '/api/agents/competitive-intel',
      '/api/agents/regulatory-analysis',
      '/api/agents/medical-writing',
      '/api/sequential-workflow'
    ]
  });
});

module.exports = app;
