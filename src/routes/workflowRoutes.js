const express = require('express');
const router = express.Router();
const { SequentialWorkflowOrchestrator } = require('../orchestrator/SequentialWorkflowOrchestrator');

const orchestrator = new SequentialWorkflowOrchestrator();

// Get workflow status
router.get('/status/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const workflow = orchestrator.getWorkflowStatus(sessionId);
    
    if (!workflow) {
      return res.status(404).json({ 
        success: false, 
        error: 'Workflow not found' 
      });
    }
    
    res.json({
      success: true,
      sessionId,
      status: workflow.error ? 'failed' : workflow.finalReport ? 'completed' : 'running',
      progress: {
        totalSteps: 6,
        completedSteps: workflow.agents.length,
        currentStep: workflow.agents.length < 6 ? workflow.agents.length + 1 : 6,
        stepNames: [
          'Vector Search',
          'Literature Analysis', 
          'Clinical Trials',
          'Competitive Intelligence',
          'Regulatory Analysis',
          'Medical Writing'
        ]
      },
      agents: workflow.agents.map(agent => ({
        name: agent.agent,
        status: 'completed',
        timestamp: agent.timestamp,
        duration: agent.duration || null
      })),
      startTime: workflow.startTime,
      error: workflow.error || null
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get all workflows
router.get('/list', async (req, res) => {
  try {
    const workflows = orchestrator.getAllWorkflows();
    res.json({
      success: true,
      workflows: workflows.map(w => ({
        sessionId: w.sessionId,
        status: w.status,
        startTime: w.startTime,
        progress: `${w.completedSteps}/${w.totalSteps}`,
        completionRate: Math.round((w.completedSteps / w.totalSteps) * 100)
      }))
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Cancel workflow
router.post('/cancel/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const workflow = orchestrator.getWorkflowStatus(sessionId);
    
    if (!workflow) {
      return res.status(404).json({ 
        success: false, 
        error: 'Workflow not found' 
      });
    }
    
    if (workflow.status === 'completed') {
      return res.status(400).json({ 
        success: false, 
        error: 'Cannot cancel completed workflow' 
      });
    }
    
    // Mark as cancelled (in a real implementation, you'd stop the actual processing)
    workflow.status = 'cancelled';
    workflow.error = 'Cancelled by user';
    
    res.json({
      success: true,
      message: 'Workflow cancelled successfully',
      sessionId
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

module.exports = router;
