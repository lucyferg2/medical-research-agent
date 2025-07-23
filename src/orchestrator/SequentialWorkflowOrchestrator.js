const { v4: uuidv4 } = require('uuid');
const { VectorSearchAgent } = require('../agents/VectorSearchAgent');
const { LiteratureAnalysisAgent } = require('../agents/LiteratureAnalysisAgent');
const { ClinicalTrialsAgent } = require('../agents/ClinicalTrialsAgent');
const { CompetitiveIntelligenceAgent } = require('../agents/CompetitiveIntelligenceAgent');
const { RegulatoryAnalysisAgent } = require('../agents/RegulatoryAnalysisAgent');
const { MedicalWritingAgent } = require('../agents/MedicalWritingAgent');
const { ContextBuilder } = require('../utils/ContextBuilder');

class SequentialWorkflowOrchestrator {
  constructor() {
    this.workflowState = new Map();
    this.contextBuilder = new ContextBuilder();
    
    // Initialize agents
    this.agents = {
      vectorSearch: new VectorSearchAgent(),
      literatureAnalysis: new LiteratureAnalysisAgent(),
      clinicalTrials: new ClinicalTrialsAgent(),
      competitiveIntelligence: new CompetitiveIntelligenceAgent(),
      regulatoryAnalysis: new RegulatoryAnalysisAgent(),
      medicalWriting: new MedicalWritingAgent()
    };
  }

  async executeSequentialWorkflow(sessionId, initialQuery, reportType = 'comprehensive') {
    const workflow = {
      sessionId,
      initialQuery,
      reportType,
      startTime: new Date(),
      agents: [],
      cumulativeContext: {},
      finalReport: null,
      status: 'running'
    };

    this.workflowState.set(sessionId, workflow);

    try {
      console.log(`Starting sequential workflow for session: ${sessionId}`);

      // Step 1: Vector Search Agent
      console.log('Step 1: Vector Search Agent');
      const vectorResults = await this.runVectorSearch(initialQuery, workflow);
      workflow.agents.push({
        agent: 'vector_search',
        query: initialQuery,
        results: vectorResults,
        timestamp: new Date()
      });
      workflow.cumulativeContext.vector = vectorResults;

      // Step 2: Literature Analysis Agent  
      console.log('Step 2: Literature Analysis Agent');
      const literatureQuery = this.buildLiteratureQuery(initialQuery, vectorResults);
      const literatureResults = await this.runLiteratureAnalysis(literatureQuery, workflow.cumulativeContext);
      workflow.agents.push({
        agent: 'literature_analysis',
        query: literatureQuery,
        results: literatureResults,
        timestamp: new Date()
      });
      workflow.cumulativeContext.literature = literatureResults;

      // Step 3: Clinical Trials Agent
      console.log('Step 3: Clinical Trials Agent');
      const trialsQuery = this.buildTrialsQuery(initialQuery, workflow.cumulativeContext);
      const trialsResults = await this.runClinicalTrialsAnalysis(trialsQuery, workflow.cumulativeContext);
      workflow.agents.push({
        agent: 'clinical_trials',
        query: trialsQuery,
        results: trialsResults,
        timestamp: new Date()
      });
      workflow.cumulativeContext.clinical = trialsResults;

      // Step 4: Competitive Intelligence Agent
      console.log('Step 4: Competitive Intelligence Agent');
      const competitiveQuery = this.buildCompetitiveQuery(initialQuery, workflow.cumulativeContext);
      const competitiveResults = await this.runCompetitiveAnalysis(competitiveQuery, workflow.cumulativeContext);
      workflow.agents.push({
        agent: 'competitive_intelligence',
        query: competitiveQuery,
        results: competitiveResults,
        timestamp: new Date()
      });
      workflow.cumulativeContext.competitive = competitiveResults;

      // Step 5: Regulatory Analysis Agent  
      console.log('Step 5: Regulatory Analysis Agent');
      const regulatoryQuery = this.buildRegulatoryQuery(initialQuery, workflow.cumulativeContext);
      const regulatoryResults = await this.runRegulatoryAnalysis(regulatoryQuery, workflow.cumulativeContext);
      workflow.agents.push({
        agent: 'regulatory_analysis',
        query: regulatoryQuery,
        results: regulatoryResults,
        timestamp: new Date()
      });
      workflow.cumulativeContext.regulatory = regulatoryResults;

      // Step 6: Medical Writing Agent - Final Synthesis
      console.log('Step 6: Medical Writing Agent - Final Synthesis');
      const synthesisQuery = this.buildSynthesisQuery(initialQuery, workflow.cumulativeContext, reportType);
      const finalReport = await this.runMedicalWritingSynthesis(synthesisQuery, workflow.cumulativeContext);
      workflow.agents.push({
        agent: 'medical_writing',
        query: synthesisQuery,
        results: finalReport,
        timestamp: new Date()
      });
      workflow.finalReport = finalReport;
      workflow.status = 'completed';

      // Store completed workflow
      this.workflowState.set(sessionId, workflow);
      
      console.log(`Workflow completed for session: ${sessionId}`);
      
      return {
        success: true,
        sessionId,
        workflow,
        executionTime: new Date() - workflow.startTime,
        agentCount: workflow.agents.length,
        finalReport: finalReport.report || finalReport
      };

    } catch (error) {
      console.error(`Workflow error for session ${sessionId}:`, error);
      workflow.error = error.message;
      workflow.status = 'failed';
      workflow.completedSteps = workflow.agents.length;
      this.workflowState.set(sessionId, workflow);
      
      return {
        success: false,
        error: error.message,
        partialWorkflow: workflow
      };
    }
  }

  async runVectorSearch(query, workflow) {
    return await this.agents.vectorSearch.search({
      query,
      sessionId: workflow.sessionId,
      context: 'initial_search'
    });
  }

  async runLiteratureAnalysis(query, context) {
    return await this.agents.literatureAnalysis.analyze(query, context);
  }

  async runClinicalTrialsAnalysis(query, context) {
    return await this.agents.clinicalTrials.search(query, context);
  }

  async runCompetitiveAnalysis(query, context) {
    return await this.agents.competitiveIntelligence.analyze(query, context);
  }

  async runRegulatoryAnalysis(query, context) {
    return await this.agents.regulatoryAnalysis.analyze(query, context);
  }

  async runMedicalWritingSynthesis(query, context) {
    return await this.agents.medicalWriting.synthesize(query, context);
  }

  buildLiteratureQuery(originalQuery, vectorContext) {
    const keyTerms = vectorContext.keyTerms || [];
    const knowledgeGaps = vectorContext.gaps || [];
    
    return {
      query: originalQuery,
      focusAreas: keyTerms.slice(0, 5),
      gapsToAddress: knowledgeGaps,
      searchStrategy: vectorContext.searchStrategy || 'comprehensive',
      evidenceLevel: 'all'
    };
  }

  buildTrialsQuery(originalQuery, context) {
    const conditions = context.literature?.conditions || [];
    const interventions = context.literature?.interventions || [];
    
    return {
      query: originalQuery,
      conditions: conditions.slice(0, 3),
      interventions: interventions.slice(0, 3),
      phases: ['Phase 2', 'Phase 3'],
      literatureInsights: context.literature?.keyFindings || null
    };
  }

  buildCompetitiveQuery(originalQuery, context) {
    const competitors = context.clinical?.sponsors || [];
    const therapeuticAreas = context.literature?.therapeuticAreas || [];
    
    return {
      query: originalQuery,
      competitors: competitors.slice(0, 10),
      therapeuticAreas: therapeuticAreas,
      clinicalInsights: context.clinical?.competitiveLandscape || null,
      literatureInsights: context.literature?.competitiveReferences || null
    };
  }

  buildRegulatoryQuery(originalQuery, context) {
    const pathways = context.clinical?.regulatoryPathways || [];
    const indications = context.competitive?.targetIndications || [];
    
    return {
      query: originalQuery,
      pathways: pathways,
      indications: indications,
      competitiveContext: context.competitive?.regulatoryStrategies || null,
      clinicalContext: context.clinical?.regulatoryConsiderations || null
    };
  }

  buildSynthesisQuery(originalQuery, context, reportType) {
    return {
      originalQuery,
      reportType,
      synthesis: {
        vectorInsights: context.vector?.summary || null,
        literatureFindings: context.literature?.summary || null,
        clinicalLandscape: context.clinical?.summary || null,
        competitiveIntelligence: context.competitive?.summary || null,
        regulatoryEnvironment: context.regulatory?.summary || null
      },
      outputFormat: 'structured_report'
    };
  }

  getWorkflowStatus(sessionId) {
    return this.workflowState.get(sessionId);
  }

  getAllWorkflows() {
    return Array.from(this.workflowState.entries()).map(([id, workflow]) => ({
      sessionId: id,
      status: workflow.status,
      startTime: workflow.startTime,
      completedSteps: workflow.agents.length,
      totalSteps: 6
    }));
  }
}

module.exports = { SequentialWorkflowOrchestrator };
