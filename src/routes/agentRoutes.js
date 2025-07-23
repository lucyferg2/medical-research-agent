const express = require('express');
const router = express.Router();
const { VectorSearchAgent } = require('../agents/VectorSearchAgent');
const { LiteratureAnalysisAgent } = require('../agents/LiteratureAnalysisAgent');
const { ClinicalTrialsAgent } = require('../agents/ClinicalTrialsAgent');
const { CompetitiveIntelligenceAgent } = require('../agents/CompetitiveIntelligenceAgent');
const { RegulatoryAnalysisAgent } = require('../agents/RegulatoryAnalysisAgent');
const { MedicalWritingAgent } = require('../agents/MedicalWritingAgent');

// Initialize agents
const agents = {
  vectorSearch: new VectorSearchAgent(),
  literatureAnalysis: new LiteratureAnalysisAgent(),
  clinicalTrials: new ClinicalTrialsAgent(),
  competitiveIntelligence: new CompetitiveIntelligenceAgent(),
  regulatoryAnalysis: new RegulatoryAnalysisAgent(),
  medicalWriting: new MedicalWritingAgent()
};

// Vector Search endpoint
router.post('/vector-search', async (req, res) => {
  try {
    const { query, namespace, top_k = 10, include_metadata = true } = req.body;
    
    const result = await agents.vectorSearch.search({
      query,
      namespace,
      top_k,
      include_metadata
    });
    
    res.json({ success: true, data: result });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Literature Analysis endpoint
router.post('/literature-analysis', async (req, res) => {
  try {
    const { query, focus_areas, evidence_level, prior_context } = req.body;
    
    const result = await agents.literatureAnalysis.analyze({
      query,
      focusAreas: focus_areas,
      evidenceLevel: evidence_level
    }, { prior: prior_context });
    
    res.json({ success: true, data: result });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Clinical Trials endpoint
router.post('/clinical-trials', async (req, res) => {
  try {
    const { 
      condition, 
      intervention, 
      phase, 
      literature_context, 
      vector_context 
    } = req.body;
    
    const result = await agents.clinicalTrials.search({
      query: `${condition} ${intervention}`,
      conditions: [condition],
      interventions: [intervention],
      phases: phase ? [phase] : undefined
    }, {
      literature: literature_context,
      vector: vector_context
    });
    
    res.json({ success: true, data: result });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Competitive Intelligence endpoint
router.post('/competitive-intel', async (req, res) => {
  try {
    const { 
      market_area, 
      competitors, 
      clinical_context, 
      literature_context 
    } = req.body;
    
    const result = await agents.competitiveIntelligence.analyze({
      query: market_area,
      competitors: competitors || [],
      therapeuticAreas: [market_area]
    }, {
      clinical: clinical_context,
      literature: literature_context
    });
    
    res.json({ success: true, data: result });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Regulatory Analysis endpoint
router.post('/regulatory-analysis', async (req, res) => {
  try {
    const { 
      therapeutic_area, 
      regulatory_region = 'FDA',
      competitive_context,
      clinical_context
    } = req.body;
    
    const result = await agents.regulatoryAnalysis.analyze({
      query: therapeutic_area,
      pathways: [],
      indications: [therapeutic_area]
    }, {
      competitive: competitive_context,
      clinical: clinical_context
    });
    
    res.json({ success: true, data: result });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Medical Writing endpoint
router.post('/medical-writing', async (req, res) => {
  try {
    const { 
      report_type = 'comprehensive',
      vector_findings,
      literature_findings,
      clinical_findings,
      competitive_findings,
      regulatory_findings
    } = req.body;
    
    const result = await agents.medicalWriting.synthesize({
      reportType: report_type,
      synthesis: {
        vectorInsights: vector_findings,
        literatureFindings: literature_findings,
        clinicalLandscape: clinical_findings,
        competitiveIntelligence: competitive_findings,
        regulatoryEnvironment: regulatory_findings
      }
    }, {
      vector: vector_findings,
      literature: literature_findings,
      clinical: clinical_findings,
      competitive: competitive_findings,
      regulatory: regulatory_findings
    });
    
    res.json({ success: true, data: result });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

module.exports = router;
