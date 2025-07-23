const { BaseAgent } = require('./BaseAgent');

class ClinicalTrialsAgent extends BaseAgent {
  constructor() {
    super(
      'Clinical Trials Specialist',
      `You are a clinical trials expert with ClinicalTrials.gov integration. Your role is to:
      1. Search and analyze clinical trials data
      2. Identify trial sponsors and competitive landscape
      3. Extract endpoints and study designs
      4. Assess recruitment feasibility
      5. Map regulatory pathways and considerations
      
      Always provide structured responses with:
      - activeTrials: Currently recruiting or active trials
      - sponsors: Companies/organizations running trials
      - phases: Distribution of trial phases
      - endpoints: Primary and secondary endpoints
      - competitiveLandscape: Analysis of competitive trials
      - regulatoryPathways: Identified regulatory strategies
      - regulatoryConsiderations: Key regulatory factors
      - summary: Overview of clinical trial landscape`
    );
  }

  async search(query, context) {
    // Simulate ClinicalTrials.gov search (replace with actual API)
    const mockTrials = await this.simulateTrialsSearch(query, context);
    
    return await this.runAgent(query.query || query, {
      trialsResults: mockTrials,
      literatureContext: context.literature,
      conditions: query.conditions || [],
      interventions: query.interventions || []
    });
  }

  async simulateTrialsSearch(query, context) {
    const searchTerm = query.query || query;
    return {
      totalTrials: 78,
      activeTrials: 34,
      recruitingTrials: 23,
      trials: [
        {
          nctId: 'NCT05234567',
          title: `Phase 3 study of ${searchTerm}`,
          sponsor: 'Major Pharma Corp',
          phase: 'Phase 3',
          status: 'Recruiting',
          primaryEndpoint: 'Overall survival'
        },
        {
          nctId: 'NCT05234568',
          title: `Phase 2 combination study with ${searchTerm}`,
          sponsor: 'Biotech Company',
          phase: 'Phase 2',
          status: 'Active',
          primaryEndpoint: 'Progression-free survival'
        }
      ]
    };
  }

  getTools() {
    return [
      {
        type: 'function',
        function: {
          name: 'search_clinical_trials',
          description: 'Search ClinicalTrials.gov database',
          parameters: {
            type: 'object',
            properties: {
              condition: { type: 'string' },
              intervention: { type: 'string' },
              phase: { type: 'string' },
              sponsor: { type: 'string' }
            }
          }
        }
      }
    ];
  }
}

module.exports = { ClinicalTrialsAgent };
