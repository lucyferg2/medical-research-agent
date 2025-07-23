class ContextBuilder {
  static buildProgressiveContext(workflowState) {
    const context = {
      keyTerms: new Set(),
      competitors: new Set(),
      therapeuticAreas: new Set(),
      regulatoryPathways: new Set(),
      clinicalEndpoints: new Set(),
      publications: new Set()
    };

    // Extract and accumulate context from each completed agent
    workflowState.agents.forEach(agent => {
      switch(agent.agent) {
        case 'vector_search':
          agent.results.keyTerms?.forEach(term => context.keyTerms.add(term));
          break;
        case 'literature_analysis':
          agent.results.therapeuticAreas?.forEach(area => context.therapeuticAreas.add(area));
          agent.results.competitors?.forEach(comp => context.competitors.add(comp));
          agent.results.publications?.forEach(pub => context.publications.add(pub));
          break;
        case 'clinical_trials':
          agent.results.sponsors?.forEach(sponsor => context.competitors.add(sponsor));
          agent.results.endpoints?.forEach(endpoint => context.clinicalEndpoints.add(endpoint));
          break;
        case 'competitive_intelligence':
          agent.results.regulatoryStrategies?.forEach(strategy => 
            context.regulatoryPathways.add(strategy));
          break;
      }
    });

    return {
      keyTerms: Array.from(context.keyTerms),
      competitors: Array.from(context.competitors),
      therapeuticAreas: Array.from(context.therapeuticAreas),
      regulatoryPathways: Array.from(context.regulatoryPathways),
      clinicalEndpoints: Array.from(context.clinicalEndpoints),
      publications: Array.from(context.publications)
    };
  }

  static extractInsights(agentResults) {
    const insights = {
      keyInsights: [],
      actionableItems: [],
      riskFactors: [],
      opportunities: []
    };

    agentResults.forEach(agent => {
      if (agent.results.keyFindings) {
        insights.keyInsights.push(...agent.results.keyFindings);
      }
      if (agent.results.recommendations) {
        insights.actionableItems.push(...agent.results.recommendations);
      }
      if (agent.results.risks) {
        insights.riskFactors.push(...agent.results.risks);
      }
      if (agent.results.opportunities) {
        insights.opportunities.push(...agent.results.opportunities);
      }
    });

    return insights;
  }
}

module.exports = { ContextBuilder };
