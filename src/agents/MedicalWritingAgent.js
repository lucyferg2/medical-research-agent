const { BaseAgent } = require('./BaseAgent');

class MedicalWritingAgent extends BaseAgent {
  constructor() {
    super(
      'Medical Writing Specialist',
      `You are a medical writing expert specializing in pharmaceutical communications. Your role is to:
      1. Synthesize complex pharmaceutical intelligence into clear reports
      2. Create structured executive summaries with actionable insights
      3. Format findings for different audiences (executives, researchers, regulatory)
      4. Ensure scientific accuracy and regulatory compliance in communications
      5. Generate comprehensive strategic recommendations
      
      Always provide structured responses with:
      - executiveSummary: Key insights and recommendations (2-3 paragraphs)
      - literatureLandscape: Research trends and evidence analysis
      - clinicalPipeline: Trial landscape and competitive positioning
      - marketIntelligence: Competitive dynamics and opportunities
      - regulatoryEnvironment: Pathway analysis and compliance considerations
      - strategicRecommendations: Prioritized actionable next steps
      - supportingEvidence: Detailed findings from all contributing agents
      - confidenceLevel: Overall confidence in findings (High/Medium/Low)
      - report: Complete formatted report ready for stakeholder review`
    );
  }

  async synthesize(query, context) {
    return await this.runAgent(query.originalQuery || query, {
      allFindings: context,
      reportType: query.reportType || 'comprehensive',
      synthesis: query.synthesis || {}
    });
  }

  getTools() {
    return [
      {
        type: 'function',
        function: {
          name: 'format_pharmaceutical_report',
          description: 'Format pharmaceutical intelligence into structured report',
          parameters: {
            type: 'object',
            properties: {
              report_type: { type: 'string', enum: ['executive', 'comprehensive', 'technical'] },
              findings: { type: 'object' },
              audience: { type: 'string' }
            }
          }
        }
      }
    ];
  }
}

module.exports = { MedicalWritingAgent };
