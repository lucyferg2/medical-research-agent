# Medical Research Agent

A sophisticated multi-agent system for pharmaceutical research, competitive intelligence, and regulatory analysis using OpenAI's Assistants API.

## Features

- **Sequential Multi-Agent Workflow**: Six specialized agents work in sequence to build comprehensive pharmaceutical intelligence
- **Vector Search Integration**: Semantic search capabilities with Pinecone vector database
- **Literature Analysis**: PubMed integration with GRADE evidence assessment
- **Clinical Trials Intelligence**: ClinicalTrials.gov integration with competitive landscape analysis
- **Regulatory Analysis**: FDA/EMA guidance analysis and pathway recommendations
- **Competitive Intelligence**: Market analysis and competitor tracking
- **Medical Writing**: Automated report generation and synthesis

## Architecture

### Agent Workflow
1. **Vector Search Agent** - Establishes baseline knowledge and identifies gaps
2. **Literature Analysis Agent** - Analyzes recent publications using vector insights
3. **Clinical Trials Agent** - Maps trial landscape using literature findings
4. **Competitive Intelligence Agent** - Analyzes market using clinical/literature data
5. **Regulatory Analysis Agent** - Assesses pathways using all gathered intelligence
6. **Medical Writing Agent** - Synthesizes everything into comprehensive report

## Quick Start

### Prerequisites
- Node.js 18+
- OpenAI API key
- Pinecone API key (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medical-research-agent.git
cd medical-research-agent
```

2. Install dependencies:
```bash
npm install
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Start the development server:
```bash
npm run dev
```

### Production Deployment (Render)

1. Connect your GitHub repository to Render
2. Set environment variables in Render dashboard
3. Deploy using the included `render.yaml` configuration

## API Endpoints

### Sequential Workflow
```
POST /api/sequential-workflow
{
  "query": "CAR-T therapy in multiple myeloma",
  "reportType": "comprehensive"
}
```

### Individual Agents
- `POST /api/agents/vector-search`
- `POST /api/agents/literature-analysis`
- `POST /api/agents/clinical-trials`
- `POST /api/agents/competitive-intel`
- `POST /api/agents/regulatory-analysis`
- `POST /api/agents/medical-writing`

### Workflow Management
- `GET /api/workflow/status/:sessionId`
- `GET /api/workflow/list`
- `POST /api/workflow/cancel/:sessionId`

### Health Checks
- `GET /api/health`
- `GET /api/health/status`

## Custom GPT Integration

### System Prompt
The system includes a structured prompt for OpenAI's Custom GPT that orchestrates the sequential workflow automatically.

### Actions Configuration
Configure the following actions in your Custom GPT:
- Vector Search: `https://medical-research-agent.onrender.com/api/agents/vector-search`
- Literature Analysis: `https://medical-research-agent.onrender.com/api/agents/literature-analysis`
- Clinical Trials: `https://medical-research-agent.onrender.com/api/agents/clinical-trials`
- Competitive Intelligence: `https://medical-research-agent.onrender.com/api/agents/competitive-intel`
- Regulatory Analysis: `https://medical-research-agent.onrender.com/api/agents/regulatory-analysis`
- Medical Writing: `https://medical-research-agent.onrender.com/api/agents/medical-writing`

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for agents | Yes |
| `PINECONE_API_KEY` | Pinecone API key for vector search | Optional |
| `PINECONE_INDEX_NAME` | Pinecone index name | Optional |
| `NODE_ENV` | Environment (development/production) | No |
| `PORT` | Server port | No |

## Development

### Project Structure
```
src/
├── agents/           # Individual agent implementations
├── orchestrator/     # Sequential workflow orchestrator
├── routes/          # Express.js routes
├── utils/           # Helper functions and utilities
└── server.js        # Main application entry point
```

### Adding New Agents
1. Create agent class extending `BaseAgent`
2. Implement required methods: `getTools()`, custom analysis methods
3. Add route in `agentRoutes.js`
4. Update orchestrator to include new agent

### Testing
```bash
npm test
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue in the GitHub repository or contact the development team.
