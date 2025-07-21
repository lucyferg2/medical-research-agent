# Medical Research Agent API

AI-powered medical research and analysis platform for pharmaceutical teams.

## Features

- **Literature Review**: Automated PubMed searches and analysis
- **Competitive Intelligence**: Market and pipeline analysis
- **Clinical Trials**: ClinicalTrials.gov integration
- **Vector Storage**: Pinecone integration for research history
- **API-First**: RESTful API for easy integration

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/medical-research-agent.git
cd medical-research-agent
```

### 2. Set Environment Variables
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Deploy to Render
1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Set environment variables in Render dashboard
4. Deploy

### 4. Test API
```bash
curl https://your-service.onrender.com/health
```

## API Endpoints

### Literature Review
```bash
POST /research/literature
{
    "query": "CAR-T cell therapy multiple myeloma",
    "therapy_area": "oncology",
    "days_back": 90
}
```

### Competitive Analysis
```bash
POST /research/competitive
{
    "competitor_query": "pembrolizumab biosimilar",
    "therapy_area": "oncology"
}
```

### General Research
```bash
POST /research/general
{
    "query": "Latest developments in gene therapy",
    "research_type": "literature_review",
    "therapy_area": "general"
}
```

## Custom GPT Integration

Use this OpenAPI schema to create Custom GPT Actions:

```json
{
  "openapi": "3.0.1",
  "info": {
    "title": "Medical Research Agent",
    "version": "1.0.0"
  },
  "servers": [
    {"url": "https://your-service.onrender.com"}
  ],
  "paths": {
    "/research/literature": {
      "post": {
        "operationId": "conductLiteratureReview",
        "summary": "Conduct medical literature review",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "query": {"type": "string"},
                  "therapy_area": {"type": "string"},
                  "days_back": {"type": "integer"}
                },
                "required": ["query"]
              }
            }
          }
        }
      }
    }
  }
}
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| OPENAI_API_KEY | Yes | OpenAI API key |
| RESEARCH_EMAIL | Yes | Email for PubMed API |
| PINECONE_API_KEY | No | Pinecone API key (optional) |
| PINECONE_ENVIRONMENT | No | Pinecone environment |
| PINECONE_INDEX_NAME | No | Pinecone index name |

## Development

### Local Development
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Testing
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test literature review
curl -X POST http://localhost:8000/research/literature \
  -H "Content-Type: application/json" \
  -d '{"query": "CAR-T therapy", "therapy_area": "oncology"}'
```

## Architecture

```
medical-research-agent/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── agents/              # Research agents
│   │   ├── medical_agents.py
│   │   └── research_tools.py
│   ├── models/
│   │   └── schemas.py       # Pydantic models
│   └── utils/
│       └── vector_store.py  # Pinecone integration
├── requirements.txt
├── Dockerfile
└── render.yaml
```

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions, please open a GitHub issue.


