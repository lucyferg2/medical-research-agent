# Medical Research Agent — System Overview

> A multi-agent pharmaceutical intelligence platform that orchestrates six specialized AI agents to gather, analyze, and synthesize data from scientific databases, clinical registries, and market sources into a unified research report.

---

## How It Works

A user submits a natural-language query (e.g., *"CAR-T cell therapy for diffuse large B-cell lymphoma"*). The system executes a **sequential six-agent pipeline** where each agent queries a different real-world data source, analyzes the results with GPT-4.1-mini, and passes its structured findings forward to the next agent. The final agent synthesizes everything into a single comprehensive report with inline citations.

```
                         User Query
                             |
                             v
               +----------------------------+
               |  1. Vector Search Agent    |  Pinecone (semantic document search)
               +----------------------------+
                             |  vector_findings
                             v
               +----------------------------+
               |  2. Literature Agent       |  PubMed API (peer-reviewed articles)
               +----------------------------+
                             |  literature_findings
                             v
               +----------------------------+
               |  3. Clinical Trials Agent  |  ClinicalTrials.gov API
               +----------------------------+
                             |  clinical_findings
                             v
               +----------------------------+
               |  4. Competitive Intel Agent|  Google Custom Search + web scraping
               +----------------------------+
                             |  competitive_findings
                             v
               +----------------------------+
               |  5. Regulatory Agent       |  Google Custom Search (.gov priority)
               +----------------------------+
                             |  regulatory_findings
                             v
               +----------------------------+
               |  6. Medical Writing Agent  |  Synthesizes ALL prior findings
               +----------------------------+
                             |
                             v
                       Final Report
          (executive summary, analysis, recommendations,
                    full reference list)
```

Each agent receives the **full structured output** of all preceding agents, enabling cumulative context and cross-referencing.

---

## Agent Breakdown

| # | Agent | Data Source | What It Does | Key Output Fields |
|---|-------|-----------|--------------|-------------------|
| 1 | **Vector Search** | Pinecone | Queries a pre-indexed vector database for semantically similar documents to establish baseline knowledge | `top_hits`, document count |
| 2 | **Literature Analysis** | PubMed (NCBI E-utilities) | Searches and retrieves article abstracts, applies GRADE evidence assessment | `keyFindings`, `evidenceGrade`, `emergingTrends`, `references` (PMIDs) |
| 3 | **Clinical Trials** | ClinicalTrials.gov v2 API | Maps the trial landscape — sponsors, phases, competitive positioning | `keySponsors`, `trialPhaseBreakdown`, `competitiveLandscape`, `references` (NCT IDs) |
| 4 | **Competitive Intelligence** | Google Custom Search + web scraping | Analyzes market dynamics, key players, opportunities, and threats | `marketOpportunities`, `competitiveThreats`, `keyPlayers`, `references` (URLs) |
| 5 | **Regulatory Analysis** | Google Custom Search (`.gov` priority) + web scraping | Identifies FDA/EMA guidance documents, submission pathways, and hurdles | `applicableGuidances`, `potentialPathways`, `keyConsiderations`, `references` (URLs) |
| 6 | **Medical Writing** | All 5 agents' outputs | Synthesizes everything into a cohesive narrative report with inline citations | `executiveSummary`, `literatureLandscape`, `clinicalPipeline`, `marketIntelligence`, `regulatoryEnvironment`, `strategicRecommendations`, `references` |

---

## AI and External Capabilities

| Capability | Technology | Purpose |
|-----------|-----------|---------|
| Text analysis & synthesis | **OpenAI GPT-4.1-mini** | Summarization, structured analysis, report generation (JSON mode, temp 0.2-0.3) |
| Embedding generation | **OpenAI text-embedding-3-small** | Converts queries to vectors for Pinecone semantic search |
| Semantic document search | **Pinecone** vector database | Retrieves similar documents from a pre-indexed knowledge base |
| Literature search | **PubMed** (NCBI E-utilities) | Searches and fetches peer-reviewed biomedical article abstracts |
| Clinical trial data | **ClinicalTrials.gov** v2 API | Retrieves trial metadata: sponsors, phases, status, NCT IDs |
| Web search | **Google Custom Search API** | Finds market research and regulatory guidance documents |
| Web scraping | **BeautifulSoup + requests** | Extracts clean text content from web pages for analysis |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3 |
| Web framework | FastAPI |
| Server | Uvicorn (ASGI) |
| Data validation | Pydantic (with field validators) |
| AI/LLM | OpenAI API (`gpt-4.1-mini`, `text-embedding-3-small`) |
| Vector DB | Pinecone |
| HTML/XML parsing | BeautifulSoup4 + lxml |
| HTTP client | requests (persistent session with browser headers) |
| Search API | google-api-python-client |
| Config | python-dotenv |

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Health check — confirms the API is running |
| `POST` | `/api/agents/vector-search` | Run vector search agent independently |
| `POST` | `/api/agents/literature-analysis` | Run literature analysis agent independently |
| `POST` | `/api/agents/clinical-trials` | Run clinical trials agent independently |
| `POST` | `/api/agents/competitive-intel` | Run competitive intelligence agent independently |
| `POST` | `/api/agents/regulatory-analysis` | Run regulatory analysis agent independently |
| `POST` | `/api/agents/medical-writing` | Run medical writing agent independently |
| `POST` | `/api/sequential-workflow` | **Run the full 6-agent pipeline end-to-end** |

The `/api/sequential-workflow` endpoint accepts a `query` string and an optional `reportType` (`comprehensive`, `executive`, `technical`, or `strategic`) and returns a fully synthesized report.

---

## Getting Started

```bash
# 1. Clone the repository
git clone <repo-url> && cd medical-research-agent

# 2. Create a virtual environment
python -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your API keys (see table below)

# 5. Run the server
python main.py
# Server starts on http://0.0.0.0:8000 (or PORT from .env)
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4.1-mini and embeddings |
| `PINECONE_API_KEY` | Yes* | Pinecone vector database key (*app degrades gracefully without it*) |
| `PINECONE_INDEX_NAME` | No | Vector index name (default: `medical-research-index`) |
| `PINECONE_ENVIRONMENT` | No | Pinecone region (default: `us-west1-gcp`) |
| `GOOGLE_API_KEY` | Yes | Google Custom Search API key (for competitive + regulatory agents) |
| `GOOGLE_CSE_ID` | Yes | Google Custom Search Engine ID |
| `PORT` | No | Server port (default: `8000`) |
| `NODE_ENV` | No | Environment mode (`production` / `development`) |
| `LOG_LEVEL` | No | Logging verbosity (default: `info`) |

---

## Deployment

Configured for **Render.com** via `render.yaml`:
- **Runtime:** Python, Starter plan
- **Build:** `pip install -r requirements.txt`
- **Start:** `python main.py`
- **Secrets:** `OPENAI_API_KEY` and `PINECONE_API_KEY` managed as non-synced env vars

---

## Repository Structure

```
medical-research-agent/
  main.py              # Entire application — models, utilities, agents, endpoints (529 lines)
  requirements.txt     # Python dependencies (10 packages)
  .env.example         # Environment variable template
  render.yaml          # Render.com deployment config
  .gitignore           # Standard Python + .env exclusions
  OVERVIEW.md          # This document
```

The application is a single-file architecture (`main.py`) organized into four sections: initialization/config, Pydantic data models, utility functions, and API endpoints.
