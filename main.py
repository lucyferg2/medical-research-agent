import os
import logging
import traceback
import json
import requests
from datetime import datetime
from typing import List, Optional

from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from googleapiclient.discovery import build
from pinecone import Pinecone
from pydantic import BaseModel
import openai

# --- INITIALIZATION AND CONFIGURATION ---

# Initialize FastAPI app
app = FastAPI(
    title="Medical Research Agent API",
    description="A multi-agent system for pharmaceutical intelligence, powered by real-time data APIs.",
    version="2.0.0"
)

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API Keys from Environment Variables
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")

# Check for essential API keys
if not openai.api_key or not pinecone_api_key or not google_api_key or not google_cse_id:
    logger.warning("One or more API keys are missing. Some functionality will be disabled.")

# Connect to Pinecone
try:
    pc = Pinecone(api_key=pinecone_api_key)
    INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-research-index")
    if INDEX_NAME not in pc.list_indexes().names():
        raise HTTPException(status_code=500, detail=f"Index '{INDEX_NAME}' not found in Pinecone.")
    index = pc.Index(INDEX_NAME)
    logger.info(f"Successfully connected to Pinecone index: '{INDEX_NAME}'")
except Exception as e:
    logger.error(f"Failed to connect to Pinecone: {e}")
    index = None # Allow app to run without Pinecone for other agents

# --- API AND DATA MODELS ---

# Agent Request Models
class VectorSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    namespace: str = "Test Deck"

class LiteratureAnalysisRequest(BaseModel):
    query: str
    focus_areas: Optional[List[str]] = []
    prior_context: Optional[str] = None

class ClinicalTrialsRequest(BaseModel):
    condition: str
    intervention: str
    phase: Optional[str] = "All"
    literature_context: Optional[str] = None

class CompetitiveIntelRequest(BaseModel):
    market_area: str
    competitors: Optional[List[str]] = []
    clinical_context: Optional[str] = None

class RegulatoryAnalysisRequest(BaseModel):
    therapeutic_area: str
    regulatory_region: str = "FDA"
    competitive_context: Optional[str] = None

class MedicalWritingRequest(BaseModel):
    report_type: str = "comprehensive"
    vector_findings: Optional[dict] = None
    literature_findings: Optional[dict] = None
    clinical_findings: Optional[dict] = None
    competitive_findings: Optional[dict] = None
    regulatory_findings: Optional[dict] = None

class SequentialWorkflowRequest(BaseModel):
    query: str
    reportType: str = "comprehensive"

# --- UTILITY FUNCTIONS ---

async def generate_embedding(text: str) -> List[float]:
    """Generates embeddings using OpenAI."""
    try:
        response = openai.embeddings.create(input=[text], model="text-embedding-3-small")
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding.")

def get_web_content(url: str) -> str:
    """Fetches and parses text content from a URL."""
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'MedicalResearchAgent/1.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav']):
            script_or_style.decompose()
        text = soup.get_text(separator='\n', strip=True)
        return ' '.join(text.split())[:4000] # Limit content length
    except Exception as e:
        logger.warning(f"Failed to fetch content from {url}: {e}")
        return ""

async def summarize_with_openai(prompt: str, context_data: str) -> dict:
    """Uses OpenAI to summarize context data based on a prompt."""
    try:
        full_prompt = f"{prompt}\n\nAnalyze the following data and provide a structured JSON response:\n\n{context_data}"
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"OpenAI summarization failed: {e}")
        return {"error": "Failed to generate summary.", "details": str(e)}

# --- API ENDPOINTS ---

@app.get("/")
async def root():
    return {"message": "Medical Research Agent API v2 is running!"}

# --- AGENT ENDPOINTS ---

@app.post("/api/agents/vector-search")
async def vector_search_agent(request: VectorSearchRequest):
    """Vector search agent using Pinecone."""
    if not index:
        raise HTTPException(status_code=503, detail="Pinecone connection not available.")
    try:
        embedding = await generate_embedding(request.query)
        results = index.query(
            namespace=request.namespace,
            vector=embedding,
            top_k=request.top_k,
            include_metadata=True
        ).to_dict()
        return {"success": True, "data": results.get('matches', [])}
    except Exception as e:
        logger.error(f"Vector search failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Vector search failed: {e}")

@app.post("/api/agents/literature-analysis")
async def literature_analysis_agent(request: LiteratureAnalysisRequest):
    """Literature analysis agent using the PubMed API."""
    try:
        # 1. Search PubMed for article IDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {"db": "pubmed", "term": request.query, "retmode": "json", "retmax": 15}
        search_res = requests.get(search_url, params=search_params).json()
        id_list = search_res.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return {"success": True, "data": {"summary": "No relevant articles found on PubMed."}}

        # 2. Fetch article abstracts
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {"db": "pubmed", "id": ",".join(id_list), "retmode": "xml"}
        fetch_res = requests.get(fetch_url, params=fetch_params).text
        
        # 3. Summarize with OpenAI
        prompt = f"""
        You are a biomedical literature expert. Analyze the provided PubMed article data (in XML format) for the query: "{request.query}".
        Focus on: {request.focus_areas or 'general analysis'}.
        Prior context: {request.prior_context or 'None'}.
        Provide a structured analysis including:
        - keyFindings: List of the most important research insights.
        - evidenceGrade: A quality assessment of the evidence (e.g., High, Medium, Low).
        - emergingTrends: Any new or emerging trends identified.
        - summary: A concise overview of the literature landscape.
        """
        summary = await summarize_with_openai(prompt, fetch_res[:25000]) # Truncate for API limits
        return {"success": True, "data": summary}
    except Exception as e:
        logger.error(f"Literature analysis failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Literature analysis failed: {e}")

@app.post("/api/agents/clinical-trials")
async def clinical_trials_agent(request: ClinicalTrialsRequest):
    """Clinical trials analysis using the ClinicalTrials.gov API."""
    try:
        # 1. Search ClinicalTrials.gov
        api_url = "https://clinicaltrials.gov/api/v2/studies"
        query = f"{request.condition} AND {request.intervention}"
        if request.phase != "All":
            query += f" AND phase:{request.phase}"
        params = {"query.term": query, "pageSize": 20, "format": "json"}
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        trials_data = response.json()

        if not trials_data.get("studies"):
            return {"success": True, "data": {"summary": "No relevant clinical trials found."}}
        
        # 2. Summarize with OpenAI
        prompt = f"""
        You are a clinical trials intelligence analyst. Based on the provided data from ClinicalTrials.gov, analyze the trial landscape for:
        - Condition: {request.condition}
        - Intervention: {request.intervention}
        Provide an analysis including:
        - keySponsors: The main companies or organizations.
        - trialPhaseBreakdown: A summary of trial phases (e.g., Phase 1, Phase 2).
        - competitiveLandscape: An analysis of the competitive environment based on the trials.
        - summary: A strategic overview of the clinical trial landscape.
        """
        summary = await summarize_with_openai(prompt, json.dumps(trials_data)[:25000])
        return {"success": True, "data": summary}
    except Exception as e:
        logger.error(f"Clinical trials analysis failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Clinical trials analysis failed: {e}")

@app.post("/api/agents/competitive-intel")
async def competitive_intelligence_agent(request: CompetitiveIntelRequest):
    """Competitive intelligence using Google Search API."""
    try:
        service = build("customsearch", "v1", developerKey=google_api_key)
        search_query = f"market analysis and competitors for {request.market_area}"
        res = service.cse().list(q=search_query, cx=google_cse_id, num=5).execute()
        
        items = res.get('items', [])
        if not items:
            return {"success": True, "data": {"summary": "No competitive intelligence found."}}

        # Scrape and compile content
        content = "\n\n".join([f"Source: {item['link']}\nContent: {get_web_content(item['link'])}" for item in items])
        
        # Summarize with OpenAI
        prompt = f"""
        You are a pharmaceutical market intelligence expert. Analyze the provided web search results about '{request.market_area}'.
        Known competitors to consider: {request.competitors or 'None'}.
        Clinical context: {request.clinical_context or 'None'}.
        Provide a structured analysis with:
        - marketOpportunities: Identified market gaps or opportunities.
        - competitiveThreats: Key threats from competitors.
        - keyPlayers: Analysis of major companies in this space.
        - summary: A strategic competitive overview.
        """
        summary = await summarize_with_openai(prompt, content)
        return {"success": True, "data": summary}
    except Exception as e:
        logger.error(f"Competitive intelligence failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Competitive intelligence failed: {e}")


@app.post("/api/agents/regulatory-analysis")
async def regulatory_analysis_agent(request: RegulatoryAnalysisRequest):
    """Regulatory analysis using Google Search API."""
    try:
        service = build("customsearch", "v1", developerKey=google_api_key)
        search_query = f"{request.regulatory_region} regulatory guidance for {request.therapeutic_area}"
        # Prioritize government and agency sites in search
        res = service.cse().list(q=search_query, cx=google_cse_id, num=5, siteSearch="*.gov").execute()

        items = res.get('items', [])
        if not items:
            return {"success": True, "data": {"summary": "No regulatory guidance documents found."}}
            
        content = "\n\n".join([f"Source: {item['link']}\nContent: {get_web_content(item['link'])}" for item in items])

        prompt = f"""
        You are a regulatory affairs expert. Analyze the provided web search results regarding '{request.therapeutic_area}' for the {request.regulatory_region} region.
        Competitive context: {request.competitive_context or 'None'}.
        Provide analysis with:
        - applicableGuidances: Summary of relevant guidance documents.
        - potentialPathways: Recommended regulatory submission strategies (e.g., accelerated approval).
        - keyConsiderations: Important regulatory hurdles or requirements.
        - summary: A comprehensive regulatory environment analysis.
        """
        summary = await summarize_with_openai(prompt, content)
        return {"success": True, "data": summary}
    except Exception as e:
        logger.error(f"Regulatory analysis failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Regulatory analysis failed: {e}")


@app.post("/api/agents/medical-writing")
async def medical_writing_agent(request: MedicalWritingRequest):
    """Medical writing agent to synthesize all findings."""
    try:
        prompt = f"""
        You are an expert medical writer. Synthesize all provided findings into a single, comprehensive '{request.report_type}' report.
        Integrate the following structured data into a cohesive narrative. Do not simply list the findings; create a well-structured report.
        
        1. Vector Search Findings: {json.dumps(request.vector_findings, indent=2)}
        2. Literature Landscape: {json.dumps(request.literature_findings, indent=2)}
        3. Clinical Pipeline: {json.dumps(request.clinical_findings, indent=2)}
        4. Market Intelligence: {json.dumps(request.competitive_findings, indent=2)}
        5. Regulatory Environment: {json.dumps(request.regulatory_findings, indent=2)}
        
        Generate a final report with these sections:
        - executiveSummary: A high-level overview with key insights and strategic recommendations.
        - literatureLandscape: Detailed analysis of research trends and evidence.
        - clinicalPipeline: In-depth look at trial data and competitive positioning.
        - marketIntelligence: Assessment of market dynamics, opportunities, and threats.
        - regulatoryEnvironment: Analysis of pathways and compliance considerations.
        - strategicRecommendations: A list of clear, actionable next steps.
        
        Format the entire output as a single JSON object containing these keys.
        """
        # Note: We are not using the `summarize_with_openai` utility here because the prompt is complex and specific.
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        report = json.loads(response.choices[0].message.content)
        return {"success": True, "data": report}
    except Exception as e:
        logger.error(f"Medical writing failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Medical writing failed: {e}")

# --- WORKFLOW ENDPOINT ---
@app.post("/api/sequential-workflow")
async def sequential_workflow(request: SequentialWorkflowRequest):
    """Execute the complete sequential multi-agent workflow."""
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"[{session_id}] Starting sequential workflow for query: '{request.query}'")
    
    # Simple extraction of condition/intervention from query for subsequent agents
    # A more advanced implementation would use an LLM call to parse this robustly
    query_parts = request.query.replace("in", ",").replace("for", ",").split(",")
    condition = query_parts[1].strip() if len(query_parts) > 1 else request.query
    intervention = query_parts[0].strip()

    try:
        # Step 1: Vector Search Agent
        vector_res = await vector_search_agent(VectorSearchRequest(query=request.query))
        vector_findings = {"summary": f"Found {len(vector_res['data'])} documents.", "top_hits": vector_res['data'][:3]}
        logger.info(f"[{session_id}] Vector Search complete.")

        # Step 2: Literature Analysis Agent
        lit_res = await literature_analysis_agent(LiteratureAnalysisRequest(query=request.query, prior_context=json.dumps(vector_findings)))
        literature_findings = lit_res['data']
        logger.info(f"[{session_id}] Literature Analysis complete.")

        # Step 3: Clinical Trials Agent
        clinical_res = await clinical_trials_agent(ClinicalTrialsRequest(condition=condition, intervention=intervention, literature_context=literature_findings.get('summary')))
        clinical_findings = clinical_res['data']
        logger.info(f"[{session_id}] Clinical Trials Analysis complete.")

        # Step 4: Competitive Intelligence Agent
        comp_res = await competitive_intelligence_agent(CompetitiveIntelRequest(market_area=request.query, clinical_context=clinical_findings.get('summary')))
        competitive_findings = comp_res['data']
        logger.info(f"[{session_id}] Competitive Intelligence complete.")
        
        # Step 5: Regulatory Analysis Agent
        reg_res = await regulatory_analysis_agent(RegulatoryAnalysisRequest(therapeutic_area=intervention, competitive_context=competitive_findings.get('summary')))
        regulatory_findings = reg_res['data']
        logger.info(f"[{session_id}] Regulatory Analysis complete.")

        # Step 6: Medical Writing Agent
        final_report_res = await medical_writing_agent(MedicalWritingRequest(
            report_type=request.reportType,
            vector_findings=vector_findings,
            literature_findings=literature_findings,
            clinical_findings=clinical_findings,
            competitive_findings=competitive_findings,
            regulatory_findings=regulatory_findings
        ))
        logger.info(f"[{session_id}] Medical Writing complete.")

        return {"success": True, "sessionId": session_id, "report": final_report_res['data']}

    except Exception as e:
        logger.error(f"Sequential workflow failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Sequential workflow failed: {e}")

if __name__ == "__main__":
    import uvicorn
    # Fallback to a default port if not set
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
