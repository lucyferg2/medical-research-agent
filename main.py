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
from pydantic import BaseModel, field_validator
import openai
from typing import List, Optional

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

    @field_validator('query', 'namespace')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('must not be empty')
        return v

    @field_validator('top_k')
    def top_k_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('must be a positive integer')
        return v

class LiteratureAnalysisRequest(BaseModel):
    query: str
    focus_areas: Optional[List[str]] = []
    prior_context: Optional[str] = None

    @field_validator('query')
    def query_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('query must not be empty')
        return v

    @field_validator('focus_areas')
    def focus_areas_not_empty(cls, v):
        if any(not area.strip() for area in v):
            raise ValueError('focus_areas must not contain empty strings')
        return v

class ClinicalTrialsRequest(BaseModel):
    condition: str
    intervention: str
    phase: Optional[str] = "All"
    literature_context: Optional[str] = None
    
    @field_validator('phase')
    def phase_must_be_valid(cls, v):
        allowed_phases = ["All", "Phase 1", "Phase 2", "Phase 3", "Phase 4"]
        if v not in allowed_phases:
            raise ValueError(f'phase must be one of {allowed_phases}')
        return v

class CompetitiveIntelRequest(BaseModel):
    market_area: str
    competitors: Optional[List[str]] = []
    clinical_context: Optional[str] = None

    @field_validator('market_area')
    def market_area_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('market_area must not be empty')
        return v

class RegulatoryAnalysisRequest(BaseModel):
    therapeutic_area: str
    regulatory_region: str = "FDA"
    competitive_context: Optional[str] = None

    @field_validator('therapeutic_area')
    def therapeutic_area_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('therapeutic_area must not be empty')
        return v

    @field_validator('regulatory_region')
    def region_must_be_valid(cls, v):
        allowed_regions = ["FDA", "EMA"]
        if v not in allowed_regions:
            raise ValueError(f'regulatory_region must be one of {allowed_regions}')
        return v

class MedicalWritingRequest(BaseModel):
    report_type: str = "comprehensive"
    vector_findings: Optional[dict] = None
    literature_findings: Optional[dict] = None
    clinical_findings: Optional[dict] = None
    competitive_findings: Optional[dict] = None
    regulatory_findings: Optional[dict] = None

    @field_validator('report_type')
    def report_type_must_be_valid(cls, v):
        allowed_types = ["comprehensive", "executive", "technical", "strategic"]
        if v not in allowed_types:
            raise ValueError(f'report_type must be one of {allowed_types}')
        return v

class SequentialWorkflowRequest(BaseModel):
    query: str
    reportType: str = "comprehensive"

    @field_validator('query')
    def query_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('query must not be empty')
        return v

    @field_validator('reportType')
    def report_type_must_be_valid(cls, v):
        allowed_types = ["comprehensive", "executive", "technical", "strategic"]
        if v not in allowed_types:
            raise ValueError(f'reportType must be one of {allowed_types}')
        return v

# --- UTILITY FUNCTIONS ---

async def generate_embedding(text: str) -> List[float]:
    """Generates embeddings using OpenAI."""
    try:
        response = openai.embeddings.create(input=[text], model="text-embedding-3-small")
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding.")

# Create a single, reusable session object to persist cookies and headers
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Cache-Control': 'max-age=0',
})

def get_web_content(url: str) -> str:
    """
    Fetches and parses content using a persistent session to better mimic a real browser.
    """
    try:
        # Use the session object for the request
        response = session.get(url, timeout=20, allow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()

        if 'pdf' in content_type:
            logger.warning(f"Skipping PDF content from {url}")
            return f"Content from {url} is a PDF and was not processed."

        if 'xml' in content_type:
            soup = BeautifulSoup(response.content, 'lxml-xml')
        else:
            soup = BeautifulSoup(response.content, 'lxml')
        
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'figure']):
            element.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        logger.info(f"Successfully scraped content from {url}")
        return ' '.join(text.split())[:8000]

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch content from {url} after enhancements. Reason: {e}")
        return f"Could not retrieve content. The site may be blocking automated access. Reason: {e}"
async def summarize_with_openai(prompt: str, context_data: str) -> dict:
    """Uses OpenAI to summarize context data based on a prompt."""
    try:
        full_prompt = f"{prompt}\n\nAnalyze the following data, always provide references to where the information came from, and provide a structured JSON response:\n\n{context_data}"
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
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
    logger.info(f"Starting literature analysis for query: {request.query}")
    logger.debug(f"Request details: {request.model_dump()}")
    try:
        # 1. Search PubMed for article IDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        logger.info(f"Search: {search_url}")
        search_params = {"db": "pubmed", "term": request.query, "retmode": "json", "retmax": 15}
        search_res = requests.get(search_url, params=search_params).json()
        id_list = search_res.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return {"success": True, "data": {"summary": "No relevant articles found on PubMed."}}

        # 2. Fetch article abstracts
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        logger.info(f"Fetch URL: {fetch_url}")
        fetch_params = {"db": "pubmed", "id": ",".join(id_list), "retmode": "xml"}
        fetch_res = requests.get(fetch_url, params=fetch_params).text
        
        # 3. Summarize with OpenAI
        prompt = f"""
        You are a biomedical literature expert. Analyze the provided PubMed article data (in XML format) for the query: "{request.query}".
        For each key finding, you MUST include the PubMed ID (PMID) of the source article in the format [PMID: XXXXXX].
        Focus on: {request.focus_areas or 'general analysis'}.
        Prior context: {request.prior_context or 'None'}.
        Provide a structured analysis including:
        - keyFindings: List of the most important research insights.
        - evidenceGrade: A quality assessment of the evidence (e.g., High, Medium, Low).
        - emergingTrends: Any new or emerging trends identified.
        - summary: A concise overview of the literature landscape.
        - references: A list of all PMIDs for the articles analyzed.
        """
        summary = await summarize_with_openai(prompt, fetch_res[:25000]) # Truncate for API limits
        logger.info(f"Successfully completed literature analysis for query: {request.query}")
        logger.debug(f"Response data: {summary}")
        return {"success": True, "data": summary}
    except Exception as e:
        logger.error(f"Literature analysis failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Literature analysis failed: {e}")

@app.post("/api/agents/clinical-trials")
async def clinical_trials_agent(request: ClinicalTrialsRequest):
    logger.info(f"Starting clinical trials analysis for condition: {request.condition}, intervention: {request.intervention}")
    logger.debug(f"Request details: {request.model_dump()}")
    """Clinical trials analysis using the ClinicalTrials.gov API."""
    try:
        # 1. Search ClinicalTrials.gov
        api_url = "https://clinicaltrials.gov/api/v2/studies"
        logger.info(f"Search: {api_url}")
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
        The ClinicalTrials.gov API returns a 'protocolSection' with an 'identificationModule' that contains the nctId. You should extract this ID for each trial.
        Provide an analysis including:
        - keySponsors: The main companies or organizations.
        - trialPhaseBreakdown: A summary of trial phases (e.g., Phase 1, Phase 2).
        - competitiveLandscape: An analysis of the competitive environment based on the trials.
        - summary: A strategic overview of the clinical trial landscape.
        - references: A list of all nctId for the trials analyzed.
        """
        summary = await summarize_with_openai(prompt, json.dumps(trials_data)[:25000])
        logger.debug(f"Response data: {summary}")
        return {"success": True, "data": summary}
    except Exception as e:
        logger.error(f"Clinical trials analysis failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Clinical trials analysis failed: {e}")

@app.post("/api/agents/competitive-intel")
async def competitive_intelligence_agent(request: CompetitiveIntelRequest):
    logger.info(f"Starting competitor intelligence for market area: {request.market_area}")
    logger.debug(f"Request details: {request.model_dump()}")
    """Competitive intelligence using Google Search API."""
    try:
        service = build("customsearch", "v1", developerKey=google_api_key)
        search_query = f"market analysis and competitors for {request.market_area}"
        logger.info(f"Search query: {search_query}")
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
        For each key finding, you MUST include the URL of the source article in the format [URL: XXXXXX]. ALways provide the full URL, never truncate or summarize 
        Provide a structured analysis with:
        - marketOpportunities: Identified market gaps or opportunities.
        - competitiveThreats: Key threats from competitors.
        - keyPlayers: Analysis of major companies in this space.
        - summary: A strategic competitive overview.
        - references:  A list of all URLs for the pages analyzed.
        """
        summary = await summarize_with_openai(prompt, content)
        logger.debug(f"Response data: {summary}")
        return {"success": True, "data": summary}
    except Exception as e:
        logger.error(f"Competitive intelligence failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Competitive intelligence failed: {e}")


@app.post("/api/agents/regulatory-analysis")
async def regulatory_analysis_agent(request: RegulatoryAnalysisRequest):
    logger.info(f"Starting regulatory analysis for therapeutic area: {request.therapeutic_area}")
    logger.debug(f"Request details: {request.model_dump()}")
    """Regulatory analysis using Google Search API."""
    try:
        service = build("customsearch", "v1", developerKey=google_api_key)
        search_query = f"{request.regulatory_region} regulatory guidance for {request.therapeutic_area}"
        logger.info(f"Search query: {search_query}")
        
        # Prioritize government and agency sites in search
        res = service.cse().list(q=search_query, cx=google_cse_id, num=5, siteSearch="*.gov").execute()

        items = res.get('items', [])
        if not items:
            return {"success": True, "data": {"summary": "No regulatory guidance documents found."}}
            
        content = "\n\n".join([f"Source: {item['link']}\nContent: {get_web_content(item['link'])}" for item in items])

        prompt = f"""
        You are a regulatory affairs expert. Analyze the provided web search results regarding '{request.therapeutic_area}' for the {request.regulatory_region} region.
        Competitive context: {request.competitive_context or 'None'}.
        For each key finding, you MUST include the URL of the source article in the format [URL: XXXXXX]. ALways provide the full URL, never truncate or summarize 
        Provide analysis with:
        - applicableGuidances: Summary of relevant guidance documents.
        - potentialPathways: Recommended regulatory submission strategies (e.g., accelerated approval).
        - keyConsiderations: Important regulatory hurdles or requirements.
        - summary: A comprehensive regulatory environment analysis.
        - references: A list of all URLs for the articles analyzed.
        """
        summary = await summarize_with_openai(prompt, content)
        logger.debug(f"Response data: {summary}")
        return {"success": True, "data": summary}
    except Exception as e:
        logger.error(f"Regulatory analysis failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Regulatory analysis failed: {e}")


@app.post("/api/agents/medical-writing")
async def medical_writing_agent(request: MedicalWritingRequest):
    logger.info(f"Starting medical writer for report type: {request.report_type}")
    logger.debug(f"Request details: {request.model_dump()}")
    """Medical writing agent to synthesize all findings."""
    try:
        prompt = f"""
        You are an expert medical writer. Synthesize all provided findings into a single, comprehensive '{request.report_type}' report.
        You MUST include inline citations for every piece of information, using the provided references. For example: [PMID: 123456] or [NCT: 12345678] or [Source: https://example.com].

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
        - references: A complete list of all sources cited in the report.
        
        Format the entire output as a single JSON object containing these keys.
        """
        # Note: We are not using the `summarize_with_openai` utility here because the prompt is complex and specific.
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        report = json.loads(response.choices[0].message.content)
        logger.debug(f"Response data: {report}")
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
        # Step 1: Vector Search Agent (No change here)
        vector_res = await vector_search_agent(VectorSearchRequest(query=request.query))
        vector_findings = {"summary": f"Found {len(vector_res['data'])} documents.", "top_hits": vector_res['data'][:3]}
        logger.info(f"[{session_id}] Vector Search complete.")

        # Step 2: Literature Analysis Agent
        # The 'prior_context' will now be the full findings from the vector search
        lit_res = await literature_analysis_agent(LiteratureAnalysisRequest(query=request.query, prior_context=json.dumps(vector_findings)))
        literature_findings = lit_res['data']
        logger.info(f"[{session_id}] Literature Analysis complete.")

        # Step 3: Clinical Trials Agent
        # Pass the FULL literature_findings object, not just the summary
        clinical_res = await clinical_trials_agent(ClinicalTrialsRequest(condition=condition, intervention=intervention, literature_context=json.dumps(literature_findings)))
        clinical_findings = clinical_res['data']
        logger.info(f"[{session_id}] Clinical Trials Analysis complete.")

        # Step 4: Competitive Intelligence Agent
        # Pass the FULL clinical_findings object
        comp_res = await competitive_intelligence_agent(CompetitiveIntelRequest(market_area=request.query, clinical_context=json.dumps(clinical_findings)))
        competitive_findings = comp_res['data']
        logger.info(f"[{session_id}] Competitive Intelligence complete.")
        
        # Step 5: Regulatory Analysis Agent
        # Pass the FULL competitive_findings object
        reg_res = await regulatory_analysis_agent(RegulatoryAnalysisRequest(therapeutic_area=intervention, competitive_context=json.dumps(competitive_findings)))
        regulatory_findings = reg_res['data']
        logger.info(f"[{session_id}] Regulatory Analysis complete.")

        # Step 6: Medical Writing Agent (No change here, it already receives the full objects)
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
