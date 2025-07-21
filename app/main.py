from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uuid
import logging
import os
import json
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical Research Agent API",
    description="AI-powered medical research platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RESEARCH_EMAIL = os.getenv("RESEARCH_EMAIL", "research@company.com")

# OpenAI client
openai_client = None
try:
    from openai import OpenAI
    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized")
    else:
        logger.warning("OpenAI API key not set")
except ImportError:
    logger.error("OpenAI package not available")

@app.get("/")
async def root():
    return {
        "message": "Medical Research Agent API - Nuclear Minimal Version",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "literature": "/research/literature"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Medical Research Agent",
        "version": "1.0.0-minimal",
        "openai_available": openai_client is not None
    }

def sync_pubmed_search(query: str, max_results: int = 5):
    """Synchronous PubMed search using requests instead of aiohttp"""
    try:
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'tool': 'medical_research_agent',
            'email': RESEARCH_EMAIL
        }
        
        response = requests.get(search_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            pmids = data.get('esearchresult', {}).get('idlist', [])
            logger.info(f"Found {len(pmids)} PMIDs for query: {query}")
            return pmids[:max_results]
        else:
            logger.error(f"PubMed search failed: {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"Error in PubMed search: {e}")
        return []

def analyze_with_openai_sync(query: str, pmid_count: int = 0):
    """Synchronous OpenAI analysis"""
    if not openai_client:
        return {
            "executive_summary": f"Mock analysis for: {query}. Found {pmid_count} PubMed articles.",
            "key_findings": [
                "OpenAI not configured - using mock response",
                f"PubMed search found {pmid_count} articles",
                "Set OPENAI_API_KEY for real analysis"
            ],
            "clinical_implications": "Configure OpenAI API for full analysis capabilities.",
            "recommendations": [
                "Set OPENAI_API_KEY environment variable",
                "Test with real medical queries"
            ]
        }
    
    try:
        prompt = f"""
        Medical research analysis for: {query}
        Found {pmid_count} relevant PubMed articles.
        
        Provide analysis as JSON with these fields:
        - executive_summary: 2-3 sentences
        - key_findings: array of 3-5 findings
        - clinical_implications: paragraph
        - recommendations: array of 2-3 recommendations
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical research analyst. Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        try:
            analysis = json.loads(response.choices[0].message.content)
            logger.info("OpenAI analysis completed successfully")
            return analysis
        except json.JSONDecodeError:
            content = response.choices[0].message.content
            return {
                "executive_summary": f"Analysis completed for {query}",
                "key_findings": ["OpenAI analysis generated", "JSON parsing handled"],
                "clinical_implications": content[:300] + "..." if len(content) > 300 else content,
                "recommendations": ["Review analysis", "Consider follow-up research"]
            }
        
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return {
            "executive_summary": f"Analysis error for {query}",
            "key_findings": [f"Error: {str(e)}"],
            "clinical_implications": "Technical error during analysis",
            "recommendations": ["Retry request", "Check API status"]
        }

@app.post("/research/literature")
async def literature_review(request: dict):
    """Literature review endpoint - fully synchronous to avoid async issues"""
    try:
        query = request.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        therapy_area = request.get("therapy_area", "general")
        max_results = min(request.get("max_results", 10), 20)  # Cap at 20
        
        research_id = str(uuid.uuid4())
        
        # Search PubMed synchronously
        pmids = sync_pubmed_search(query, max_results)
        
        # Analyze with OpenAI
        analysis = analyze_with_openai_sync(query, len(pmids))
        
        response = {
            "success": True,
            "research_id": research_id,
            "query": query,
            "therapy_area": therapy_area,
            "sources_analyzed": len(pmids),
            "pubmed_ids": pmids,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
            "version": "nuclear-minimal"
        }
        
        logger.info(f"Literature review completed for: {query}")
        return response
        
    except Exception as e:
        logger.error(f"Literature review error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/research/competitive")
async def competitive_analysis(request: dict):
    """Competitive analysis endpoint"""
    try:
        competitor_query = request.get("competitor_query", request.get("query", ""))
        if not competitor_query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        therapy_area = request.get("therapy_area", "general")
        research_id = str(uuid.uuid4())
        
        # Search literature
        pmids = sync_pubmed_search(competitor_query, 8)
        
        # Analyze
        analysis = analyze_with_openai_sync(f"Competitive analysis: {competitor_query}", len(pmids))
        
        return {
            "success": True,
            "research_id": research_id,
            "query": competitor_query,
            "therapy_area": therapy_area,
            "literature_sources": len(pmids),
            "analysis": {
                "competitive_landscape": analysis.get("executive_summary", ""),
                "key_competitors": analysis.get("key_findings", []),
                "strategic_implications": analysis.get("clinical_implications", ""),
                "recommendations": analysis.get("recommendations", [])
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Competitive analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Competitive analysis failed: {str(e)}")

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {
        "message": "Test successful!",
        "timestamp": datetime.now().isoformat(),
        "dependencies": {
            "fastapi": "✅",
            "uvicorn": "✅", 
            "openai": "✅" if openai_client else "❌",
            "requests": "✅"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
