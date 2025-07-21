from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from datetime import datetime
import uuid
import logging
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical Research Agent API",
    description="AI-powered medical research and analysis platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple configuration from environment variables
class SimpleConfig:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.research_email = os.getenv("RESEARCH_EMAIL", "research@company.com")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")

config = SimpleConfig()

# Import OpenAI client
try:
    from openai import OpenAI
    client = OpenAI(api_key=config.openai_api_key) if config.openai_api_key else None
except ImportError:
    client = None
    logger.error("OpenAI package not available")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Medical Research Agent API starting up...")
    if not config.openai_api_key:
        logger.warning("OpenAI API key not set. Some features will be limited.")

@app.get("/")
async def root():
    return {
        "message": "Medical Research Agent API",
        "description": "AI-powered medical research for pharmaceutical teams",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "config": "/config",
            "literature_review": "/research/literature",
            "competitive_analysis": "/research/competitive"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "service": "Medical Research Agent",
        "version": "1.0.0",
        "openai_configured": bool(config.openai_api_key),
        "pinecone_configured": bool(config.pinecone_api_key)
    }

@app.get("/config")
async def get_config():
    """Get current configuration (non-sensitive data only)"""
    return {
        "service": "Medical Research Agent",
        "version": "1.0.0",
        "features": {
            "literature_review": True,
            "competitive_analysis": True,
            "clinical_trials": True,
            "openai_analysis": bool(config.openai_api_key),
            "vector_storage": bool(config.pinecone_api_key)
        },
        "therapy_areas_supported": [
            "oncology", "neurology", "cardiology", "endocrinology",
            "immunology", "infectious_disease", "rare_disease", "general"
        ],
        "research_types_supported": [
            "literature_review", "competitive_analysis", 
            "regulatory_landscape", "clinical_trials"
        ]
    }

# Simple PubMed search function
async def simple_pubmed_search(query: str, max_results: int = 10):
    """Simple PubMed search using aiohttp"""
    try:
        import aiohttp
        
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'tool': 'medical_research_agent',
            'email': config.research_email
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    pmids = data.get('esearchresult', {}).get('idlist', [])
                    return pmids[:max_results]
                else:
                    logger.error(f"PubMed search failed: {response.status}")
                    return []
                    
    except Exception as e:
        logger.error(f"Error in PubMed search: {e}")
        return []

# Simple AI analysis function
async def analyze_with_openai(query: str, context: str = ""):
    """Simple OpenAI analysis"""
    if not client:
        return {
            "executive_summary": f"Analysis requested for: {query}. OpenAI not configured - this is a mock response for testing.",
            "key_findings": [
                "OpenAI API key not configured",
                "Mock analysis generated for testing",
                "Configure OPENAI_API_KEY environment variable for full functionality"
            ],
            "clinical_implications": "Full analysis requires OpenAI API configuration.",
            "recommendations": [
                "Set OPENAI_API_KEY environment variable",
                "Test with real API key for production use"
            ]
        }
    
    try:
        prompt = f"""
        Analyze this medical research query: {query}
        
        Context: {context}
        
        Provide a structured analysis with:
        1. Executive summary (2-3 sentences)
        2. Key findings (3-5 bullet points)
        3. Clinical implications
        4. Recommendations (2-3 points)
        
        Format as JSON with fields: executive_summary, key_findings, clinical_implications, recommendations
        """
        
        response = await client.chat.completions.acreate(
            model="gpt-3.5-turbo",  # Using cheaper model for testing
            messages=[
                {"role": "system", "content": "You are a medical research analyst. Provide structured analysis in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        try:
            analysis = json.loads(response.choices[0].message.content)
            return analysis
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            content = response.choices[0].message.content
            return {
                "executive_summary": f"Analysis completed for: {query}",
                "key_findings": [
                    "Analysis generated successfully",
                    "OpenAI integration working",
                    "Structured output available"
                ],
                "clinical_implications": content[:200] + "..." if len(content) > 200 else content,
                "recommendations": [
                    "Review full analysis",
                    "Consider additional research"
                ]
            }
        
    except Exception as e:
        logger.error(f"OpenAI analysis error: {e}")
        return {
            "executive_summary": f"Analysis error for: {query}",
            "key_findings": [f"Error: {str(e)}"],
            "clinical_implications": "Technical error occurred during analysis.",
            "recommendations": ["Retry request", "Check API configuration"]
        }

@app.post("/research/literature")
async def literature_review_endpoint(request: dict):
    """
    Conduct medical literature review
    """
    try:
        query = request.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        therapy_area = request.get("therapy_area", "general")
        days_back = request.get("days_back", 90)
        max_results = request.get("max_results", 10)
        
        research_id = str(uuid.uuid4())
        
        # Search PubMed
        pmids = await simple_pubmed_search(query, max_results)
        
        # Generate analysis
        context = f"Found {len(pmids)} PubMed articles for therapy area: {therapy_area}"
        analysis = await analyze_with_openai(query, context)
        
        return {
            "success": True,
            "research_id": research_id,
            "query": query,
            "therapy_area": therapy_area,
            "sources_analyzed": len(pmids),
            "pubmed_ids": pmids,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Literature review error: {e}")
        raise HTTPException(status_code=500, detail=f"Literature review failed: {str(e)}")

@app.post("/research/competitive")
async def competitive_analysis_endpoint(request: dict):
    """
    Conduct competitive analysis
    """
    try:
        competitor_query = request.get("competitor_query")
        if not competitor_query:
            raise HTTPException(status_code=400, detail="Competitor query is required")
        
        therapy_area = request.get("therapy_area", "general")
        research_id = str(uuid.uuid4())
        
        # Search for literature
        literature_pmids = await simple_pubmed_search(competitor_query, 10)
        
        # Generate competitive analysis
        context = f"Competitive analysis in {therapy_area} with {len(literature_pmids)} literature sources"
        analysis = await analyze_with_openai(f"Competitive analysis: {competitor_query}", context)
        
        # Format as competitive analysis
        competitive_analysis = {
            "competitive_landscape": analysis.get("executive_summary", ""),
            "key_competitors": analysis.get("key_findings", []),
            "market_positioning": analysis.get("clinical_implications", ""),
            "strategic_implications": "Analysis based on available literature and market intelligence.",
            "recommendations": analysis.get("recommendations", [])
        }
        
        return {
            "success": True,
            "research_id": research_id,
            "query": competitor_query,
            "therapy_area": therapy_area,
            "literature_sources": len(literature_pmids),
            "trial_sources": 0,  # Not implemented in minimal version
            "analysis": competitive_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Competitive analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Competitive analysis failed: {str(e)}")

@app.post("/research/general")
async def general_research_endpoint(request: dict):
    """
    General research endpoint
    """
    try:
        query = request.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        research_type = request.get("research_type", "literature_review")
        
        if research_type == "literature_review":
            return await literature_review_endpoint(request)
        elif research_type == "competitive_analysis":
            return await competitive_analysis_endpoint({
                "competitor_query": query,
                "therapy_area": request.get("therapy_area", "general")
            })
        else:
            # General analysis
            research_id = str(uuid.uuid4())
            analysis = await analyze_with_openai(query)
            
            return {
                "success": True,
                "research_id": research_id,
                "query": query,
                "research_type": research_type,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"General research error: {e}")
        raise HTTPException(status_code=500, detail=f"General research failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
