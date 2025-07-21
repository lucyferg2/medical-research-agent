from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
from datetime import datetime
import uuid
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical Research Agent API",
    description="AI-powered medical research and analysis platform for pharmaceutical teams",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import your modules (will create these)
try:
    from app.config import settings
    from app.models.schemas import (
        ResearchRequest, LiteratureRequest, CompetitiveRequest,
        ResearchResponse, ErrorResponse
    )
    from app.utils.vector_store import vector_store
    from app.agents.medical_agents import (
        conduct_literature_review, conduct_competitive_analysis
    )
except ImportError as e:
    logger.warning(f"Import error: {e}. Some features may not be available.")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Initialize vector store if available
        if 'vector_store' in globals():
            await vector_store.initialize_index()
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Startup error: {e}")

@app.get("/")
async def root():
    return {
        "message": "Medical Research Agent API",
        "description": "AI-powered medical research for pharmaceutical teams",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "literature_review": "/research/literature",
            "competitive_analysis": "/research/competitive",
            "general_research": "/research/general"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now(),
        "service": "Medical Research Agent",
        "version": "1.0.0"
    }

@app.post("/research/literature")
async def literature_review_endpoint(request: dict):
    """
    Conduct medical literature review
    
    Example request:
    {
        "query": "CAR-T cell therapy multiple myeloma",
        "therapy_area": "oncology",
        "days_back": 90,
        "max_results": 20
    }
    """
    try:
        query = request.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        therapy_area = request.get("therapy_area", "general")
        days_back = request.get("days_back", 90)
        max_results = request.get("max_results", 20)
        
        # For now, return a mock response until agents are fully set up
        research_id = str(uuid.uuid4())
        
        # If the agent function is available, use it
        if 'conduct_literature_review' in globals():
            result = await conduct_literature_review(
                query=query,
                therapy_area=therapy_area,
                days_back=days_back
            )
            
            if result['success']:
                return {
                    "success": True,
                    "research_id": research_id,
                    "query": query,
                    "therapy_area": therapy_area,
                    "sources_analyzed": result['sources_count'],
                    "analysis": result['analysis'],
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise HTTPException(status_code=500, detail=result['error'])
        else:
            # Mock response for initial testing
            return {
                "success": True,
                "research_id": research_id,
                "query": query,
                "therapy_area": therapy_area,
                "sources_analyzed": 15,
                "analysis": {
                    "executive_summary": f"Literature review completed for '{query}' in {therapy_area}. This is a mock response for testing.",
                    "key_findings": [
                        "Recent developments show promising results",
                        "Multiple clinical trials are ongoing",
                        "Safety profile appears favorable"
                    ],
                    "clinical_implications": "The findings suggest potential for clinical application with continued research.",
                    "recommendations": [
                        "Monitor ongoing clinical trials",
                        "Review safety data from recent studies",
                        "Consider competitive landscape analysis"
                    ]
                },
                "timestamp": datetime.now().isoformat(),
                "note": "This is a mock response. Full agent functionality will be available once all components are deployed."
            }
            
    except Exception as e:
        logger.error(f"Literature review error: {e}")
        raise HTTPException(status_code=500, detail=f"Literature review failed: {str(e)}")

@app.post("/research/competitive")
async def competitive_analysis_endpoint(request: dict):
    """
    Conduct competitive analysis
    
    Example request:
    {
        "competitor_query": "pembrolizumab lung cancer",
        "therapy_area": "oncology",
        "include_trials": true
    }
    """
    try:
        competitor_query = request.get("competitor_query")
        if not competitor_query:
            raise HTTPException(status_code=400, detail="Competitor query is required")
        
        therapy_area = request.get("therapy_area", "general")
        include_trials = request.get("include_trials", True)
        
        research_id = str(uuid.uuid4())
        
        # If the agent function is available, use it
        if 'conduct_competitive_analysis' in globals():
            result = await conduct_competitive_analysis(
                competitor_query=competitor_query,
                therapy_area=therapy_area
            )
            
            if result['success']:
                return {
                    "success": True,
                    "research_id": research_id,
                    "query": competitor_query,
                    "therapy_area": therapy_area,
                    "literature_sources": result['literature_sources'],
                    "trial_sources": result['trial_sources'],
                    "analysis": result['analysis'],
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise HTTPException(status_code=500, detail=result['error'])
        else:
            # Mock response for initial testing
            return {
                "success": True,
                "research_id": research_id,
                "query": competitor_query,
                "therapy_area": therapy_area,
                "literature_sources": 12,
                "trial_sources": 8,
                "analysis": {
                    "competitive_landscape": f"Analysis of {competitor_query} in {therapy_area} market. This is a mock response for testing.",
                    "key_competitors": [
                        "Company A - Leading market position",
                        "Company B - Strong pipeline",
                        "Company C - Emerging player"
                    ],
                    "market_positioning": "Current competitive dynamics show established players with emerging threats.",
                    "development_pipeline": [
                        "Phase III trials ongoing",
                        "Multiple Phase II programs",
                        "Early-stage research active"
                    ],
                    "strategic_implications": "Market entry timing and differentiation strategy are critical.",
                    "risk_assessment": "Moderate competitive risk with opportunities for differentiation"
                },
                "timestamp": datetime.now().isoformat(),
                "note": "This is a mock response. Full agent functionality will be available once all components are deployed."
            }
            
    except Exception as e:
        logger.error(f"Competitive analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Competitive analysis failed: {str(e)}")

@app.post("/research/general")
async def general_research_endpoint(request: dict):
    """
    General research endpoint that routes to appropriate specialized agents
    
    Example request:
    {
        "query": "What are the latest developments in CAR-T cell therapy?",
        "research_type": "literature_review",
        "therapy_area": "oncology",
        "max_sources": 20
    }
    """
    try:
        query = request.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        research_type = request.get("research_type", "literature_review")
        therapy_area = request.get("therapy_area", "general")
        max_sources = request.get("max_sources", 20)
        
        research_id = str(uuid.uuid4())
        
        # Route based on research type
        if research_type == "literature_review":
            return await literature_review_endpoint({
                "query": query,
                "therapy_area": therapy_area,
                "max_results": max_sources
            })
        elif research_type == "competitive_analysis":
            return await competitive_analysis_endpoint({
                "competitor_query": query,
                "therapy_area": therapy_area
            })
        else:
            # General mock response
            return {
                "success": True,
                "research_id": research_id,
                "query": query,
                "research_type": research_type,
                "therapy_area": therapy_area,
                "sources_analyzed": 18,
                "executive_summary": f"General research completed for '{query}' with focus on {research_type} in {therapy_area}.",
                "key_findings": [
                    "Research area shows active development",
                    "Multiple stakeholders involved",
                    "Regulatory landscape is evolving"
                ],
                "clinical_implications": "Findings suggest continued monitoring and analysis recommended.",
                "recommendations": [
                    "Continue tracking developments",
                    "Monitor regulatory changes",
                    "Assess competitive implications"
                ],
                "timestamp": datetime.now().isoformat(),
                "note": "This is a mock response. Full agent functionality will be available once all components are deployed."
            }
            
    except Exception as e:
        logger.error(f"General research error: {e}")
        raise HTTPException(status_code=500, detail=f"General research failed: {str(e)}")

@app.get("/research/history")
async def research_history(limit: int = 10):
    """Get research history from vector store"""
    try:
        # This would query your vector store for recent research
        # For now, return mock data
        return {
            "success": True,
            "message": "Research history endpoint",
            "limit": limit,
            "history": [
                {
                    "research_id": str(uuid.uuid4()),
                    "query": "CAR-T cell therapy safety",
                    "research_type": "literature_review",
                    "therapy_area": "oncology",
                    "timestamp": datetime.now().isoformat(),
                    "sources_analyzed": 15
                },
                {
                    "research_id": str(uuid.uuid4()),
                    "query": "GLP-1 agonist competition",
                    "research_type": "competitive_analysis",
                    "therapy_area": "endocrinology",
                    "timestamp": datetime.now().isoformat(),
                    "sources_analyzed": 23
                }
            ],
            "note": "This endpoint will connect to vector store when fully implemented"
        }
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")

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
            "vector_storage": "pinecone" if 'vector_store' in globals() else "not_configured"
        },
        "therapy_areas_supported": [
            "oncology",
            "neurology", 
            "cardiology",
            "endocrinology",
            "immunology",
            "infectious_disease",
            "rare_disease",
            "general"
        ],
        "research_types_supported": [
            "literature_review",
            "competitive_analysis",
            "regulatory_landscape",
            "clinical_trials"
        ]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
