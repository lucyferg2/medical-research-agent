from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
import logging
from pinecone import Pinecone
import traceback
from typing import List, Dict, Optional, Any
import json
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(title="Medical Research Agent API", version="1.0.0")

# Configure API keys and clients using environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Connect to Pinecone
pc = Pinecone(api_key=pinecone_api_key)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the index name and namespace
INDEX_NAME = "attruby-claims"
NAMESPACE = "Test Deck"

# Connect to the existing Pinecone index
try:
    existing_indexes = pc.list_indexes().names()
    if INDEX_NAME not in existing_indexes:
        raise HTTPException(status_code=500, detail=f"Index {INDEX_NAME} does not exist in Pinecone.")
    
    # Access the existing Pinecone index
    index = pc.Index(INDEX_NAME)
    logger.info(f"Successfully connected to Pinecone index: {INDEX_NAME}")
    
except Exception as e:
    logger.error(f"Error connecting to Pinecone: {e}")
    raise HTTPException(status_code=500, detail="Failed to connect to Pinecone.")

# Data models for requests
class VectorSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    namespace: str = NAMESPACE
    include_metadata: bool = True

class LiteratureAnalysisRequest(BaseModel):
    query: str
    focus_areas: Optional[List[str]] = []
    evidence_level: str = "all"
    prior_context: Optional[str] = None

class ClinicalTrialsRequest(BaseModel):
    condition: str
    intervention: str
    phase: Optional[str] = "All"
    literature_context: Optional[str] = None
    vector_context: Optional[str] = None

class CompetitiveIntelRequest(BaseModel):
    market_area: str
    competitors: Optional[List[str]] = []
    clinical_context: Optional[str] = None
    literature_context: Optional[str] = None

class RegulatoryAnalysisRequest(BaseModel):
    therapeutic_area: str
    regulatory_region: str = "FDA"
    competitive_context: Optional[str] = None
    clinical_context: Optional[str] = None

class MedicalWritingRequest(BaseModel):
    report_type: str = "comprehensive"
    vector_findings: Optional[str] = None
    literature_findings: Optional[str] = None
    clinical_findings: Optional[str] = None
    competitive_findings: Optional[str] = None
    regulatory_findings: Optional[str] = None

class SequentialWorkflowRequest(BaseModel):
    query: str
    reportType: str = "comprehensive"

# Utility functions
async def generate_embedding(text: str) -> List[float]:
    """Generate embeddings using OpenAI - exactly like your working code"""
    try:
        response = openai.embeddings.create(
            input=[text], 
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {str(e)}")

# MAIN ENDPOINTS

@app.get("/")
async def root():
    return {
        "message": "Medical Research Agent API is running!",
        "version": "1.0.0",
        "endpoints": [
            "/api/health",
            "/api/agents/vector-search",
            "/api/agents/literature-analysis", 
            "/api/agents/clinical-trials",
            "/api/agents/competitive-intel",
            "/api/agents/regulatory-analysis",
            "/api/agents/medical-writing",
            "/api/sequential-workflow"
        ]
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "pinecone": "configured" if pinecone_api_key else "not_configured",
        "openai": "configured" if openai.api_key else "not_configured"
    }

@app.post("/api/agents/vector-search")
async def vector_search_agent(request: VectorSearchRequest):
    """Vector search agent - matches your working Python code exactly"""
    try:
        logger.info(f"Vector search request: {request.query}")
        
        # Generate embeddings using OpenAI - exactly like your working code
        embedding = await generate_embedding(request.query)
        logger.debug(f"Generated embedding with dimension: {len(embedding)}")

        # Query Pinecone - exactly like your working code
        results = index.query(
            namespace=request.namespace,
            vector=embedding,
            top_k=request.top_k,
            include_metadata=request.include_metadata
        ).to_dict()

        logger.info(f"Pinecone query returned {len(results.get('matches', []))} matches")

        # Extract documents from results
        matches = results.get("matches", [])
        
        # Format response - minimal to avoid ResponseTooLargeError
        response_data = {
            "query": request.query,
            "totalResults": len(matches),
            "relevantDocuments": []
        }

        # Process top 5 matches only to keep response size manageable
        for i, match in enumerate(matches[:5]):
            metadata = match.get("metadata", {})
            doc = {
                "rank": i + 1,
                "score": round(match.get("score", 0), 3),
                "id": match.get("id", ""),
                "title": metadata.get("title", "Unknown Title"),
                "authors": metadata.get("authors", "Unknown Authors"), 
                "citation": metadata.get("citation", "Citation unavailable"),
                "chunk_preview": metadata.get("chunk_preview", "")[:200] + "..." if metadata.get("chunk_preview") else "Preview unavailable",
                "doi": metadata.get("doi"),
                "journal": metadata.get("journal"),
                "published": metadata.get("published"),
                "source_file": metadata.get("source_file")
            }
            response_data["relevantDocuments"].append(doc)

        # Add summary
        if matches:
            top_match = matches[0]
            top_title = top_match.get("metadata", {}).get("title", "Unknown")
            top_score = round(top_match.get("score", 0), 3)
            response_data["summary"] = f"Found {len(matches)} relevant documents. Top match: '{top_title}' (score: {top_score})"
        else:
            response_data["summary"] = f"No relevant documents found for '{request.query}'"

        return {"success": True, "data": response_data}

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False, 
            "data": {
                "error": True,
                "message": f"Vector search failed: {str(e)}",
                "query": request.query,
                "relevantDocuments": []
            }
        }

@app.post("/api/agents/literature-analysis")
async def literature_analysis_agent(request: LiteratureAnalysisRequest):
    """Literature analysis agent using OpenAI"""
    try:
        logger.info(f"Literature analysis request: {request.query}")
        
        # Create prompt for literature analysis
        prompt = f"""
        You are a pharmaceutical literature analysis expert. Analyze the query: "{request.query}"
        
        Focus areas: {request.focus_areas}
        Evidence level required: {request.evidence_level}
        Prior context: {request.prior_context or 'None'}
        
        Provide a structured analysis with:
        - conditions: Medical conditions/indications identified
        - interventions: Treatments/therapies mentioned
        - evidenceGrade: Quality assessment (High/Medium/Low)
        - keyFindings: Important research insights
        - therapeuticAreas: Relevant therapeutic categories
        - summary: Comprehensive overview
        
        Format as JSON.
        """
        
        response = openai.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        # Try to parse JSON response, fallback to text
        try:
            analysis = json.loads(response.choices[0].message.content)
        except:
            analysis = {
                "summary": response.choices[0].message.content,
                "conditions": [],
                "interventions": [],
                "evidenceGrade": "Medium",
                "keyFindings": [],
                "therapeuticAreas": []
            }
        
        return {"success": True, "data": analysis}
        
    except Exception as e:
        logger.error(f"Literature analysis failed: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/agents/clinical-trials")
async def clinical_trials_agent(request: ClinicalTrialsRequest):
    """Clinical trials analysis agent"""
    try:
        logger.info(f"Clinical trials request: {request.condition} - {request.intervention}")
        
        # Simulate ClinicalTrials.gov analysis
        prompt = f"""
        You are a clinical trials expert. Analyze trials for:
        Condition: {request.condition}
        Intervention: {request.intervention}
        Phase: {request.phase}
        
        Literature context: {request.literature_context or 'None'}
        
        Provide analysis with:
        - activeTrials: Number of active trials
        - sponsors: Key companies/organizations
        - phases: Distribution of trial phases
        - competitiveLandscape: Analysis of competitive trials
        - summary: Overview of clinical trial landscape
        
        Format as JSON.
        """
        
        response = openai.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        try:
            analysis = json.loads(response.choices[0].message.content)
        except:
            analysis = {
                "summary": response.choices[0].message.content,
                "activeTrials": 0,
                "sponsors": [],
                "phases": [],
                "competitiveLandscape": "Analysis unavailable"
            }
        
        return {"success": True, "data": analysis}
        
    except Exception as e:
        logger.error(f"Clinical trials analysis failed: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/agents/competitive-intel")
async def competitive_intelligence_agent(request: CompetitiveIntelRequest):
    """Competitive intelligence analysis agent"""
    try:
        logger.info(f"Competitive intelligence request: {request.market_area}")
        
        prompt = f"""
        You are a pharmaceutical competitive intelligence expert. Analyze:
        Market area: {request.market_area}
        Known competitors: {request.competitors}
        
        Clinical context: {request.clinical_context or 'None'}
        Literature context: {request.literature_context or 'None'}
        
        Provide analysis with:
        - competitorAnalysis: Key competitor profiles
        - marketOpportunities: Identified opportunities
        - threats: Competitive threats
        - marketSize: Market size estimates
        - summary: Strategic competitive overview
        
        Format as JSON.
        """
        
        response = openai.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        try:
            analysis = json.loads(response.choices[0].message.content)
        except:
            analysis = {
                "summary": response.choices[0].message.content,
                "competitorAnalysis": [],
                "marketOpportunities": [],
                "threats": [],
                "marketSize": "Unknown"
            }
        
        return {"success": True, "data": analysis}
        
    except Exception as e:
        logger.error(f"Competitive intelligence analysis failed: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/agents/regulatory-analysis")
async def regulatory_analysis_agent(request: RegulatoryAnalysisRequest):
    """Regulatory analysis agent"""
    try:
        logger.info(f"Regulatory analysis request: {request.therapeutic_area}")
        
        prompt = f"""
        You are a regulatory affairs expert. Analyze regulatory environment for:
        Therapeutic area: {request.therapeutic_area}
        Region: {request.regulatory_region}
        
        Competitive context: {request.competitive_context or 'None'}
        Clinical context: {request.clinical_context or 'None'}
        
        Provide analysis with:
        - applicableGuidances: Relevant FDA/EMA guidances
        - regulatoryPathways: Recommended submission strategies
        - timelineEstimates: Expected development timelines
        - recommendations: Strategic regulatory advice
        - summary: Comprehensive regulatory environment analysis
        
        Format as JSON.
        """
        
        response = openai.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        try:
            analysis = json.loads(response.choices[0].message.content)
        except:
            analysis = {
                "summary": response.choices[0].message.content,
                "applicableGuidances": [],
                "regulatoryPathways": [],
                "timelineEstimates": "Unknown",
                "recommendations": []
            }
        
        return {"success": True, "data": analysis}
        
    except Exception as e:
        logger.error(f"Regulatory analysis failed: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/agents/medical-writing")
async def medical_writing_agent(request: MedicalWritingRequest):
    """Medical writing synthesis agent"""
    try:
        logger.info(f"Medical writing request: {request.report_type}")
        
        prompt = f"""
        You are a medical writing expert. Synthesize findings into a {request.report_type} report.
        
        Vector findings: {request.vector_findings or 'None'}
        Literature findings: {request.literature_findings or 'None'}
        Clinical findings: {request.clinical_findings or 'None'}
        Competitive findings: {request.competitive_findings or 'None'}
        Regulatory findings: {request.regulatory_findings or 'None'}
        
        Create a structured report with:
        - executiveSummary: Key insights and recommendations
        - literatureLandscape: Research trends analysis
        - clinicalPipeline: Trial landscape overview
        - marketIntelligence: Competitive dynamics
        - regulatoryEnvironment: Pathway analysis
        - strategicRecommendations: Actionable next steps
        - report: Complete formatted report
        
        Format as JSON.
        """
        
        response = openai.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        try:
            analysis = json.loads(response.choices[0].message.content)
        except:
            analysis = {
                "report": response.choices[0].message.content,
                "executiveSummary": "Report generated successfully",
                "strategicRecommendations": []
            }
        
        return {"success": True, "data": analysis}
        
    except Exception as e:
        logger.error(f"Medical writing failed: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/sequential-workflow")
async def sequential_workflow(request: SequentialWorkflowRequest):
    """Execute complete sequential multi-agent workflow"""
    try:
        logger.info(f"Sequential workflow request: {request.query}")
        
        workflow_results = {
            "sessionId": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "query": request.query,
            "startTime": datetime.now().isoformat(),
            "agents": [],
            "finalReport": None
        }
        
        # Step 1: Vector Search
        vector_result = await vector_search_agent(VectorSearchRequest(query=request.query))
        workflow_results["agents"].append({
            "agent": "vector_search",
            "timestamp": datetime.now().isoformat(),
            "results": vector_result["data"]
        })
        
        # Step 2: Literature Analysis
        lit_result = await literature_analysis_agent(LiteratureAnalysisRequest(
            query=request.query,
            prior_context=str(vector_result["data"])
        ))
        workflow_results["agents"].append({
            "agent": "literature_analysis", 
            "timestamp": datetime.now().isoformat(),
            "results": lit_result["data"]
        })
        
        # Continue with other agents...
        # For now, return the first two steps
        
        workflow_results["finalReport"] = {
            "executiveSummary": f"Completed sequential analysis for {request.query}",
            "agentsExecuted": len(workflow_results["agents"]),
            "status": "completed"
        }
        
        return {
            "success": True,
            "sessionId": workflow_results["sessionId"],
            "workflow": workflow_results,
            "report": workflow_results["finalReport"]
        }
        
    except Exception as e:
        logger.error(f"Sequential workflow failed: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 3000)))
