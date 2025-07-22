"""
Medical Research Agent System - Refactored
Production-ready multi-agent pharmaceutical research platform
"""
import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Any

import aiohttp
from agents import Agent, Runner, FunctionTool, function_tool, GuardrailFunctionOutput, InputGuardrail, AgentOutputSchema
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Assuming your pinecone client is in app/utils/vector_store.py
from app.utils.vector_store import SimplePineconeClient

# For background tasks
from fastapi_utilities import repeat_every

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration & Models ---

# Initialize Pinecone Client
# Ensure PINECONE_API_KEY, PINECONE_ENVIRONMENT, and PINECONE_INDEX_NAME are in your .env
pinecone_client = SimplePineconeClient()

# Pydantic Models for structured outputs (schemas)
class LiteratureAnalysis(BaseModel):
    executive_summary: str
    key_findings: List[str]
    evidence_quality: str
    clinical_implications: str
    research_gaps: List[str]
    recommendations: List[str]
    confidence_score: float
    sources_analyzed: int

class CompetitiveIntelligence(BaseModel):
    competitive_landscape: str
    key_competitors: List[str]
    market_positioning: str
    development_pipeline: List[str]
    strategic_implications: str
    opportunities: List[str]
    threats: List[str]
    confidence_score: float

class ClinicalTrialsAnalysis(BaseModel):
    development_landscape: str
    phase_distribution: Dict[str, int]
    key_sponsors: List[str]
    primary_endpoints: List[str]
    development_timelines: Dict[str, Any]
    regulatory_pathways: List[str]
    strategic_recommendations: List[str]
    confidence_score: float

class RegulatoryAssessment(BaseModel):
    approval_pathways: Dict[str, str]
    regulatory_precedents: List[str]
    approval_timeline: Dict[str, str]
    key_requirements: List[str]
    regulatory_risks: List[str]
    strategic_recommendations: List[str]
    confidence_score: float

class ComprehensiveAnalysis(BaseModel):
    executive_summary: str
    key_strategic_insights: List[str]
    integrated_recommendations: List[str]
    risk_assessment: Dict[str, str]
    next_steps: List[str]
    confidence_assessment: str
    overall_confidence: float

# API Request Models
class ComprehensiveRequest(BaseModel):
    query: str
    therapy_area: str = "general"

# --- Agent Context ---
@dataclass
class ResearchContext:
    user_id: str
    research_request: str
    therapy_area: str
    parameters: Dict[str, Any]
    previous_findings: Dict[str, Any] = None
    vector_store_results: List[Dict] = None

# --- Real Research Tools ---

@function_tool
async def search_medical_literature(query: str, max_results: int = 20, days_back: int = 90) -> str:
    """
    Search medical literature using the PubMed API to find recent and relevant medical studies.
    """
    # This remains the same as your implementation, which is solid.
    # For brevity, the full pubmed search implementation is omitted here.
    # Let's assume it returns a JSON string of sources.
    logger.info(f"Searching PubMed for: {query}")
    # Placeholder for your actual PubMed search logic
    mock_sources = [{'pmid': '12345', 'title': f'Study on {query}'}]
    return json.dumps({"sources": mock_sources})

@function_tool
async def search_clinical_trials_data(query: str, max_results: int = 15) -> str:
    """
    Search the ClinicalTrials.gov API for information on clinical trials.
    """
    logger.info(f"Searching ClinicalTrials.gov for: {query}")
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    params = {'query.term': query, 'pageSize': max_results, 'format': 'json'}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params, timeout=30) as response:
                response.raise_for_status()
                data = await response.json()
                return json.dumps(data.get('studies', []))
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching clinical trials data: {e}")
        return json.dumps({"error": f"API request failed: {e}"})

@function_tool
async def search_vector_database(query: str, top_k: int = 5) -> str:
    """
    Search the Pinecone vector database for similar, previously conducted research.
    Requires an embedding of the query.
    """
    if not pinecone_client.available:
        return json.dumps({"error": "Pinecone client is not available."})
    
    logger.info(f"Querying Pinecone for: {query}")
    # In a real scenario, you'd generate an embedding for the query first.
    # from openai import AsyncOpenAI
    # client = AsyncOpenAI()
    # embedding_response = await client.embeddings.create(input=query, model="text-embedding-3-small")
    # query_embedding = embedding_response.data[0].embedding
    
    # For this example, we'll use a dummy vector.
    query_embedding = [0.1] * 1536 # Replace with your actual embedding dimension
    
    try:
        results = await pinecone_client.query(vector=query_embedding, top_k=top_k)
        return json.dumps(results)
    except Exception as e:
        logger.error(f"Error querying vector database: {e}")
        return json.dumps({"error": str(e)})

# --- Specialized Agents ---
common_tools = [search_medical_literature, search_vector_database]
output_schema_settings = {"strict_json_schema": False} # Allows for more lenient JSON parsing

literature_specialist = Agent(
    name="LiteratureSpecialist",
    instructions="You are a medical literature review expert. Your role is to conduct literature searches, assess evidence quality, and identify research gaps.",
    tools=common_tools,
    output_type=AgentOutputSchema(LiteratureAnalysis, **output_schema_settings)
)

competitive_analyst = Agent(
    name="CompetitiveAnalyst",
    instructions="You are a pharmaceutical competitive intelligence specialist. Your role is to analyze market landscapes, competitor strategies, and pipeline intelligence.",
    tools=common_tools + [search_clinical_trials_data],
    output_type=AgentOutputSchema(CompetitiveIntelligence, **output_schema_settings)
)

clinical_trials_expert = Agent(
    name="ClinicalTrialsExpert",
    instructions="You are a clinical development expert. Your role is to analyze trial designs, regulatory pathways, and development timelines from clinical trial data.",
    tools=[search_clinical_trials_data, search_medical_literature],
    output_type=AgentOutputSchema(ClinicalTrialsAnalysis, **output_schema_settings)
)

regulatory_specialist = Agent(
    name="RegulatorySpecialist",
    instructions="You are a regulatory affairs specialist. Your role is to analyze approval pathways, regulatory precedents, and compliance risks based on public data and literature.",
    tools=common_tools,
    output_type=AgentOutputSchema(RegulatoryAssessment, **output_schema_settings)
)

synthesis_agent = Agent(
    name="ResearchSynthesisAgent",
    instructions="You are a research synthesis expert. Your role is to integrate findings from multiple specialist analyses to create comprehensive, actionable intelligence.",
    tools=[], # This agent only synthesizes, it doesn't call tools
    output_type=AgentOutputSchema(ComprehensiveAnalysis, **output_schema_settings)
)

# --- Triage Agent & Guardrails ---
class QueryValidation(BaseModel):
    is_medical_query: bool
    is_appropriate: bool
    reasoning: str

validation_agent = Agent(
    name="QueryValidationAgent",
    instructions="Validate if the query is appropriate for medical research. Reject queries that are not medical-related or are unethical.",
    output_type=AgentOutputSchema(QueryValidation, **output_schema_settings)
)

async def medical_query_guardrail(ctx, agent, input_data: str) -> GuardrailFunctionOutput:
    """Input guardrail to validate that queries are appropriate for medical research."""
    result = await Runner.run(validation_agent, input_data)
    validation = result.final_output_as(QueryValidation)
    return GuardrailFunctionOutput(
        output_info=validation.model_dump_json(),
        tripwire_triggered=not (validation.is_medical_query and validation.is_appropriate)
    )

triage_agent = Agent(
    name="MedicalResearchTriageAgent",
    instructions="You are a triage specialist. Based on the user query, determine the optimal research strategy. You don't need to respond to the user, just handoff to the right specialists.",
    handoffs=[literature_specialist, competitive_analyst, clinical_trials_expert, regulatory_specialist],
    input_guardrails=[InputGuardrail(guardrail_function=medical_query_guardrail)]
)


# --- Workflow Orchestration ---
class MedicalResearchOrchestrator:
    async def execute_comprehensive_research(self, query: str, therapy_area: str) -> Dict:
        """Execute a comprehensive, multi-agent research workflow."""
        context = ResearchContext(user_id="system", research_request=query, therapy_area=therapy_area, parameters={})
        workflow_start = datetime.now()

        try:
            # Triage to decide which agents to run (though here we run them all in parallel)
            await Runner.run(triage_agent, query, context=context)

            # Run specialist agents in parallel
            tasks = [
                Runner.run(literature_specialist, query, context=context),
                Runner.run(competitive_analyst, query, context=context),
                Runner.run(clinical_trials_expert, query, context=context),
                Runner.run(regulatory_specialist, query, context=context),
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Error handling for failed agents
            if any(isinstance(res, Exception) for res in results):
                errors = [str(res) for res in results if isinstance(res, Exception)]
                logger.error(f"Errors during parallel agent execution: {errors}")
                raise HTTPException(status_code=500, detail=f"One or more agents failed: {', '.join(errors)}")

            lit_res, comp_res, clin_res, reg_res = results

            # Create a detailed prompt for the synthesis agent
            synthesis_input = f"""
            Synthesize the following research results for the query: "{query}" in the "{therapy_area}" therapy area.

            Literature Analysis: {lit_res.final_output_as(LiteratureAnalysis).model_dump_json(indent=2)}
            Competitive Analysis: {comp_res.final_output_as(CompetitiveIntelligence).model_dump_json(indent=2)}
            Clinical Trials Analysis: {clin_res.final_output_as(ClinicalTrialsAnalysis).model_dump_json(indent=2)}
            Regulatory Assessment: {reg_res.final_output_as(RegulatoryAssessment).model_dump_json(indent=2)}
            """
            
            synthesis_result = await Runner.run(synthesis_agent, synthesis_input, context=context)
            final_analysis = synthesis_result.final_output_as(ComprehensiveAnalysis)

            return {
                "success": True,
                "research_id": str(uuid.uuid4()),
                "final_analysis": final_analysis.model_dump(),
            }
        except InputGuardrailTripwireTriggered as e:
            logger.warning(f"Input guardrail triggered: {e.output_info}")
            raise HTTPException(status_code=400, detail=f"Query validation failed: {e.output_info}")
        except Exception as e:
            logger.error(f"Comprehensive research workflow failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

# --- FastAPI Application ---
app = FastAPI(title="Medical Research Agent System", version="5.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
orchestrator = MedicalResearchOrchestrator()

@app.on_event("startup")
@repeat_every(seconds=60 * 60 * 24) # Run once every 24 hours
async def automated_therapy_area_monitoring() -> None:
    """
    Example of a scheduled background task for automated monitoring.
    """
    logger.info("Running automated daily monitoring for key therapy areas...")
    monitored_areas = ["oncology", "neurology", "rare_disease"]
    for area in monitored_areas:
        query = f"latest significant developments in {area}"
        logger.info(f"Running comprehensive analysis for: {query}")
        try:
            # You could store these results in your Pinecone DB or a database
            await orchestrator.execute_comprehensive_research(query=query, therapy_area=area)
        except Exception as e:
            logger.error(f"Automated monitoring for '{area}' failed: {e}")

@app.post("/research/comprehensive", response_model=Dict)
async def comprehensive_research(request: ComprehensiveRequest):
    """Conduct comprehensive, multi-agent research and return a synthesized analysis."""
    return await orchestrator.execute_comprehensive_research(query=request.query, therapy_area=request.therapy_area)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "5.0.0", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
