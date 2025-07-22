"""
Medical Research Agent System - Final Version
Production-ready multi-agent pharmaceutical research platform with both specialized and comprehensive endpoints.
"""
import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Union

import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from pydantic.alias_generators import to_snake

# Use the real OpenAI Agents SDK
from agents import Agent, Runner, FunctionTool, function_tool, GuardrailFunctionOutput, InputGuardrail, AgentOutputSchema
from agents.exceptions import InputGuardrailTripwireTriggered, ModelBehaviorError

# Your custom Pinecone client
from app.utils.vector_store import SimplePineconeClient

# For the scheduled monitoring task
from fastapi_utilities import repeat_every

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models & Configuration ---

# A reusable base model to handle inconsistent JSON key casing from the AI
class CamelCaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_snake,  # Converts CamelCase/TitleCase from AI to snake_case for Python
        populate_by_name=True,     # Allows using both the alias and the original field name
    )

# Schemas for Agent Outputs
class LiteratureAnalysis(CamelCaseModel):
    executive_summary: str
    key_findings: List[str]
    evidence_quality: str
    clinical_implications: str
    research_gaps: List[str]
    recommendations: List[str]
    confidence_score: float
    sources_analyzed: int

class CompetitiveIntelligence(CamelCaseModel):
    competitive_landscape: str
    key_competitors: List[str]
    market_positioning: str
    development_pipeline: List[str]
    strategic_implications: str
    opportunities: List[str]
    threats: List[str]
    confidence_score: float

class ClinicalTrialsAnalysis(CamelCaseModel):
    development_landscape: str
    phase_distribution: Dict[str, Union[List[str], int]] # Flexible to handle different AI outputs
    key_sponsors: List[str]
    primary_endpoints: List[str]
    development_timelines: Dict[str, Any]
    regulatory_pathways: List[str]
    strategic_recommendations: List[str]
    confidence_score: float

class RegulatoryAssessment(CamelCaseModel):
    approval_pathways: Dict[str, str]
    regulatory_precedents: List[str]
    approval_timeline: Dict[str, str]
    key_requirements: List[str]
    regulatory_risks: List[str]
    strategic_recommendations: List[str]
    confidence_score: float

class ComprehensiveAnalysis(CamelCaseModel):
    executive_summary: str
    key_strategic_insights: List[str]
    integrated_recommendations: List[str]
    risk_assessment: Dict[str, str]
    next_steps: List[str]
    confidence_assessment: str
    overall_confidence: float

# Schemas for API Requests
class LiteratureRequest(BaseModel):
    query: str = Field(..., description="The research query for the literature review.")
    max_results: int = Field(5, description="Maximum number of sources to analyze.")

class CompetitiveRequest(BaseModel):
    query: str = Field(..., description="The query for the competitive analysis.")
    therapy_area: str = Field("general", description="The therapy area of focus.")

class ComprehensiveRequest(BaseModel):
    query: str
    therapy_area: str = "general"

# --- Agent Context ---
@dataclass
class ResearchContext:
    user_id: str
    research_request: str
    therapy_area: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    previous_findings: Dict[str, Any] = field(default_factory=dict)
    vector_store_results: List[Dict] = field(default_factory=list)

# --- Research Tools ---
pinecone_client = SimplePineconeClient()

@function_tool
async def search_medical_literature(query: str, max_results: int = 5) -> str:
    """Search PubMed for medical literature abstracts. Returns a JSON string of sources."""
    logger.info(f"Searching PubMed for '{query}' with max_results={max_results}")
    # This is where you would place your actual PubMed API call logic
    sources = [{
        "title": f"Mock Study {i+1} on {query}",
        "abstract": "This is a concise mock abstract summarizing key findings."
    } for i in range(max_results)]
    return json.dumps({"sources": sources})

@function_tool
async def search_clinical_trials_data(query: str, max_results: int = 15) -> str:
    """Search ClinicalTrials.gov API. Returns a JSON string of trials."""
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

# --- Specialized Agents ---
output_schema_settings = {"strict_json_schema": False}

literature_specialist = Agent(
    name="LiteratureSpecialist",
    instructions="You are a medical literature review expert. Analyze the provided abstracts to assess evidence quality and identify research gaps.",
    tools=[search_medical_literature],
    output_type=AgentOutputSchema(LiteratureAnalysis, **output_schema_settings)
)

competitive_analyst = Agent(
    name="CompetitiveAnalyst",
    instructions="You are a pharmaceutical competitive intelligence specialist. Analyze market landscapes and competitor strategies.",
    tools=[search_medical_literature, search_clinical_trials_data],
    output_type=AgentOutputSchema(CompetitiveIntelligence, **output_schema_settings)
)

clinical_trials_expert = Agent(
    name="ClinicalTrialsExpert",
    instructions="You are a clinical development expert. Analyze trial designs and timelines. For the 'phase_distribution' field, provide a dictionary where keys are phases and values are a LIST of trial titles or IDs.",
    tools=[search_clinical_trials_data],
    output_type=AgentOutputSchema(ClinicalTrialsAnalysis, **output_schema_settings)
)

regulatory_specialist = Agent(
    name="RegulatorySpecialist",
    instructions="You are a regulatory affairs specialist. Analyze approval pathways and regulatory risks based on available data.",
    tools=[search_medical_literature],
    output_type=AgentOutputSchema(RegulatoryAssessment, **output_schema_settings)
)

synthesis_agent = Agent(
    name="ResearchSynthesisAgent",
    instructions="You are a research synthesis expert. Integrate the summaries from multiple specialist analyses to create a single, comprehensive report.",
    tools=[], # This agent only synthesizes data
    output_type=AgentOutputSchema(ComprehensiveAnalysis, **output_schema_settings)
)

# --- Triage Agent & Guardrails ---
triage_agent = Agent(
    name="MedicalResearchTriageAgent",
    instructions="You are a triage specialist. Based on the user query, determine the optimal research strategy and handoff to the correct specialist agents.",
    handoffs=[literature_specialist, competitive_analyst, clinical_trials_expert, regulatory_specialist],
)

# --- Workflow Orchestrator ---
class MedicalResearchOrchestrator:
    async def execute_comprehensive_research(self, query: str, therapy_area: str) -> Dict:
        context = ResearchContext(user_id="system", research_request=query, therapy_area=therapy_area)
        try:
            await Runner.run(triage_agent, query, context=context)
            tasks = [
                Runner.run(literature_specialist, query, context=context),
                Runner.run(competitive_analyst, query, context=context),
                Runner.run(clinical_trials_expert, query, context=context),
                Runner.run(regulatory_specialist, query, context=context),
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            successful_results, agent_errors = [], []
            for res in results:
                if isinstance(res, Exception):
                    logger.error(f"Agent execution failed: {res}", exc_info=True)
                    agent_errors.append(str(res))
                else:
                    successful_results.append(res)
            
            if agent_errors:
                raise HTTPException(status_code=500, detail=f"One or more agents failed: {'; '.join(agent_errors)}")

            lit_res, comp_res, clin_res, reg_res = successful_results
            lit_summary, comp_summary, clin_summary, reg_summary = (
                lit_res.final_output_as(LiteratureAnalysis),
                comp_res.final_output_as(CompetitiveIntelligence),
                clin_res.final_output_as(ClinicalTrialsAnalysis),
                reg_res.final_output_as(RegulatoryAssessment),
            )
            
            synthesis_input = f"""
            Synthesize the following research summaries for the query: "{query}".
            - Literature Summary: {lit_summary.executive_summary}
            - Competitive Summary: {comp_summary.competitive_landscape}
            - Clinical Trials Summary: {clin_summary.development_landscape}
            - Regulatory Summary: {reg_summary.approval_pathways}
            Based on these summaries, create a single, comprehensive analysis.
            """
            
            synthesis_result = await Runner.run(synthesis_agent, synthesis_input, context=context)
            return synthesis_result.final_output_as(ComprehensiveAnalysis).model_dump()

        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logger.error(f"Comprehensive workflow failed unexpectedly: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# --- FastAPI Application ---
app = FastAPI(
    title="Medical Research Agent System",
    version="6.0.0",
    description="A multi-agent system for pharmaceutical research with specialized and comprehensive endpoints."
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
orchestrator = MedicalResearchOrchestrator()

# --- Endpoints ---

@app.get("/", summary="Root Endpoint")
async def root():
    return {"message": "Welcome to the Medical Research Agent System v6.0.0"}

@app.get("/health", summary="Health Check")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/research/literature", response_model=LiteratureAnalysis, summary="Run Literature Review Agent")
async def literature_review(request: LiteratureRequest):
    """Conducts a targeted medical literature review using the Literature Specialist agent."""
    context = ResearchContext(user_id="api", research_request=request.query, therapy_area="general")
    try:
        result = await Runner.run(literature_specialist, request.query, context=context)
        return result.final_output_as(LiteratureAnalysis)
    except Exception as e:
        logger.error(f"Literature review endpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research/competitive", response_model=CompetitiveIntelligence, summary="Run Competitive Intelligence Agent")
async def competitive_analysis(request: CompetitiveRequest):
    """Conducts a competitive intelligence analysis using the Competitive Analyst agent."""
    context = ResearchContext(user_id="api", research_request=request.query, therapy_area=request.therapy_area)
    try:
        result = await Runner.run(competitive_analyst, request.query, context=context)
        return result.final_output_as(CompetitiveIntelligence)
    except Exception as e:
        logger.error(f"Competitive analysis endpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research/comprehensive", response_model=Dict, summary="Run Comprehensive Multi-Agent Workflow")
async def comprehensive_research(request: ComprehensiveRequest):
    """Conducts a comprehensive, multi-agent research task and returns a synthesized analysis."""
    return await orchestrator.execute_comprehensive_research(query=request.query, therapy_area=request.therapy_area)

# --- Scheduled Monitoring Task ---

@app.on_event("startup")
@repeat_every(seconds=60 * 60 * 24) # Repeats once every 24 hours
async def automated_therapy_area_monitoring() -> None:
    """A scheduled background task to automatically monitor key therapy areas."""
    logger.info("Running automated daily monitoring for key therapy areas...")
    monitored_areas = ["oncology", "neurology", "rare_disease"]
    for area in monitored_areas:
        query = f"latest significant developments in {area}"
        logger.info(f"Running comprehensive analysis for: {query}")
        try:
            # You could store these results in Pinecone or another database
            await orchestrator.execute_comprehensive_research(query=query, therapy_area=area)
        except Exception as e:
            logger.error(f"Automated monitoring for '{area}' failed: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
