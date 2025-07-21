from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime

class TherapyArea(str, Enum):
    ONCOLOGY = "oncology"
    NEUROLOGY = "neurology"
    CARDIOLOGY = "cardiology"
    ENDOCRINOLOGY = "endocrinology"
    IMMUNOLOGY = "immunology"
    INFECTIOUS_DISEASE = "infectious_disease"
    RARE_DISEASE = "rare_disease"
    GENERAL = "general"

class ResearchType(str, Enum):
    LITERATURE_REVIEW = "literature_review"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    REGULATORY_LANDSCAPE = "regulatory_landscape"
    CLINICAL_TRIALS = "clinical_trials"
    DRUG_SAFETY = "drug_safety"
    MARKET_ANALYSIS = "market_analysis"

# API Request Models
class ResearchRequest(BaseModel):
    query: str = Field(..., description="Research query or question")
    therapy_area: Optional[TherapyArea] = Field(TherapyArea.GENERAL, description="Therapy area focus")
    research_type: Optional[ResearchType] = Field(ResearchType.LITERATURE_REVIEW, description="Type of research to conduct")
    max_sources: Optional[int] = Field(20, description="Maximum number of sources to analyze")
    days_back: Optional[int] = Field(90, description="Number of days to look back for recent publications")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "CAR-T cell therapy safety in multiple myeloma",
                "therapy_area": "oncology",
                "research_type": "literature_review",
                "max_sources": 20,
                "days_back": 90
            }
        }

class LiteratureRequest(BaseModel):
    query: str = Field(..., description="Literature search query")
    therapy_area: str = Field("general", description="Therapy area focus")
    days_back: int = Field(90, description="Days to look back for recent publications")
    max_results: int = Field(20, description="Maximum number of results")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "immune checkpoint inhibitors melanoma",
                "therapy_area": "oncology",
                "days_back": 90,
                "max_results": 20
            }
        }

class CompetitiveRequest(BaseModel):
    competitor_query: str = Field(..., description="Competitive analysis query")
    therapy_area: str = Field(..., description="Therapy area for competitive analysis")
    include_trials: bool = Field(True, description="Include clinical trials in analysis")
    
    class Config:
        schema_extra = {
            "example": {
                "competitor_query": "pembrolizumab biosimilar development",
                "therapy_area": "oncology",
                "include_trials": True
            }
        }

# Response Models
class AnalysisOutput(BaseModel):
    executive_summary: str
    key_findings: List[str]
    clinical_implications: str
    recommendations: List[str]
    confidence_score: Optional[float] = None

class ResearchResponse(BaseModel):
    success: bool
    research_id: str
    query: str
    research_type: str
    therapy_area: str
    sources_analyzed: int
    analysis: AnalysisOutput
    timestamp: str
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "research_id": "12345-abcde",
                "query": "CAR-T cell therapy safety",
                "research_type": "literature_review",
                "therapy_area": "oncology",
                "sources_analyzed": 15,
                "analysis": {
                    "executive_summary": "Recent literature shows promising safety profile...",
                    "key_findings": ["Finding 1", "Finding 2"],
                    "clinical_implications": "Clinical implications...",
                    "recommendations": ["Recommendation 1", "Recommendation 2"],
                    "confidence_score": 8.5
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }

class CompetitiveAnalysisOutput(BaseModel):
    competitive_landscape: str
    key_competitors: List[str]
    market_positioning: str
    development_pipeline: List[str]
    strategic_implications: str
    risk_assessment: str

class CompetitiveResponse(BaseModel):
    success: bool
    research_id: str
    query: str
    therapy_area: str
    literature_sources: int
    trial_sources: int
    analysis: CompetitiveAnalysisOutput
    timestamp: str

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    service: str
    version: str
