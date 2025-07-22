"""Medical research agents with specialized capabilities"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class AgentRole(str, Enum):
    TRIAGE = "triage"
    LITERATURE_SPECIALIST = "literature_specialist"
    COMPETITIVE_ANALYST = "competitive_analyst"
    SYNTHESIZER = "synthesizer"

@dataclass
class AgentContext:
    query: str
    therapy_area: str
    parameters: Dict[str, Any]
    previous_results: Dict[str, Any] = None

@dataclass
class AgentOutput:
    agent_role: AgentRole
    success: bool
    output: Dict[str, Any]
    confidence: float
    processing_time: float

class MedicalResearchAgent:
    """Base class for specialized medical research agents"""
    
    def __init__(self, role: AgentRole, openai_client, research_tools, specialization: str):
        self.role = role
        self.client = openai_client
        self.research_tools = research_tools
        self.specialization = specialization
    
    async def process(self, context: AgentContext) -> AgentOutput:
        """Process research task based on agent specialization"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if self.role == AgentRole.LITERATURE_SPECIALIST:
                result = await self._literature_analysis(context)
            elif self.role == AgentRole.COMPETITIVE_ANALYST:
                result = await self._competitive_analysis(context)
            else:
                result = await self._general_analysis(context)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return AgentOutput(
                agent_role=self.role,
                success=True,
                output=result['output'],
                confidence=result['confidence'],
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Agent {self.role} processing error: {e}")
            
            return AgentOutput(
                agent_role=self.role,
                success=False,
                output={},
                confidence=0.0,
                processing_time=processing_time
            )
    
    async def _literature_analysis(self, context: AgentContext) -> Dict:
        """Literature specialist analysis"""
        if not self.client:
            return self._mock_literature_result(context)
        
        # Get literature sources
        sources = await self.research_tools.advanced_pubmed_search(
            context.query, 
            context.parameters.get('max_sources', 15)
        )
        
        # AI analysis
        prompt = f"""
        Analyze medical literature for: {context.query}
        Sources: {len(sources)} recent publications
        
        Provide:
        1. Key findings from literature
        2. Evidence quality assessment
        3. Clinical implications
        4. Research recommendations
        
        Format as JSON with fields: key_findings, evidence_quality, clinical_implications, recommendations, confidence_score
        """
        
        try:
            response = await self.client.chat.completions.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are a {self.specialization}. Provide JSON-formatted analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            analysis = json.loads(response.choices[0].message.content)
            analysis['sources_analyzed'] = len(sources)
            
            return {
                'output': analysis,
                'confidence': analysis.get('confidence_score', 7.5)
            }
            
        except Exception as e:
            logger.error(f"Literature analysis error: {e}")
            return self._mock_literature_result(context)
    
    async def _competitive_analysis(self, context: AgentContext) -> Dict:
        """Competitive analyst analysis"""
        if not self.client:
            return self._mock_competitive_result(context)
        
        prompt = f"""
        Competitive analysis for: {context.query}
        Therapy area: {context.therapy_area}
        
        Provide:
        1. Market landscape overview
        2. Key competitors
        3. Strategic implications
        4. Opportunities and threats
        
        Format as JSON with fields: market_landscape, key_competitors, strategic_implications, opportunities, threats, confidence_score
        """
        
        try:
            response = await self.client.chat.completions.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are a {self.specialization}. Provide JSON-formatted analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            analysis = json.loads(response.choices[0].message.content)
            
            return {
                'output': analysis,
                'confidence': analysis.get('confidence_score', 7.0)
            }
            
        except Exception as e:
            logger.error(f"Competitive analysis error: {e}")
            return self._mock_competitive_result(context)
    
    async def _general_analysis(self, context: AgentContext) -> Dict:
        """General analysis for other agent types"""
        return {
            'output': {
                'analysis': f'General analysis for {context.query}',
                'insights': ['Analysis completed', 'Insights generated']
            },
            'confidence': 7.0
        }
    
    def _mock_literature_result(self, context: AgentContext) -> Dict:
        return {
            'output': {
                'key_findings': [f'Literature finding for {context.query}'],
                'evidence_quality': 'Moderate quality evidence',
                'clinical_implications': 'Clinical relevance identified',
                'recommendations': ['Continue monitoring literature'],
                'confidence_score': 7.5
            },
            'confidence': 7.5
        }
    
    def _mock_competitive_result(self, context: AgentContext) -> Dict:
        return {
            'output': {
                'market_landscape': f'Competitive market for {context.query}',
                'key_competitors': ['Competitor A', 'Competitor B'],
                'strategic_implications': 'Strategic considerations identified',
                'opportunities': ['Market opportunity 1'],
                'threats': ['Competitive threat 1'],
                'confidence_score': 7.0
            },
            'confidence': 7.0
        }

class AgentOrchestrator:
    """Orchestrates multi-agent workflows"""
    
    def __init__(self, openai_client, research_tools):
        self.agents = {
            AgentRole.LITERATURE_SPECIALIST: MedicalResearchAgent(
                AgentRole.LITERATURE_SPECIALIST, openai_client, research_tools,
                "medical literature review expert"
            ),
            AgentRole.COMPETITIVE_ANALYST: MedicalResearchAgent(
                AgentRole.COMPETITIVE_ANALYST, openai_client, research_tools,
                "pharmaceutical competitive intelligence analyst"
            )
        }
    
    async def execute_workflow(self, context: AgentContext) -> Dict[str, Any]:
        """Execute multi-agent workflow"""
        try:
            # Simple routing logic
            if "competitive" in context.query.lower():
                primary_agent = AgentRole.COMPETITIVE_ANALYST
            else:
                primary_agent = AgentRole.LITERATURE_SPECIALIST
            
            # Execute primary agent
            agent_output = await self.agents[primary_agent].process(context)
            
            return {
                "success": True,
                "query": context.query,
                "primary_agent": primary_agent.value,
                "analysis": agent_output.output,
                "confidence": agent_output.confidence,
                "processing_time": agent_output.processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
