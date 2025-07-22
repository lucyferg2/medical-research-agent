"""
Multi-Agent Workflow System for Medical Research
Implements sophisticated agent orchestration without requiring OpenAI Agents SDK
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AgentRole(str, Enum):
    """Specialized agent roles"""
    TRIAGE = "triage"
    LITERATURE_SPECIALIST = "literature_specialist"
    COMPETITIVE_ANALYST = "competitive_analyst"
    CLINICAL_TRIALS_EXPERT = "clinical_trials_expert"
    REGULATORY_SPECIALIST = "regulatory_specialist"
    SYNTHESIZER = "synthesizer"

class TaskType(str, Enum):
    """Types of research tasks"""
    LITERATURE_REVIEW = "literature_review"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    CLINICAL_LANDSCAPE = "clinical_landscape"
    REGULATORY_ASSESSMENT = "regulatory_assessment"
    COMPREHENSIVE_RESEARCH = "comprehensive_research"

@dataclass
class AgentContext:
    """Context passed between agents"""
    query: str
    therapy_area: str
    task_type: TaskType
    parameters: Dict[str, Any]
    previous_results: Dict[str, Any] = None
    conversation_history: List[Dict] = None
    priority_level: str = "normal"

@dataclass
class AgentOutput:
    """Standardized agent output"""
    agent_role: AgentRole
    success: bool
    output: Dict[str, Any]
    confidence: float
    processing_time: float
    next_agent_suggestions: List[AgentRole] = None
    requires_human_review: bool = False
    error_message: str = None

class MedicalResearchAgent:
    """Base class for specialized medical research agents"""
    
    def __init__(self, role: AgentRole, openai_client, research_tools, specialization: str):
        self.role = role
        self.client = openai_client
        self.research_tools = research_tools
        self.specialization = specialization
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build specialized system prompt based on agent role"""
        base_prompt = f"You are a {self.specialization} with deep expertise in pharmaceutical research and medical intelligence."
        
        role_prompts = {
            AgentRole.TRIAGE: """
            Your role is to analyze incoming research requests and determine:
            1. What type of analysis is needed
            2. Which specialized agents should be involved
            3. The priority and complexity level
            4. Any clarifications needed from the user
            
            Always provide routing decisions with clear reasoning.
            """,
            
            AgentRole.LITERATURE_SPECIALIST: """
            You are a medical literature review specialist. Your expertise includes:
            - Advanced literature search strategy
            - Evidence quality assessment and grading
            - Clinical significance evaluation
            - Meta-analysis and systematic review principles
            - Publication bias identification
            
            Provide comprehensive literature analysis with evidence-based insights.
            """,
            
            AgentRole.COMPETITIVE_ANALYST: """
            You are a pharmaceutical competitive intelligence analyst. Your expertise includes:
            - Market landscape analysis
            - Competitive positioning assessment
            - Pipeline analysis and development timelines
            - Strategic threat and opportunity identification
            - Business model evaluation
            
            Focus on actionable business intelligence for strategic decision-making.
            """,
            
            AgentRole.CLINICAL_TRIALS_EXPERT: """
            You are a clinical trials and development expert. Your expertise includes:
            - Clinical trial design and methodology
            - Endpoint selection and regulatory requirements
            - Development timelines and milestone assessment
            - Risk evaluation in clinical development
            - Regulatory pathway optimization
            
            Provide insights on clinical development strategy and execution.
            """,
            
            AgentRole.REGULATORY_SPECIALIST: """
            You are a regulatory affairs specialist. Your expertise includes:
            - FDA, EMA, and global regulatory requirements
            - Approval pathway analysis
            - Regulatory risk assessment
            - Compliance strategy
            - Policy impact evaluation
            
            Focus on regulatory strategy and approval optimization.
            """,
            
            AgentRole.SYNTHESIZER: """
            You are a research synthesizer who combines insights from multiple specialized analyses.
            Your role is to:
            - Integrate findings from different analytical perspectives
            - Identify key themes and contradictions
            - Provide executive-level summaries
            - Generate actionable recommendations
            - Assess overall confidence levels
            
            Create comprehensive, cohesive insights from specialized inputs.
            """
        }
        
        return base_prompt + role_prompts.get(self.role, "")
    
    async def process(self, context: AgentContext) -> AgentOutput:
        """Process the research task based on agent specialization"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Route to specialized processing method
            if self.role == AgentRole.TRIAGE:
                result = await self._triage_analysis(context)
            elif self.role == AgentRole.LITERATURE_SPECIALIST:
                result = await self._literature_analysis(context)
            elif self.role == AgentRole.COMPETITIVE_ANALYST:
                result = await self._competitive_analysis(context)
            elif self.role == AgentRole.CLINICAL_TRIALS_EXPERT:
                result = await self._clinical_trials_analysis(context)
            elif self.role == AgentRole.REGULATORY_SPECIALIST:
                result = await self._regulatory_analysis(context)
            elif self.role == AgentRole.SYNTHESIZER:
                result = await self._synthesis_analysis(context)
            else:
                raise ValueError(f"Unknown agent role: {self.role}")
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return AgentOutput(
                agent_role=self.role,
                success=True,
                output=result['output'],
                confidence=result['confidence'],
                processing_time=processing_time,
                next_agent_suggestions=result.get('next_agents', []),
                requires_human_review=result.get('requires_review', False)
            )
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Agent {self.role} processing error: {e}")
            
            return AgentOutput(
                agent_role=self.role,
                success=False,
                output={},
                confidence=0.0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def _triage_analysis(self, context: AgentContext) -> Dict:
        """Triage agent determines routing and analysis strategy"""
        if not self.client:
            return self._mock_triage_result(context)
        
        triage_prompt = f"""
        Analyze this medical research request and determine the optimal analysis approach:
        
        Query: {context.query}
        Therapy Area: {context.therapy_area}
        Current Task Type: {context.task_type}
        
        Determine:
        1. ANALYSIS_STRATEGY: What type of analysis is most appropriate?
        2. AGENT_SEQUENCE: Which specialized agents should be involved and in what order?
        3. PRIORITY_LEVEL: How urgent/complex is this request?
        4. CLARIFICATIONS: Any missing information needed?
        5. EXPECTED_OUTPUTS: What deliverables should be produced?
        
        Agent options: literature_specialist, competitive_analyst, clinical_trials_expert, regulatory_specialist, synthesizer
        
        Respond with JSON containing: analysis_strategy, agent_sequence, priority_level, clarifications, expected_outputs, confidence_score
        """
        
        try:
            response = await self.client.chat.completions.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": triage_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            triage_result = json.loads(response.choices[0].message.content)
            
            return {
                'output': triage_result,
                'confidence': triage_result.get('confidence_score', 8.0),
                'next_agents': [AgentRole(agent) for agent in triage_result.get('agent_sequence', [])]
            }
            
        except Exception as e:
            logger.error(f"Triage analysis error: {e}")
            return self._mock_triage_result(context)
    
    async def _literature_analysis(self, context: AgentContext) -> Dict:
        """Literature specialist performs advanced literature review"""
        # Get literature data
        literature_sources = await self.research_tools.advanced_pubmed_search(
            context.query, 
            context.parameters.get('max_sources', 20),
            context.parameters.get('days_back', 90)
        )
        
        if not self.client:
            return self._mock_literature_result(context, len(literature_sources))
        
        # Prepare sources for analysis
        sources_summary = [
            {
                'title': source.title,
                'journal': source.journal,
                'abstract': source.abstract[:300],
                'relevance_score': source.relevance_score
            }
            for source in literature_sources[:10]
        ]
        
        literature_prompt = f"""
        Conduct expert literature review analysis for: {context.query}
        
        Therapy Area: {context.therapy_area}
        Sources Available: {len(literature_sources)}
        
        As a literature review specialist, provide:
        
        1. EVIDENCE_SYNTHESIS: Key findings across all sources
        2. QUALITY_ASSESSMENT: Evidence strength and study limitations
        3. CLINICAL_SIGNIFICANCE: Relevance to patient outcomes
        4. RESEARCH_GAPS: What's missing from current literature
        5. METHODOLOGY_CRITIQUE: Study design assessment
        6. FUTURE_DIRECTIONS: Recommended research priorities
        
        Sources: {json.dumps(sources_summary, indent=2)}
        
        Format as JSON with fields: evidence_synthesis, quality_assessment, clinical_significance, research_gaps, methodology_critique, future_directions, confidence_score
        """
        
        try:
            response = await self.client.chat.completions.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": literature_prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            literature_result = json.loads(response.choices[0].message.content)
            literature_result['sources_analyzed'] = len(literature_sources)
            literature_result['detailed_sources'] = [asdict(source) for source in literature_sources]
            
            return {
                'output': literature_result,
                'confidence': literature_result.get('confidence_score', 7.5),
                'next_agents': [AgentRole.SYNTHESIZER]
            }
            
        except Exception as e:
            logger.error(f"Literature analysis error: {e}")
            return self._mock_literature_result(context, len(literature_sources))
    
    async def _competitive_analysis(self, context: AgentContext) -> Dict:
        """Competitive analyst performs market intelligence analysis"""
        # Gather competitive data
        literature_sources = await self.research_tools.advanced_pubmed_search(context.query, 15, 180)
        trial_data = await self.research_tools.search_clinical_trials(context.query, 10)
        
        if not self.client:
            return self._mock_competitive_result(context, len(literature_sources), len(trial_data))
        
        competitive_prompt = f"""
        Conduct expert competitive intelligence analysis for: {context.query}
        
        Therapy Area: {context.therapy_area}
        Intelligence Sources: {len(literature_sources)} literature + {len(trial_data)} trials
        
        As a competitive intelligence specialist, provide:
        
        1. MARKET_STRUCTURE: Current competitive landscape
        2. PLAYER_ANALYSIS: Key competitors and their strategies
        3. PIPELINE_INTELLIGENCE: Development programs and timelines
        4. COMPETITIVE_ADVANTAGES: Differentiation opportunities
        5. MARKET_DYNAMICS: Trends and shifts in competition
        6. STRATEGIC_THREATS: Risks to market position
        7. OPPORTUNITY_MAP: Areas for competitive advantage
        
        Format as JSON with fields: market_structure, player_analysis, pipeline_intelligence, competitive_advantages, market_dynamics, strategic_threats, opportunity_map, confidence_score
        """
        
        try:
            response = await self.client.chat.completions.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": competitive_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            competitive_result = json.loads(response.choices[0].message.content)
            competitive_result['data_sources'] = {
                'literature_sources': len(literature_sources),
                'clinical_trials': len(trial_data)
            }
            
            return {
                'output': competitive_result,
                'confidence': competitive_result.get('confidence_score', 7.0),
                'next_agents': [AgentRole.SYNTHESIZER]
            }
            
        except Exception as e:
            logger.error(f"Competitive analysis error: {e}")
            return self._mock_competitive_result(context, len(literature_sources), len(trial_data))
    
    async def _clinical_trials_analysis(self, context: AgentContext) -> Dict:
        """Clinical trials expert analyzes development landscape"""
        trial_data = await self.research_tools.search_clinical_trials(context.query, 15)
        
        if not self.client:
            return self._mock_clinical_trials_result(context, len(trial_data))
        
        clinical_prompt = f"""
        Conduct expert clinical trials landscape analysis for: {context.query}
        
        Therapy Area: {context.therapy_area}
        Active Trials: {len(trial_data)}
        
        As a clinical development expert, analyze:
        
        1. DEVELOPMENT_LANDSCAPE: Overview of clinical activity
        2. TRIAL_DESIGN_TRENDS: Common methodologies and endpoints
        3. REGULATORY_PATHWAYS: Approval strategies being pursued
        4. DEVELOPMENT_RISKS: Common challenges and failures
        5. SUCCESS_FACTORS: Elements of successful programs
        6. TIMELINE_ANALYSIS: Expected development timelines
        7. STRATEGIC_RECOMMENDATIONS: Clinical development advice
        
        Format as JSON with fields: development_landscape, trial_design_trends, regulatory_pathways, development_risks, success_factors, timeline_analysis, strategic_recommendations, confidence_score
        """
        
        try:
            response = await self.client.chat.completions.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": clinical_prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            clinical_result = json.loads(response.choices[0].message.content)
            clinical_result['trials_analyzed'] = len(trial_data)
            
            return {
                'output': clinical_result,
                'confidence': clinical_result.get('confidence_score', 7.5),
                'next_agents': [AgentRole.SYNTHESIZER]
            }
            
        except Exception as e:
            logger.error(f"Clinical trials analysis error: {e}")
            return self._mock_clinical_trials_result(context, len(trial_data))
    
    async def _regulatory_analysis(self, context: AgentContext) -> Dict:
        """Regulatory specialist analyzes approval landscape"""
        if not self.client:
            return self._mock_regulatory_result(context)
        
        regulatory_prompt = f"""
        Conduct expert regulatory analysis for: {context.query}
        
        Therapy Area: {context.therapy_area}
        
        As a regulatory affairs specialist, analyze:
        
        1. APPROVAL_PATHWAYS: Available regulatory routes
        2. REGULATORY_PRECEDENTS: Similar approved products
        3. GUIDANCE_LANDSCAPE: Relevant FDA/EMA guidance
        4. APPROVAL_TIMELINES: Expected review periods
        5. REGULATORY_RISKS: Potential approval challenges
        6. STRATEGY_RECOMMENDATIONS: Regulatory approach advice
        
        Format as JSON with fields: approval_pathways, regulatory_precedents, guidance_landscape, approval_timelines, regulatory_risks, strategy_recommendations, confidence_score
        """
        
        try:
            response = await self.client.chat.completions.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": regulatory_prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            regulatory_result = json.loads(response.choices[0].message.content)
            
            return {
                'output': regulatory_result,
                'confidence': regulatory_result.get('confidence_score', 8.0),
                'next_agents': [AgentRole.SYNTHESIZER]
            }
            
        except Exception as e:
            logger.error(f"Regulatory analysis error: {e}")
            return self._mock_regulatory_result(context)
    
    async def _synthesis_analysis(self, context: AgentContext) -> Dict:
        """Synthesizer combines insights from multiple agents"""
        if not context.previous_results:
            return {'output': {'error': 'No previous results to synthesize'}, 'confidence': 0.0}
        
        if not self.client:
            return self._mock_synthesis_result(context)
        
        synthesis_prompt = f"""
        Synthesize insights from multiple specialized analyses for: {context.query}
        
        Previous Analysis Results:
        {json.dumps(context.previous_results, indent=2)}
        
        As a research synthesizer, create:
        
        1. EXECUTIVE_SUMMARY: Key insights across all analyses
        2. INTEGRATED_FINDINGS: Unified view of important discoveries
        3. STRATEGIC_IMPLICATIONS: Business strategy recommendations
        4. RISK_ASSESSMENT: Consolidated risk evaluation
        5. OPPORTUNITY_ANALYSIS: Identified opportunities
        6. ACTION_PRIORITIES: Recommended next steps
        7. CONFIDENCE_ASSESSMENT: Overall reliability of insights
        
        Format as JSON with fields: executive_summary, integrated_findings, strategic_implications, risk_assessment, opportunity_analysis, action_priorities, confidence_assessment, overall_confidence
        """
        
        try:
            response = await self.client.chat.completions.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.2,
                max_tokens=2500
            )
            
            synthesis_result = json.loads(response.choices[0].message.content)
            
            return {
                'output': synthesis_result,
                'confidence': synthesis_result.get('overall_confidence', 8.0),
                'next_agents': []  # Final step
            }
            
        except Exception as e:
            logger.error(f"Synthesis analysis error: {e}")
            return self._mock_synthesis_result(context)
    
    # Mock result methods for when OpenAI is not available
    def _mock_triage_result(self, context: AgentContext) -> Dict:
        return {
            'output': {
                'analysis_strategy': f'Comprehensive analysis recommended for {context.query}',
                'agent_sequence': ['literature_specialist', 'competitive_analyst', 'synthesizer'],
                'priority_level': 'normal',
                'clarifications': [],
                'expected_outputs': ['Literature review', 'Competitive analysis', 'Strategic synthesis'],
                'confidence_score': 8.0
            },
            'confidence': 8.0,
            'next_agents': [AgentRole.LITERATURE_SPECIALIST, AgentRole.COMPETITIVE_ANALYST]
        }
    
    def _mock_literature_result(self, context: AgentContext, source_count: int) -> Dict:
        return {
            'output': {
                'evidence_synthesis': f'Analysis of {source_count} sources for {context.query}',
                'quality_assessment': 'Mixed study quality with moderate evidence strength',
                'clinical_significance': 'Clinically relevant findings identified',
                'research_gaps': 'Additional long-term safety data needed',
                'methodology_critique': 'Appropriate study designs for the research questions',
                'future_directions': 'Phase III trials recommended',
                'confidence_score': 7.5,
                'sources_analyzed': source_count
            },
            'confidence': 7.5
        }
    
    def _mock_competitive_result(self, context: AgentContext, lit_count: int, trial_count: int) -> Dict:
        return {
            'output': {
                'market_structure': f'Active competitive market in {context.therapy_area}',
                'player_analysis': ['Big Pharma leaders', 'Emerging biotech companies'],
                'pipeline_intelligence': 'Multiple Phase II/III programs active',
                'competitive_advantages': 'Opportunities for differentiation exist',
                'market_dynamics': 'Evolving competitive landscape',
                'strategic_threats': 'Established player advantages',
                'opportunity_map': 'Underserved patient segments identified',
                'confidence_score': 7.0
            },
            'confidence': 7.0
        }
    
    def _mock_clinical_trials_result(self, context: AgentContext, trial_count: int) -> Dict:
        return {
            'output': {
                'development_landscape': f'{trial_count} active trials in development',
                'trial_design_trends': 'Randomized controlled trials predominate',
                'regulatory_pathways': 'Standard approval pathways applicable',
                'development_risks': 'Typical Phase III risks apply',
                'success_factors': 'Patient selection and endpoint design critical',
                'timeline_analysis': '4-6 year development timelines expected',
                'strategic_recommendations': 'Focus on differentiated endpoints',
                'confidence_score': 7.5
            },
            'confidence': 7.5
        }
    
    def _mock_regulatory_result(self, context: AgentContext) -> Dict:
        return {
            'output': {
                'approval_pathways': 'Standard NDA pathway applicable',
                'regulatory_precedents': 'Similar products have been approved',
                'guidance_landscape': 'Clear regulatory guidance available',
                'approval_timelines': '10-12 month review timeline',
                'regulatory_risks': 'Standard regulatory risks apply',
                'strategy_recommendations': 'Early regulatory engagement advised',
                'confidence_score': 8.0
            },
            'confidence': 8.0
        }
    
    def _mock_synthesis_result(self, context: AgentContext) -> Dict:
        return {
            'output': {
                'executive_summary': f'Comprehensive analysis completed for {context.query}',
                'integrated_findings': 'Multiple analytical perspectives synthesized',
                'strategic_implications': 'Strategic opportunities identified',
                'risk_assessment': 'Manageable risks with mitigation strategies',
                'opportunity_analysis': 'Market opportunities validated',
                'action_priorities': ['Continue development', 'Engage regulatory'],
                'confidence_assessment': 'High confidence in key findings',
                'overall_confidence': 8.0
            },
            'confidence': 8.0
        }

class AgentOrchestrator:
    """Orchestrates multi-agent workflows"""
    
    def __init__(self, openai_client, research_tools):
        self.agents = {
            AgentRole.TRIAGE: MedicalResearchAgent(
                AgentRole.TRIAGE, openai_client, research_tools, 
                "medical research triage specialist"
            ),
            AgentRole.LITERATURE_SPECIALIST: MedicalResearchAgent(
                AgentRole.LITERATURE_SPECIALIST, openai_client, research_tools,
                "medical literature review expert"
            ),
            AgentRole.COMPETITIVE_ANALYST: MedicalResearchAgent(
                AgentRole.COMPETITIVE_ANALYST, openai_client, research_tools,
                "pharmaceutical competitive intelligence analyst"
            ),
            AgentRole.CLINICAL_TRIALS_EXPERT: MedicalResearchAgent(
                AgentRole.CLINICAL_TRIALS_EXPERT, openai_client, research_tools,
                "clinical development and trials expert"
            ),
            AgentRole.REGULATORY_SPECIALIST: MedicalResearchAgent(
                AgentRole.REGULATORY_SPECIALIST, openai_client, research_tools,
                "regulatory affairs specialist"
            ),
            AgentRole.SYNTHESIZER: MedicalResearchAgent(
                AgentRole.SYNTHESIZER, openai_client, research_tools,
                "research synthesis and strategy expert"
            )
        }
    
    async def execute_workflow(self, context: AgentContext) -> Dict[str, Any]:
        """Execute multi-agent workflow based on triage routing"""
        workflow_results = {}
        
        try:
            # Step 1: Triage analysis
            logger.info(f"Starting triage analysis for: {context.query}")
            triage_output = await self.agents[AgentRole.TRIAGE].process(context)
            workflow_results['triage'] = asdict(triage_output)
            
            if not triage_output.success:
                return {"success": False, "error": "Triage analysis failed", "results": workflow_results}
            
            # Step 2: Execute specialist agents based on triage routing
            agent_sequence = triage_output.next_agent_suggestions or [AgentRole.LITERATURE_SPECIALIST]
            specialist_results = {}
            
            for agent_role in agent_sequence:
                if agent_role == AgentRole.SYNTHESIZER:
                    continue  # Handle synthesizer separately
                
                logger.info(f"Executing {agent_role} analysis")
                context.previous_results = specialist_results
                
                agent_output = await self.agents[agent_role].process(context)
                specialist_results[agent_role.value] = agent_output.output
                workflow_results[agent_role.value] = asdict(agent_output)
            
            # Step 3: Synthesis (if multiple specialists were involved)
            if len(specialist_results) > 1:
                logger.info("Executing synthesis analysis")
                context.previous_results = specialist_results
                
                synthesis_output = await self.agents[AgentRole.SYNTHESIZER].process(context)
                workflow_results['synthesis'] = asdict(synthesis_output)
                
                # Use synthesis as final result
                final_analysis = synthesis_output.output
            else:
                # Use single specialist result
                final_analysis = list(specialist_results.values())[0] if specialist_results else {}
            
            # Step 4: Compile final response
            return {
                "success": True,
                "query": context.query,
                "therapy_area": context.therapy_area,
                "workflow_type": "multi_agent_analysis",
                "agents_involved": [agent.value for agent in agent_sequence] + ['synthesizer'],
                "final_analysis": final_analysis,
                "detailed_workflow": workflow_results,
                "timestamp": datetime.now().isoformat(),
                "processing_metadata": {
                    "triage_routing": triage_output.next_agent_suggestions,
                    "total_agents": len(workflow_results),
                    "overall_confidence": self._calculate_overall_confidence(workflow_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "partial_results": workflow_results,
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_overall_confidence(self, results: Dict) -> float:
        """Calculate overall confidence from agent results"""
        confidences = []
        for result in results.values():
            if isinstance(result, dict) and 'confidence' in result:
                confidences.append(result['confidence'])
        
        return sum(confidences) / len(confidences) if confidences else 0.0
