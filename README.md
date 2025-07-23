# Medical Research Agent

A sophisticated multi-agent system for pharmaceutical research, competitive intelligence, and regulatory analysis using OpenAI's Assistants API.

## Features

- **Sequential Multi-Agent Workflow**: Six specialized agents work in sequence to build comprehensive pharmaceutical intelligence
- **Vector Search Integration**: Semantic search capabilities with Pinecone vector database
- **Literature Analysis**: PubMed integration with GRADE evidence assessment
- **Clinical Trials Intelligence**: ClinicalTrials.gov integration with competitive landscape analysis
- **Regulatory Analysis**: FDA/EMA guidance analysis and pathway recommendations
- **Competitive Intelligence**: Market analysis and competitor tracking
- **Medical Writing**: Automated report generation and synthesis

## Architecture

### Agent Workflow
1. **Vector Search Agent** - Establishes baseline knowledge and identifies gaps
2. **Literature Analysis Agent** - Analyzes recent publications using vector insights
3. **Clinical Trials Agent** - Maps trial landscape using literature findings
4. **Competitive Intelligence Agent** - Analyzes market using clinical/literature data
5. **Regulatory Analysis Agent** - Assesses pathways using all gathered intelligence
6. **Medical Writing Agent** - Synthesizes everything into comprehensive report


