"""
Medical Research Tools for integrating with external data sources
Handles PubMed, ClinicalTrials.gov, and other medical databases
"""

import aiohttp
import asyncio
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)

class MedicalResearchTools:
    """
    Tools for accessing medical research databases and APIs
    """
    
    def __init__(self, email: str):
        self.email = email  # Required for PubMed API courtesy
        self.pubmed_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.clinicaltrials_base_url = "https://clinicaltrials.gov/api/v2"
        
        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 0.5  # 500ms between requests
    
    async def _rate_limit(self, service: str):
        """Implement rate limiting for API requests"""
        current_time = asyncio.get_event_loop().time()
        if service in self.last_request_time:
            time_since_last = current_time - self.last_request_time[service]
            if time_since_last < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time[service] = asyncio.get_event_loop().time()
    
    async def search_pubmed(self, query: str, max_results: int = 20, 
                           days_back: int = 90) -> List[Dict]:
        """
        Search PubMed for recent medical literature
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            days_back: Number of days to look back for recent publications
            
        Returns:
            List of article dictionaries with metadata and abstracts
        """
        try:
            await self._rate_limit("pubmed")
            
            # Calculate date range for recent articles
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            date_range = f"{start_date.strftime('%Y/%m/%d')}:{end_date.strftime('%Y/%m/%d')}"
            
            # Step 1: Search for article IDs
            search_url = f"{self.pubmed_base_url}/esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': f"{query} AND {date_range}[pdat]",
                'retmax': max_results,
                'retmode': 'xml',
                'tool': 'medical_research_agent',
                'email': self.email
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=search_params) as response:
                    if response.status != 200:
                        logger.error(f"PubMed search failed with status {response.status}")
                        return []
                    
                    xml_content = await response.text()
            
            # Parse search results to get PMIDs
            root = ET.fromstring(xml_content)
            pmids = [id_elem.text for id_elem in root.findall('.//Id')]
            
            if not pmids:
                logger.info(f"No PubMed articles found for query: {query}")
                return []
            
            logger.info(f"Found {len(pmids)} PubMed articles for query: {query}")
            
            # Step 2: Fetch detailed article information
            return await self.fetch_pubmed_details(pmids)
            
        except Exception as e:
            logger.error(f"Error fetching PubMed details: {e}")
            return []
    
    def _parse_pubmed_article(self, article) -> Optional[Dict]:
        """
        Parse individual PubMed article XML
        
        Args:
            article: XML element representing a PubMed article
            
        Returns:
            Dictionary with article metadata and content
        """
        try:
            # Extract PMID
            pmid_elem = article.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else "Unknown"
            
            # Extract title
            title_elem = article.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else "Title not available"
            
            # Extract authors
            authors = []
            author_list = article.find('.//AuthorList')
            if author_list is not None:
                for author in author_list.findall('.//Author'):
                    lastname_elem = author.find('.//LastName')
                    forename_elem = author.find('.//ForeName')
                    
                    if lastname_elem is not None and forename_elem is not None:
                        authors.append(f"{forename_elem.text} {lastname_elem.text}")
                    elif lastname_elem is not None:
                        authors.append(lastname_elem.text)
            
            # Extract journal information
            journal_elem = article.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else "Journal not available"
            
            # Extract publication date
            pub_date = self._extract_publication_date(article)
            
            # Extract abstract
            abstract = self._extract_abstract(article)
            
            # Extract keywords
            keywords = self._extract_keywords(article)
            
            # Extract DOI
            doi = self._extract_doi(article)
            
            return {
                'pmid': pmid,
                'title': title,
                'authors': authors,
                'journal': journal,
                'publication_date': pub_date,
                'abstract': abstract,
                'keywords': keywords,
                'doi': doi,
                'source_type': 'pubmed',
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            }
            
        except Exception as e:
            logger.warning(f"Error parsing PubMed article: {e}")
            return None
    
    def _extract_publication_date(self, article) -> str:
        """Extract publication date from PubMed article"""
        try:
            # Try electronic publication date first
            epub_date = article.find('.//ArticleDate[@DateType="Electronic"]')
            if epub_date is not None:
                year = epub_date.find('.//Year')
                month = epub_date.find('.//Month')
                day = epub_date.find('.//Day')
                
                if year is not None:
                    date_str = year.text
                    if month is not None:
                        date_str += f"-{month.text.zfill(2)}"
                        if day is not None:
                            date_str += f"-{day.text.zfill(2)}"
                    return date_str
            
            # Fallback to journal publication date
            pub_date = article.find('.//PubDate')
            if pub_date is not None:
                year = pub_date.find('.//Year')
                month = pub_date.find('.//Month')
                
                if year is not None:
                    date_str = year.text
                    if month is not None:
                        # Convert month name to number if needed
                        month_text = month.text
                        month_map = {
                            'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                            'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                            'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
                        }
                        if month_text in month_map:
                            date_str += f"-{month_map[month_text]}"
                        elif month_text.isdigit():
                            date_str += f"-{month_text.zfill(2)}"
                    return date_str
            
            return "Date not available"
            
        except Exception:
            return "Date not available"
    
    def _extract_abstract(self, article) -> str:
        """Extract abstract from PubMed article"""
        try:
            # Handle structured abstracts
            abstract_parts = []
            abstract_elem = article.find('.//Abstract')
            
            if abstract_elem is not None:
                abstract_texts = abstract_elem.findall('.//AbstractText')
                
                for abstract_text in abstract_texts:
                    label = abstract_text.get('Label', '')
                    text = abstract_text.text or ''
                    
                    if label and text:
                        abstract_parts.append(f"{label}: {text}")
                    elif text:
                        abstract_parts.append(text)
                
                if abstract_parts:
                    return ' '.join(abstract_parts)
            
            # Fallback to simple abstract text
            simple_abstract = article.find('.//AbstractText')
            if simple_abstract is not None and simple_abstract.text:
                return simple_abstract.text
            
            return "Abstract not available"
            
        except Exception:
            return "Abstract not available"
    
    def _extract_keywords(self, article) -> List[str]:
        """Extract keywords from PubMed article"""
        try:
            keywords = []
            keyword_list = article.find('.//KeywordList')
            
            if keyword_list is not None:
                for keyword in keyword_list.findall('.//Keyword'):
                    if keyword.text:
                        keywords.append(keyword.text)
            
            return keywords
            
        except Exception:
            return []
    
    def _extract_doi(self, article) -> Optional[str]:
        """Extract DOI from PubMed article"""
        try:
            article_ids = article.findall('.//ArticleId')
            for article_id in article_ids:
                if article_id.get('IdType') == 'doi':
                    return article_id.text
            return None
        except Exception:
            return None
    
    async def search_clinical_trials(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search ClinicalTrials.gov for relevant trials
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of clinical trial information
        """
        try:
            await self._rate_limit("clinicaltrials")
            
            search_url = f"{self.clinicaltrials_base_url}/studies"
            params = {
                'query.cond': query,
                'countTotal': 'true',
                'pageSize': max_results,
                'format': 'json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"ClinicalTrials.gov API returned status {response.status}")
                        return []
                    
                    data = await response.json()
                    studies = data.get('studies', [])
                    
                    formatted_trials = []
                    for study in studies:
                        trial_data = self._parse_clinical_trial(study)
                        if trial_data:
                            formatted_trials.append(trial_data)
                    
                    logger.info(f"Found {len(formatted_trials)} clinical trials for query: {query}")
                    return formatted_trials
                    
        except Exception as e:
            logger.error(f"Error searching clinical trials: {e}")
            return []
    
    def _parse_clinical_trial(self, study: Dict) -> Optional[Dict]:
        """
        Parse clinical trial data from ClinicalTrials.gov API
        
        Args:
            study: Raw study data from API
            
        Returns:
            Formatted trial information
        """
        try:
            protocol = study.get('protocolSection', {})
            
            # Extract basic identification
            identification = protocol.get('identificationModule', {})
            nct_id = identification.get('nctId', 'N/A')
            title = identification.get('briefTitle', 'Title not available')
            
            # Extract status information
            status_module = protocol.get('statusModule', {})
            overall_status = status_module.get('overallStatus', 'N/A')
            
            # Extract design information
            design_module = protocol.get('designModule', {})
            phases = design_module.get('phases', [])
            phase = phases[0] if phases else 'N/A'
            study_type = design_module.get('studyType', 'N/A')
            
            # Extract conditions
            conditions_module = protocol.get('conditionsModule', {})
            conditions = conditions_module.get('conditions', [])
            primary_condition = conditions[0] if conditions else 'N/A'
            
            # Extract sponsor information
            sponsor_module = protocol.get('sponsorCollaboratorsModule', {})
            lead_sponsor = sponsor_module.get('leadSponsor', {})
            sponsor_name = lead_sponsor.get('name', 'N/A')
            
            # Extract enrollment info
            enrollment_info = status_module.get('enrollmentInfo', {})
            enrollment_count = enrollment_info.get('count')
            
            # Extract start date
            start_date_struct = status_module.get('startDateStruct', {})
            start_date = start_date_struct.get('date', 'N/A')
            
            # Extract primary endpoints
            outcomes_module = protocol.get('outcomesModule', {})
            primary_outcomes = outcomes_module.get('primaryOutcomes', [])
            primary_endpoint = primary_outcomes[0].get('measure', 'Available in full record') if primary_outcomes else 'Available in full record'
            
            # Extract eligibility criteria
            eligibility_module = protocol.get('eligibilityModule', {})
            min_age = eligibility_module.get('minimumAge', 'N/A')
            max_age = eligibility_module.get('maximumAge', 'N/A')
            gender = eligibility_module.get('sex', 'N/A')
            
            return {
                'nct_id': nct_id,
                'title': title,
                'status': overall_status,
                'phase': phase,
                'study_type': study_type,
                'condition': primary_condition,
                'all_conditions': conditions,
                'sponsor': sponsor_name,
                'enrollment': enrollment_count,
                'start_date': start_date,
                'primary_endpoint': primary_endpoint,
                'eligibility': {
                    'min_age': min_age,
                    'max_age': max_age,
                    'gender': gender
                },
                'source_type': 'clinical_trial',
                'url': f"https://clinicaltrials.gov/study/{nct_id}"
            }
            
        except Exception as e:
            logger.warning(f"Error parsing clinical trial: {e}")
            return None
    
    async def search_fda_approvals(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search FDA Orange Book and other FDA databases
        Note: This is a placeholder for FDA API integration
        """
        try:
            # This would integrate with FDA APIs when available
            # For now, return placeholder data
            logger.info(f"FDA search requested for: {query} (placeholder implementation)")
            
            return [
                {
                    'drug_name': 'Example Drug',
                    'active_ingredient': 'Example Compound',
                    'approval_date': '2023-01-01',
                    'indication': query,
                    'sponsor': 'Example Pharma',
                    'approval_type': 'NDA',
                    'source_type': 'fda_approval',
                    'url': 'https://www.fda.gov/drugs/drug-approvals-and-databases'
                }
            ]
            
        except Exception as e:
            logger.error(f"Error searching FDA data: {e}")
            return []
    
    async def get_drug_interactions(self, drug_name: str) -> List[Dict]:
        """
        Get drug interaction information
        Note: This would integrate with drug interaction databases
        """
        try:
            # Placeholder for drug interaction API integration
            logger.info(f"Drug interaction search for: {drug_name} (placeholder implementation)")
            
            return [
                {
                    'drug_a': drug_name,
                    'drug_b': 'Example Interacting Drug',
                    'interaction_type': 'Major',
                    'mechanism': 'CYP450 inhibition',
                    'clinical_significance': 'Monitor closely',
                    'source_type': 'drug_interaction'
                }
            ]
            
        except Exception as e:
            logger.error(f"Error getting drug interactions: {e}")
            return []
    
    async def batch_search(self, queries: List[str], search_type: str = "pubmed") -> Dict[str, List[Dict]]:
        """
        Perform batch searches for multiple queries
        
        Args:
            queries: List of search queries
            search_type: Type of search ("pubmed", "clinical_trials", etc.)
            
        Returns:
            Dictionary mapping queries to their results
        """
        results = {}
        
        for query in queries:
            try:
                if search_type == "pubmed":
                    results[query] = await self.search_pubmed(query)
                elif search_type == "clinical_trials":
                    results[query] = await self.search_clinical_trials(query)
                else:
                    logger.warning(f"Unknown search type: {search_type}")
                    results[query] = []
                
                # Rate limiting between batch requests
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in batch search for '{query}': {e}")
                results[query] = []
        
        return results
    
    async def get_source_metrics(self) -> Dict:
        """
        Get metrics about data source availability and performance
        """
        return {
            'pubmed': {
                'available': True,
                'base_url': self.pubmed_base_url,
                'rate_limit': f"{1/self.min_request_interval} requests/second"
            },
            'clinical_trials': {
                'available': True,
                'base_url': self.clinicaltrials_base_url,
                'rate_limit': f"{1/self.min_request_interval} requests/second"
            },
            'fda': {
                'available': False,
                'note': 'Placeholder implementation'
            }
        } as e:
            logger.error(f"Error searching PubMed: {e}")
            return []
    
    async def fetch_pubmed_details(self, pmids: List[str]) -> List[Dict]:
        """
        Fetch detailed article information from PubMed
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of detailed article information
        """
        if not pmids:
            return []
        
        try:
            await self._rate_limit("pubmed")
            
            # Batch fetch article details
            fetch_url = f"{self.pubmed_base_url}/efetch.fcgi"
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'xml',
                'tool': 'medical_research_agent',
                'email': self.email
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(fetch_url, params=fetch_params) as response:
                    if response.status != 200:
                        logger.error(f"PubMed fetch failed with status {response.status}")
                        return []
                    
                    xml_content = await response.text()
            
            # Parse detailed article information
            root = ET.fromstring(xml_content)
            articles = []
            
            for article in root.findall('.//PubmedArticle'):
                try:
                    article_data = self._parse_pubmed_article(article)
                    if article_data:
                        articles.append(article_data)
                except Exception as e:
                    logger.warning(f"Error parsing individual PubMed article: {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(articles)} PubMed articles")
            return articles
            
        except Exception
