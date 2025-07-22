"""
Simple Pinecone HTTP client to avoid Rust compilation issues
Uses direct HTTP requests instead of the official client
"""

import aiohttp
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional
import asyncio
import os

logger = logging.getLogger(__name__)

class SimplePineconeClient:
    """
    Simple HTTP-based Pinecone client to avoid Rust dependencies
    """
    
    def __init__(self):
        self.api_key = None
        self.environment = None
        self.index_name = None
        self.base_url = None
        self.available = False
        
        # Try to initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize Pinecone connection settings."""
        try:
            self.api_key = os.getenv("PINECONE_API_KEY")
            self.project_id = os.getenv("PINECONE_PROJECT_ID")
            self.environment = os.getenv("PINECONE_ENVIRONMENT")
            self.index_name = os.getenv("PINECONE_INDEX_NAME")

            if not all([self.api_key, self.project_id, self.environment, self.index_name]):
                logger.warning("One or more Pinecone environment variables are missing. Vector storage will be disabled.")
                return

            # This line builds the final URL from your .env variables
            self.base_url = f"https://{self.index_name}-{self.project_id}.svc.{self.environment}.pinecone.io"
            self.available = True

            logger.info(f"Simple Pinecone client initialized successfully for URL: {self.base_url}")

        except Exception as e:
            logger.error(f"Error initializing Pinecone client: {e}")
            self.available = False
    
    async def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make HTTP request to Pinecone API"""
        if not self.available:
            return {"error": "Pinecone not available"}
        
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Api-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                if method.upper() == "GET":
                    async with session.get(url, headers=headers) as response:
                        return await response.json()
                elif method.upper() == "POST":
                    async with session.post(url, headers=headers, json=data) as response:
                        return await response.json()
                else:
                    return {"error": f"Unsupported method: {method}"}
                    
        except Exception as e:
            logger.error(f"Pinecone API request failed: {e}")
            return {"error": str(e)}
    
    async def initialize_index(self, dimension: int = 1536):
        """Initialize index (Note: Index creation requires Pinecone console)"""
        if not self.available:
            logger.warning("Pinecone not available. Skipping index initialization.")
            return False
        
        try:
            # Test if index exists by making a simple stats call
            stats = await self._make_request("GET", "/describe_index_stats")
            
            if "error" not in stats:
                logger.info(f"Pinecone index {self.index_name} is ready")
                return True
            else:
                logger.warning(f"Pinecone index {self.index_name} may not exist. Create it in Pinecone console.")
                return False
                
        except Exception as e:
            logger.error(f"Error checking Pinecone index: {e}")
            return False
    
    def generate_id(self, content: str) -> str:
        """Generate unique ID for content"""
        return hashlib.md5(content.encode()).hexdigest()
    
    async def upsert(self, vectors: List[Dict]) -> bool:
        """Upsert vectors into Pinecone index"""
        if not self.available:
            logger.warning("Pinecone not available. Skipping vector upsert.")
            return False
        
        try:
            data = {"vectors": vectors}
            result = await self._make_request("POST", "/vectors/upsert", data)
            
            if "error" not in result:
                logger.info(f"Successfully upserted {len(vectors)} vectors")
                return True
            else:
                logger.error(f"Vector upsert failed: {result.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"Error upserting vectors: {e}")
            return False
    
    async def query(self, vector: List[float], top_k: int = 5, 
                   include_metadata: bool = True) -> List[Dict]:
        """Query similar vectors"""
        if not self.available:
            logger.warning("Pinecone not available. Cannot query vectors.")
            return []
        
        try:
            data = {
                "vector": vector,
                "topK": top_k,
                "includeMetadata": include_metadata
            }
            
            result = await self._make_request("POST", "/query", data)
            
            if "error" not in result and "matches" in result:
                return result["matches"]
            else:
                logger.error(f"Vector query failed: {result.get('error', 'Unknown error')}")
                return []
                
        except Exception as e:
            logger.error(f"Error querying vectors: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status"""
        return {
            'available': self.available,
            'index_name': self.index_name,
            'environment': self.environment,
            'base_url': self.base_url,
            'client_type': 'simple_http_client'
        }

class PineconeManager:
    """
    Vector database manager using simple HTTP client
    """
    
    def __init__(self):
        self.client = SimplePineconeClient()
        self.available = self.client.available
        
    async def initialize_index(self, dimension: int = 1536):
        """Initialize Pinecone index"""
        return await self.client.initialize_index(dimension)
    
    def generate_id(self, content: str) -> str:
        """Generate unique ID for content"""
        return self.client.generate_id(content)
    
    async def store_research(self, research_data: Dict[str, Any], 
                           embedding: Optional[List[float]] = None) -> bool:
        """Store research results in vector database"""
        if not self.available:
            logger.warning("Vector storage not available. Research not stored.")
            return False
        
        if not embedding:
            logger.warning("No embedding provided for vector storage.")
            return False
        
        try:
            research_id = self.generate_id(research_data['query'])
            
            # Prepare metadata (Pinecone has limits on metadata size)
            metadata = {
                'query': research_data['query'][:500],  # Truncate long queries
                'therapy_area': research_data.get('therapy_area', 'general'),
                'research_type': research_data.get('research_type', 'literature_review'),
                'timestamp': research_data.get('timestamp', ''),
                'sources_count': min(research_data.get('sources_analyzed', 0), 9999)  # Ensure it's a reasonable number
            }
            
            # Prepare vector for upsert
            vector_data = [{
                "id": research_id,
                "values": embedding,
                "metadata": metadata
            }]
            
            success = await self.client.upsert(vector_data)
            
            if success:
                logger.info(f"Research stored in vector database with ID: {research_id}")
                return True
            else:
                logger.error("Failed to store research in vector database")
                return False
                
        except Exception as e:
            logger.error(f"Error storing research in vector database: {e}")
            return False
    
    async def search_similar(self, query_embedding: List[float], 
                           top_k: int = 5) -> List[Dict]:
        """Search for similar research"""
        if not self.available:
            logger.warning("Vector search not available.")
            return []
        
        try:
            matches = await self.client.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            return matches
            
        except Exception as e:
            logger.error(f"Error searching similar research: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get vector store status"""
        return self.client.get_status()

# Initialize global vector store instance
vector_store = PineconeManager()
