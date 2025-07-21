import hashlib
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class PineconeManager:
    """
    Pinecone vector database manager for storing and retrieving research results
    """
    
    def __init__(self):
        self.api_key = None
        self.environment = None
        self.index_name = None
        self.index = None
        self.available = False
        
        # Try to initialize Pinecone
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone connection"""
        try:
            from app.config import settings
            
            if not settings.pinecone_api_key:
                logger.warning("Pinecone API key not provided. Vector storage will be disabled.")
                return
            
            # Import pinecone here to avoid import errors if not installed
            import pinecone
            
            pinecone.init(
                api_key=settings.pinecone_api_key,
                environment=settings.pinecone_environment
            )
            
            self.api_key = settings.pinecone_api_key
            self.environment = settings.pinecone_environment
            self.index_name = settings.pinecone_index_name
            self.available = True
            
            logger.info("Pinecone initialized successfully")
            
        except ImportError:
            logger.warning("Pinecone package not installed. Vector storage will be disabled.")
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
    
    async def initialize_index(self, dimension: int = 1536):
        """Initialize Pinecone index if it doesn't exist"""
        if not self.available:
            logger.warning("Pinecone not available. Skipping index initialization.")
            return False
        
        try:
            import pinecone
            
            # Check if index exists
            existing_indexes = pinecone.list_indexes()
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                pinecone.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine"
                )
            
            self.index = pinecone.Index(self.index_name)
            logger.info(f"Pinecone index {self.index_name} ready")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {e}")
            return False
    
    def generate_id(self, content: str) -> str:
        """Generate unique ID for content"""
        return hashlib.md5(content.encode()).hexdigest()
    
    async def store_research(self, research_data: Dict[str, Any], 
                           embedding: Optional[List[float]] = None) -> bool:
        """Store research results in vector database"""
        if not self.available or not self.index:
            logger.warning("Pinecone not available. Research not stored in vector database.")
            return False
        
        try:
            research_id = self.generate_id(research_data['query'])
            
            # If no embedding provided, create a placeholder or skip
            if not embedding:
                logger.warning("No embedding provided. Skipping vector storage.")
                return False
            
            metadata = {
                'query': research_data['query'][:1000],  # Limit metadata size
                'therapy_area': research_data.get('therapy_area', 'general'),
                'research_type': research_data.get('research_type', 'literature_review'),
                'timestamp': research_data.get('timestamp'),
                'sources_count': research_data.get('sources_analyzed', 0)
            }
            
            self.index.upsert(vectors=[(research_id, embedding, metadata)])
            logger.info(f"Research stored in vector database with ID: {research_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing research in vector database: {e}")
            return False
    
    async def search_similar(self, query_embedding: List[float], 
                           top_k: int = 5) -> List[Dict]:
        """Search for similar research"""
        if not self.available or not self.index:
            logger.warning("Pinecone not available. Cannot search similar research.")
            return []
        
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            return results.matches
            
        except Exception as e:
            logger.error(f"Error searching similar research: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get Pinecone connection status"""
        return {
            'available': self.available,
            'index_name': self.index_name,
            'environment': self.environment,
            'index_ready': self.index is not None
        }

# Initialize global vector store instance
vector_store = PineconeManager()
