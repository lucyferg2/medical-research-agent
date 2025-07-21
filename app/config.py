import os
from typing import Optional

class Settings:
    """Application configuration settings"""
    
    def __init__(self):
        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
        # Pinecone Configuration
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "medical-research")
        
        # Research Configuration
        self.research_email = os.getenv("RESEARCH_EMAIL", "research@company.com")
        
        # API Configuration
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("PORT", 8000))  # Render uses PORT env var
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Validate required settings
        self._validate()
    
    def _validate(self):
        """Validate required configuration"""
        required_vars = ["OPENAI_API_KEY", "RESEARCH_EMAIL"]
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"WARNING: Missing required environment variables: {missing_vars}")
            print("Some features may not work correctly.")

settings = Settings()
