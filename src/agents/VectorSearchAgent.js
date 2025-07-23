const { Pinecone } = require('@pinecone-database/pinecone');

class VectorSearchAgent {
  constructor() {
    this.name = 'Vector Search Specialist';
    this.pinecone = null;
    this.index = null;
    this.initializePinecone();
  }

  async initializePinecone() {
    if (!process.env.PINECONE_API_KEY) {
      console.error('CRITICAL: Pinecone API key not found.');
      return;
    }

    try {
      this.pinecone = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY
      });
      
      this.index = this.pinecone.index('attruby-claims');
      console.log('Pinecone initialized successfully');
      
    } catch (error) {
      console.error('Failed to initialize Pinecone:', error);
      this.index = null;
    }
  }

  async search(params) {
    const { query, top_k = 10 } = params;
    
    if (!this.index) {
      return {
        error: true,
        message: 'Pinecone not available',
        query: query,
        relevantDocuments: []
      };
    }

    try {
      // Generate embedding
      const embedding = await this.generateEmbedding(query);
      
      // Simple query - matching your Python exactly
      const queryResponse = await this.index.query({
        vector: embedding,
        topK: top_k,
        includeMetadata: true
      });
      
      // Minimal response to avoid ResponseTooLargeError
      const results = {
        query: query,
        totalResults: queryResponse.matches ? queryResponse.matches.length : 0,
        relevantDocuments: []
      };

      // Only include essential data for first 5 matches
      if (queryResponse.matches && queryResponse.matches.length > 0) {
        results.relevantDocuments = queryResponse.matches.slice(0, 5).map((match, index) => ({
          rank: index + 1,
          score: Math.round(match.score * 1000) / 1000,
          title: match.metadata?.title || 'Unknown',
          authors: match.metadata?.authors || 'Unknown',
          citation: match.metadata?.citation || 'Citation unavailable',
          preview: match.metadata?.chunk_preview?.substring(0, 150) + '...' || 'Preview unavailable'
        }));
      }

      // Add minimal summary
      results.summary = `Found ${results.totalResults} documents. Top match: ${results.relevantDocuments[0]?.title || 'None'} (score: ${results.relevantDocuments[0]?.score || 0})`;
      
      return results;
      
    } catch (error) {
      console.error('Vector search error:', error);
      return {
        error: true,
        message: error.message,
        query: query,
        relevantDocuments: []
      };
    }
  }

  async generateEmbedding(text) {
    const OpenAI = require('openai');
    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    
    const response = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: text,
      encoding_format: 'float',
    });

    return response.data[0].embedding;
  }
}

module.exports = { VectorSearchAgent };
