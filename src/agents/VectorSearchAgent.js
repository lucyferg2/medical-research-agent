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
      console.error('CRITICAL: Pinecone API key not found. Vector search will not function.');
      console.error('Medical research requires actual data - no mock results will be provided.');
      return;
    }

    try {
      this.pinecone = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY
      });
      
      this.index = this.pinecone.index('attruby-claims');
      console.log('Pinecone initialized successfully for medical research');
      
      // Test the connection
      await this.testConnection();
      
    } catch (error) {
      console.error('CRITICAL: Failed to initialize Pinecone for medical research:', error);
      console.error('System will not provide mock medical data. Fix Pinecone connection immediately.');
      this.index = null;
    }
  }

  async testConnection() {
    try {
      // Test with a simple query to verify connection
      const testQuery = await this.index.describeIndexStats();
      console.log('Pinecone connection verified. Index contains:', testQuery.totalVectorCount, 'vectors');
      console.log('Index dimension:', testQuery.dimension);
    } catch (error) {
      console.error('Pinecone connection test failed:', error);
      throw error;
    }
  }

  async search(params) {
    const { query, namespace = 'Test Deck', top_k = 10, include_metadata = true } = params;
    
    try {
      // Require Pinecone to be available - no mock data for medical content
      if (!this.index) {
        console.error('Vector search attempted but Pinecone not available');
        return this.getVectorServiceUnavailableResponse(query);
      }

      const results = await this.queryPinecone(query, namespace, top_k, include_metadata);
      return this.formatPineconeResults(results, query);
      
    } catch (error) {
      console.error('Vector search error:', error);
      return this.getErrorResponse(error, query);
    }
  }

  async queryPinecone(query, namespace, topK, includeMetadata) {
    try {
      // Generate embedding for the query using OpenAI
      const embedding = await this.generateEmbedding(query);
      
      console.log('Generated embedding:', {
        length: embedding.length,
        firstFew: embedding.slice(0, 3),
        lastFew: embedding.slice(-3)
      });

      // Try multiple query formats to match different SDK versions
      let queryResponse;
      
      // Format 1: Direct parameters (like Python SDK)
      try {
        console.log('Attempting query format 1: Direct parameters');
        queryResponse = await this.index.query({
          namespace: namespace,
          vector: embedding,
          topK: topK,
          includeMetadata: includeMetadata
        });
        console.log('Format 1 successful');
      } catch (error1) {
        console.log('Format 1 failed:', error1.message);
        
        // Format 2: Camel case parameters
        try {
          console.log('Attempting query format 2: Camel case');
          queryResponse = await this.index.query({
            namespace: namespace,
            vector: embedding,
            top_k: topK,
            include_metadata: includeMetadata
          });
          console.log('Format 2 successful');
        } catch (error2) {
          console.log('Format 2 failed:', error2.message);
          
          // Format 3: Without namespace (if namespace is causing issues)
          try {
            console.log('Attempting query format 3: No namespace');
            queryResponse = await this.index.query({
              vector: embedding,
              topK: topK,
              includeMetadata: includeMetadata
            });
            console.log('Format 3 successful');
          } catch (error3) {
            console.log('Format 3 failed:', error3.message);
            
            // Format 4: Minimal parameters only
            try {
              console.log('Attempting query format 4: Minimal');
              queryResponse = await this.index.query({
                vector: embedding,
                topK: topK
              });
              console.log('Format 4 successful');
            } catch (error4) {
              console.log('All query formats failed. SDK might be incompatible.');
              throw error4;
            }
          }
        }
      }
      
      console.log('Pinecone query successful. Found', queryResponse.matches?.length || 0, 'matches');
      return queryResponse;
      
    } catch (error) {
      console.error('Pinecone query error:', error);
      console.error('Full error object:', JSON.stringify(error, null, 2));
      throw error;
    }
  }

  async generateEmbedding(text) {
    try {
      const OpenAI = require('openai');
      const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
      
      console.log('Generating embedding for text:', text.substring(0, 100) + '...');
      
      const response = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float',
      });

      const embedding = response.data[0].embedding;
      console.log('Embedding generated successfully. Length:', embedding.length);
      
      return embedding;
    } catch (error) {
      console.error('Embedding generation error:', error);
      throw error;
    }
  }

  formatPineconeResults(queryResponse, originalQuery) {
    const matches = queryResponse.matches || [];
    
    console.log('Formatting results:', {
      matchCount: matches.length,
      firstMatchId: matches[0]?.id,
      firstMatchScore: matches[0]?.score,
      hasMetadata: !!matches[0]?.metadata
    });
    
    const formattedResults = {
      query: originalQuery,
      totalResults: matches.length,
      keyTerms: this.extractKeyTerms(originalQuery),
      relevantDocuments: matches.map((match, index) => ({
        id: match.id,
        score: Math.round(match.score * 1000) / 1000, // Round to 3 decimal places
        rank: index + 1,
        title: match.metadata?.title || 'Unknown Title',
        authors: match.metadata?.authors || 'Unknown Authors',
        citation: match.metadata?.citation || this.buildCitation(match.metadata),
        chunk_preview: match.metadata?.chunk_preview || match.metadata?.document?.substring(0, 200) + '...',
        doi: match.metadata?.doi || null,
        journal: match.metadata?.journal || null,
        published: match.metadata?.published || null,
        page_reference: match.metadata?.page_reference || null,
        citation_count: match.metadata?.citation_count || null,
        source_file: match.metadata?.source_file || null
      })),
      knowledgeGaps: this.identifyKnowledgeGaps(matches, originalQuery),
      searchStrategy: 'semantic_similarity',
      summary: this.generateSummary(matches, originalQuery),
      metadata: {
        namespace: 'Test Deck',
        searchTime: new Date().toISOString(),
        averageScore: matches.length > 0 ? Math.round((matches.reduce((sum, m) => sum + m.score, 0) / matches.length) * 1000) / 1000 : 0,
        topScore: matches.length > 0 ? Math.round(matches[0].score * 1000) / 1000 : 0,
        indexUsed: 'attruby-claims'
      }
    };

    return formattedResults;
  }

  buildCitation(metadata) {
    if (!metadata) return 'Citation information unavailable';
    
    const title = metadata.title || 'Unknown Title';
    const authors = metadata.authors || 'Unknown Authors';
    const journal = metadata.journal || '';
    const published = metadata.published || '';
    const doi = metadata.doi || '';
    const citationCount = metadata.citation_count || 0;
    
    let citation = `"${title}" by ${authors}`;
    if (published) citation += ` (${published})`;
    if (journal) citation += `, ${journal}`;
    if (doi) citation += `, DOI: ${doi}`;
    if (citationCount > 0) citation += ` [Cited by: ${citationCount}]`;
    
    return citation;
  }

  extractKeyTerms(query) {
    // Simple key term extraction - you could enhance this with NLP
    const terms = query.toLowerCase()
      .split(/\s+/)
      .filter(term => term.length > 3)
      .filter(term => !['therapy', 'treatment', 'analysis', 'study'].includes(term));
    
    return [...new Set(terms)];
  }

  identifyKnowledgeGaps(matches, query) {
    const gaps = [];
    
    if (matches.length === 0) {
      gaps.push(`No relevant documents found for "${query}"`);
      gaps.push('Consider using broader or alternative search terms');
    } else if (matches.length < 5) {
      gaps.push(`Limited literature available on "${query}" - only ${matches.length} relevant documents found`);
    }
    
    // Check for low similarity scores
    const lowScoreMatches = matches.filter(m => m.score < 0.7);
    if (lowScoreMatches.length > matches.length / 2) {
      gaps.push('Query may be too specific or use terminology not well-represented in the knowledge base');
    }
    
    // Check for date gaps
    const recentDocs = matches.filter(m => {
      const pubDate = new Date(m.metadata?.published || '2000-01-01');
      const twoYearsAgo = new Date();
      twoYearsAgo.setFullYear(twoYearsAgo.getFullYear() - 2);
      return pubDate > twoYearsAgo;
    });
    
    if (recentDocs.length < matches.length * 0.3) {
      gaps.push('Limited recent literature (published within last 2 years) found');
      gaps.push('Consider searching for more recent publications outside this knowledge base');
    }
    
    return gaps;
  }

  generateSummary(matches, query) {
    if (matches.length === 0) {
      return `No relevant documents found in the knowledge base for "${query}". Consider broadening search terms or checking for alternative terminology.`;
    }
    
    const topMatch = matches[0];
    const avgScore = matches.reduce((sum, m) => sum + m.score, 0) / matches.length;
    const uniqueAuthors = new Set(matches.map(m => m.metadata?.authors).filter(Boolean)).size;
    const uniqueJournals = new Set(matches.map(m => m.metadata?.journal).filter(Boolean)).size;
    
    return `Found ${matches.length} relevant documents for "${query}" with average similarity score of ${avgScore.toFixed(3)}. ` +
           `Top result: "${topMatch.metadata?.title || 'Unknown'}" (score: ${topMatch.score.toFixed(3)}). ` +
           `Results span ${uniqueAuthors} unique author groups across ${uniqueJournals} different journals. ` +
           `This provides a solid foundation for literature analysis and further research direction.`;
  }

  getVectorServiceUnavailableResponse(query) {
    // Safe failure response - no fake medical data
    return {
      error: true,
      query: query,
      message: 'Vector search service is not available. Pinecone database connection required.',
      summary: `Unable to search knowledge base for "${query}". Vector database is not configured or accessible.`,
      relevantDocuments: [],
      keyTerms: [],
      knowledgeGaps: ['Vector search service unavailable - cannot access knowledge base'],
      searchStrategy: 'service_unavailable',
      recommendations: [
        'Verify Pinecone API configuration',
        'Check network connectivity to Pinecone service',
        'Ensure proper environment variables are set',
        'Contact system administrator if problem persists'
      ],
      timestamp: new Date().toISOString(),
      metadata: {
        hasError: true,
        errorType: 'ServiceUnavailable',
        requiresPinecone: true
      }
    };
  }

  getErrorResponse(error, query) {
    return {
      error: true,
      query: query,
      message: `Vector search failed: ${error.message}`,
      summary: `Unable to complete vector search for "${query}" due to system error. No results available.`,
      relevantDocuments: [],
      keyTerms: [],
      knowledgeGaps: ['Vector search system error - cannot access knowledge base'],
      searchStrategy: 'error_fallback',
      recommendations: [
        'Check system logs for detailed error information',
        'Verify database connectivity and configuration',
        'Retry the search after resolving technical issues',
        'Contact technical support if error persists'
      ],
      timestamp: new Date().toISOString(),
      metadata: {
        hasError: true,
        errorType: error.name || 'Unknown',
        errorMessage: error.message,
        requiresManualReview: true
      }
    };
  }
}

module.exports = { VectorSearchAgent };
