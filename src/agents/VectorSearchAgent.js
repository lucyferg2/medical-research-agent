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
      // Initialize Pinecone client
      this.pinecone = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY
      });
      
      // Get index reference - namespace handled differently in v3+
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
      // Test connection and get index stats
      const indexStats = await this.index.describeIndexStats();
      
      console.log('Pinecone connection verified.');
      console.log('Index stats:', {
        totalVectorCount: indexStats.totalVectorCount || indexStats.total_vector_count || 'unknown',
        dimension: indexStats.dimension,
        indexFullness: indexStats.indexFullness,
        namespaces: indexStats.namespaces ? Object.keys(indexStats.namespaces) : 'none'
      });
      
      // Check if our namespace exists
      if (indexStats.namespaces && indexStats.namespaces['Test Deck']) {
        console.log('Test Deck namespace found with', indexStats.namespaces['Test Deck'].vectorCount, 'vectors');
      } else {
        console.log('Available namespaces:', Object.keys(indexStats.namespaces || {}));
      }
      
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

      // For JavaScript SDK v3+, we need to get a namespace-specific index reference
      let indexToQuery = this.index;
      
      if (namespace && namespace !== '' && namespace !== 'default') {
        console.log('Creating namespace-specific index reference for:', namespace);
        // In v3+ SDK, namespace is specified when getting the index
        indexToQuery = this.pinecone.index('attruby-claims').namespace(namespace);
      } else {
        console.log('Using default namespace (no namespace specified)');
      }

      const results = await this.queryPineconeWithIndex(indexToQuery, query, top_k, include_metadata);
      return this.formatPineconeResults(results, query);
      
    } catch (error) {
      console.error('Vector search error:', error);
      return this.getErrorResponse(error, query);
    }
  }

  async queryPineconeWithIndex(indexRef, query, topK, includeMetadata) {
    try {
      // Generate embedding for the query using OpenAI
      const embedding = await this.generateEmbedding(query);
      
      console.log('Generated embedding:', {
        length: embedding.length,
        dimension: embedding.length,
        sample: embedding.slice(0, 3)
      });

      // FIXED: Use only valid properties for Pinecone JavaScript SDK v3+
      // Valid properties: id, vector, sparseVector, includeValues, includeMetadata, filter, topK
      console.log('Querying Pinecone with JavaScript SDK v3+ format');
      
      const queryOptions = {
        vector: embedding,
        topK: topK,
        includeMetadata: includeMetadata,
        includeValues: false  // We don't need vector values in response
      };

      console.log('Query options:', {
        hasVector: !!queryOptions.vector,
        vectorDimension: queryOptions.vector.length,
        topK: queryOptions.topK,
        includeMetadata: queryOptions.includeMetadata,
        includeValues: queryOptions.includeValues
      });

      // Execute query using the index reference (with or without namespace)
      const queryResponse = await indexRef.query(queryOptions);
      
      console.log('Pinecone query successful:', {
        matchCount: queryResponse.matches?.length || 0,
        hasUsage: !!queryResponse.usage
      });
      
      if (queryResponse.matches && queryResponse.matches.length > 0) {
        const firstMatch = queryResponse.matches[0];
        console.log('Top result preview:', {
          id: firstMatch.id,
          score: Math.round(firstMatch.score * 1000) / 1000,
          hasMetadata: !!firstMatch.metadata,
          metadataKeys: firstMatch.metadata ? Object.keys(firstMatch.metadata).slice(0, 5) : []
        });
      }
      
      return queryResponse;
      
    } catch (error) {
      console.error('Pinecone query failed:', error);
      
      // Enhanced error logging for debugging
      console.error('Error details:', {
        name: error.name,
        message: error.message,
        status: error.status,
        body: error.body
      });
      
      throw error;
    }
  }

  // Remove the old queryPinecone method since we now use queryPineconeWithIndex

  async generateEmbedding(text) {
    try {
      const OpenAI = require('openai');
      const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
      
      console.log('Generating embedding for:', text.substring(0, 50) + '...');
      
      const response = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float',
      });

      const embedding = response.data[0].embedding;
      
      // Validate embedding dimension matches index
      if (embedding.length !== 1536) {
        throw new Error(`Embedding dimension mismatch: got ${embedding.length}, expected 1536`);
      }
      
      console.log('Embedding generated successfully:', {
        dimension: embedding.length,
        model: 'text-embedding-3-small'
      });
      
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
      hasUsage: !!queryResponse.usage
    });
    
    const formattedResults = {
      query: originalQuery,
      totalResults: matches.length,
      keyTerms: this.extractKeyTerms(originalQuery),
      relevantDocuments: matches.map((match, index) => ({
        id: match.id,
        score: Math.round(match.score * 1000) / 1000,
        rank: index + 1,
        title: match.metadata?.title || 'Unknown Title',
        authors: match.metadata?.authors || 'Unknown Authors',
        citation: match.metadata?.citation || this.buildCitation(match.metadata),
        chunk_preview: match.metadata?.chunk_preview || 
                      match.metadata?.document?.substring(0, 300) + '...' ||
                      'Preview not available',
        doi: match.metadata?.doi || null,
        journal: match.metadata?.journal || null,
        published: match.metadata?.published || null,
        page_reference: match.metadata?.page_reference || null,
        citation_count: match.metadata?.citation_count || null,
        source_file: match.metadata?.source_file || null,
        chunk_index: match.metadata?.chunk_index || null
      })),
      knowledgeGaps: this.identifyKnowledgeGaps(matches, originalQuery),
      searchStrategy: 'semantic_similarity',
      summary: this.generateSummary(matches, originalQuery),
      metadata: {
        namespace: 'Test Deck',
        searchTime: new Date().toISOString(),
        averageScore: matches.length > 0 ? 
          Math.round((matches.reduce((sum, m) => sum + m.score, 0) / matches.length) * 1000) / 1000 : 0,
        topScore: matches.length > 0 ? Math.round(matches[0].score * 1000) / 1000 : 0,
        indexUsed: 'attruby-claims',
        usage: queryResponse.usage || null
      }
    };

    return formattedResults;
  }

  buildCitation(metadata) {
    if (!metadata) return 'Citation information unavailable';
    
    // Use existing citation if available, otherwise build one
    if (metadata.citation) {
      return metadata.citation;
    }
    
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
    const stopWords = ['therapy', 'treatment', 'analysis', 'study', 'research', 'clinical', 'medical'];
    const terms = query.toLowerCase()
      .split(/\s+/)
      .filter(term => term.length > 3)
      .filter(term => !stopWords.includes(term))
      .slice(0, 10); // Limit to top 10 terms
    
    return [...new Set(terms)];
  }

  identifyKnowledgeGaps(matches, query) {
    const gaps = [];
    
    if (matches.length === 0) {
      gaps.push(`No relevant documents found for "${query}"`);
      gaps.push('Try broader search terms or synonyms');
      gaps.push('Check if topic is covered in knowledge base');
    } else if (matches.length < 3) {
      gaps.push(`Limited literature available - only ${matches.length} relevant documents found`);
      gaps.push('Consider expanding search scope');
    }
    
    // Analyze score distribution
    const scores = matches.map(m => m.score);
    const avgScore = scores.reduce((sum, score) => sum + score, 0) / scores.length;
    
    if (avgScore < 0.7) {
      gaps.push('Low relevance scores suggest query terms may not match knowledge base content well');
      gaps.push('Consider using more specific terminology or domain-specific keywords');
    }
    
    // Check for recent publications
    const recentDocs = matches.filter(m => {
      const pubDate = new Date(m.metadata?.published || '2000-01-01');
      const twoYearsAgo = new Date();
      twoYearsAgo.setFullYear(twoYearsAgo.getFullYear() - 2);
      return pubDate > twoYearsAgo;
    });
    
    if (matches.length > 0 && recentDocs.length < matches.length * 0.3) {
      gaps.push('Limited recent literature (within last 2 years) found in knowledge base');
      gaps.push('May need to search external databases for latest research');
    }
    
    return gaps.slice(0, 5); // Limit to top 5 gaps
  }

  generateSummary(matches, query) {
    if (matches.length === 0) {
      return `No relevant documents found in the knowledge base for "${query}". The knowledge base may not contain information on this specific topic, or different search terms may be needed.`;
    }
    
    const topMatch = matches[0];
    const avgScore = matches.reduce((sum, m) => sum + m.score, 0) / matches.length;
    const uniqueAuthors = new Set(matches.map(m => m.metadata?.authors).filter(Boolean)).size;
    const uniqueJournals = new Set(matches.map(m => m.metadata?.journal).filter(Boolean)).size;
    const uniqueSources = new Set(matches.map(m => m.metadata?.source_file).filter(Boolean)).size;
    
    return `Retrieved ${matches.length} relevant documents for "${query}" with average similarity score of ${avgScore.toFixed(3)}. ` +
           `Best match: "${topMatch.metadata?.title || 'Unknown'}" (relevance: ${topMatch.score.toFixed(3)}). ` +
           `Results represent ${uniqueAuthors} different author groups across ${uniqueJournals} journals from ${uniqueSources} source documents. ` +
           `This provides a ${matches.length >= 5 ? 'comprehensive' : 'limited but relevant'} evidence base for further analysis.`;
  }

  getVectorServiceUnavailableResponse(query) {
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
        'Verify Pinecone API configuration in environment variables',
        'Check network connectivity to Pinecone service',
        'Ensure PINECONE_API_KEY is set correctly',
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
        'Verify Pinecone index configuration and accessibility',
        'Ensure embedding model compatibility (text-embedding-3-small)',
        'Retry the search after resolving technical issues'
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
