const OpenAI = require('openai');

class BaseAgent {
  constructor(name, instructions) {
    this.name = name;
    this.instructions = instructions;
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    });
  }

  async createOpenAIAgent() {
    try {
      const assistant = await this.openai.beta.assistants.create({
        name: this.name,
        instructions: this.instructions,
        model: 'gpt-4.1',
        tools: this.getTools()
      });
      
      this.assistantId = assistant.id;
      return assistant;
    } catch (error) {
      console.error(`Error creating OpenAI agent ${this.name}:`, error);
      throw error;
    }
  }

  async runAgent(query, context = {}) {
    try {
      // Create a thread
      const thread = await this.openai.beta.threads.create();
      
      // Add message to thread
      await this.openai.beta.threads.messages.create(thread.id, {
        role: 'user',
        content: this.formatQuery(query, context)
      });

      // Run the assistant
      const run = await this.openai.beta.threads.runs.create(thread.id, {
        assistant_id: this.assistantId || await this.getOrCreateAssistant()
      });

      // Wait for completion
      const completedRun = await this.waitForRunCompletion(thread.id, run.id);
      
      // Get response
      const messages = await this.openai.beta.threads.messages.list(thread.id);
      const response = messages.data[0].content[0].text.value;

      return this.parseResponse(response, query, context);
      
    } catch (error) {
      console.error(`Error running agent ${this.name}:`, error);
      return this.getErrorResponse(error);
    }
  }

  async waitForRunCompletion(threadId, runId, maxAttempts = 30) {
    for (let i = 0; i < maxAttempts; i++) {
      const run = await this.openai.beta.threads.runs.retrieve(threadId, runId);
      
      if (run.status === 'completed') {
        return run;
      } else if (run.status === 'failed' || run.status === 'cancelled') {
        throw new Error(`Run ${run.status}: ${run.last_error?.message || 'Unknown error'}`);
      }
      
      // Wait before checking again
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
    
    throw new Error('Run timed out');
  }

  async getOrCreateAssistant() {
    if (!this.assistantId) {
      const assistant = await this.createOpenAIAgent();
      this.assistantId = assistant.id;
    }
    return this.assistantId;
  }

  formatQuery(query, context) {
    return `Query: ${query}\n\nContext: ${JSON.stringify(context, null, 2)}`;
  }

  parseResponse(response, query, context) {
    try {
      // Try to parse as JSON first
      return JSON.parse(response);
    } catch {
      // Return as structured object
      return {
        summary: response.substring(0, 500),
        fullResponse: response,
        query: query,
        timestamp: new Date().toISOString()
      };
    }
  }

  getErrorResponse(error) {
    return {
      error: true,
      message: error.message,
      summary: `Error in ${this.name}: ${error.message}`,
      timestamp: new Date().toISOString()
    };
  }

  getTools() {
    return []; // Override in subclasses
  }
}

module.exports = { BaseAgent };
