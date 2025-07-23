const { v4: uuidv4 } = require('uuid');

function generateSessionId() {
  return uuidv4();
}

async function validateInput(query, context = {}) {
  const errors = [];
  const warnings = [];

  if (!query || typeof query !== 'string' || query.trim().length === 0) {
    errors.push('Query is required and must be a non-empty string');
  }

  if (query && query.length > 5000) {
    errors.push('Query is too long (maximum 5000 characters)');
  }

  // Check for potential PHI
  const phiPatterns = [
    /\b\d{3}-\d{2}-\d{4}\b/, // SSN
    /\b[A-Z]{2}\d{6}\b/, // Medical record patterns
    /\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b/, // Credit card
    /\b\d{10,11}\b/ // Phone numbers
  ];

  const containsPHI = phiPatterns.some(pattern => pattern.test(query));
  if (containsPHI) {
    warnings.push('Query may contain sensitive information');
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings
  };
}

function sanitizeInput(input) {
  if (typeof input !== 'string') return input;
  
  // Remove potential harmful characters
  return input
    .replace(/[<>]/g, '') // Remove angle brackets
    .replace(/javascript:/gi, '') // Remove javascript: protocol
    .replace(/on\w+=/gi, '') // Remove event handlers
    .trim();
}

module.exports = {
  generateSessionId,
  validateInput,
  sanitizeInput
};
