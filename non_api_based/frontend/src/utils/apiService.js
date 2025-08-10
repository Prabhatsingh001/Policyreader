// API Service for handling all API calls and data processing

/**
 * Simulates API delay for realistic user experience
 * @param {number} ms - Delay in milliseconds
 * @returns {Promise}
 */
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Process user query and return analysis results
 * @param {string} query - User's natural language query
 * @returns {Promise<Object>} - Analysis results
 */
export const processQuery = async (query) => {
  // Simulate API delay
  await delay(2000);

  // TODO: Replace with actual LLM API calls
  // const parsedQuery = await callGeminiAPI('parse', query);
  // const decision = await callGeminiAPI('evaluate', parsedQuery, documents);

  // Simulated response based on query analysis
  const mockResponse = generateMockResponse(query);
  
  return mockResponse;
};

/**
 * Generate mock response based on query content
 * @param {string} query - User query
 * @returns {Object} - Mock analysis response
 */
const generateMockResponse = (query) => {
  // Extract basic information from query for more realistic response
  const extractedInfo = extractQueryInfo(query);
  
  return {
    query_analysis: {
      patient_age: extractedInfo.age || 46,
      gender: extractedInfo.gender || "male",
      procedure: extractedInfo.procedure || "knee surgery",
      location: extractedInfo.location || "Pune",
      policy_age: extractedInfo.policyAge || "3 months"
    },
    decision: determineDecision(extractedInfo),
    coverage_amount: calculateCoverage(extractedInfo),
    justification: generateJustification(extractedInfo),
    relevant_clauses: [
      "Clause 3.1: Pre-existing conditions coverage",
      "Clause 3.3: Orthopedic surgery coverage limit"
    ],
    confidence_score: Math.random() * 0.3 + 0.7 // Random between 0.7-1.0
  };
};

/**
 * Extract information from user query using simple pattern matching
 * @param {string} query - User query
 * @returns {Object} - Extracted information
 */
const extractQueryInfo = (query) => {
  const info = {};
  
  // Extract age
  const ageMatch = query.match(/(\d+)[-\s]?year[-\s]?old/i);
  if (ageMatch) info.age = parseInt(ageMatch[1]);
  
  // Extract gender
  if (query.toLowerCase().includes('female') || query.toLowerCase().includes('woman')) {
    info.gender = 'female';
  } else if (query.toLowerCase().includes('male') || query.toLowerCase().includes('man')) {
    info.gender = 'male';
  }
  
  // Extract procedure types
  const procedures = ['surgery', 'operation', 'treatment', 'procedure', 'therapy'];
  for (const proc of procedures) {
    if (query.toLowerCase().includes(proc)) {
      info.procedure = query.toLowerCase().includes('knee') ? 'knee surgery' :
                      query.toLowerCase().includes('heart') ? 'heart surgery' :
                      query.toLowerCase().includes('eye') ? 'eye surgery' :
                      `${proc}`;
      break;
    }
  }
  
  // Extract location
  const cities = ['pune', 'mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata'];
  for (const city of cities) {
    if (query.toLowerCase().includes(city)) {
      info.location = city.charAt(0).toUpperCase() + city.slice(1);
      break;
    }
  }
  
  // Extract policy age
  const policyMatch = query.match(/(\d+)[-\s]?(month|year)/i);
  if (policyMatch) {
    info.policyAge = `${policyMatch[1]} ${policyMatch[2]}${policyMatch[1] > 1 ? 's' : ''}`;
  }
  
  return info;
};

/**
 * Determine approval decision based on extracted information
 * @param {Object} info - Extracted query information
 * @returns {string} - Decision status
 */
const determineDecision = (info) => {
  // Simple logic for demo purposes
  if (info.age && info.age > 80) return "REQUIRES_REVIEW";
  if (info.policyAge && info.policyAge.includes('month') && parseInt(info.policyAge) < 6) {
    return "CONDITIONAL_APPROVAL";
  }
  return "APPROVED";
};

/**
 * Calculate coverage amount based on procedure and other factors
 * @param {Object} info - Extracted query information
 * @returns {string} - Coverage amount
 */
const calculateCoverage = (info) => {
  let baseAmount = 200000;
  
  if (info.procedure) {
    if (info.procedure.includes('heart')) baseAmount = 500000;
    else if (info.procedure.includes('knee')) baseAmount = 350000;
    else if (info.procedure.includes('eye')) baseAmount = 150000;
  }
  
  // Format as Indian currency
  return `₹${baseAmount.toLocaleString('en-IN')}`;
};

/**
 * Generate justification text based on decision
 * @param {Object} info - Extracted query information
 * @returns {string} - Justification text
 */
const generateJustification = (info) => {
  const procedure = info.procedure || 'the requested procedure';
  
  return `Based on the health insurance policy guidelines, ${procedure} is covered under the applicable medical procedures. The patient meets the eligibility criteria including policy maintenance requirements and medical necessity standards.`;
};

/**
 * Call Gemini API (placeholder for actual implementation)
 * @param {string} action - API action type
 * @param {string} query - User query
 * @param {Object} context - Additional context
 * @returns {Promise} - API response
 */
export const callGeminiAPI = async (action, query, context = {}) => {
  // TODO: Implement actual Gemini API calls
  throw new Error('Gemini API integration not implemented yet');
};

/**
 * Upload and process document
 * @param {File} file - Document file
 * @returns {Promise<Object>} - Processing result
 */
export const uploadDocument = async (file) => {
  // TODO: Implement document upload and processing
  await delay(1500);
  
  return {
    success: true,
    filename: file.name,
    size: file.size,
    processed: true,
    message: 'Document successfully processed and indexed for policy analysis'
  };
};