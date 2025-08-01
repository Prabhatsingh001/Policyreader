from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
#from langchain_ollama import OllamaLLM # Commented out Ollama import
from langchain_google_genai import ChatGoogleGenerativeAI # Import Google's LLM
from document_processor import DocumentProcessor, DocumentChunk
from vector_store import VectorStore
import json
from datetime import datetime
import os
from config import GOOGLE_API_KEY, DEFAULT_LLM_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS # Import config variables

class QueryResponse(BaseModel):
    """Structured response for query results."""
    answer: str = Field(description="The main answer to the query")
    confidence: float = Field(description="Confidence score (0-1)")
    sources: List[str] = Field(description="List of source documents used")
    reasoning: str = Field(description="Explanation of how the answer was derived")
    relevant_clauses: List[str] = Field(description="Relevant clauses or sections found")
    domain: str = Field(description="Domain classification (insurance, legal, hr, compliance)")
    timestamp: str = Field(description="Timestamp of the response")

class IntelligentQuerySystem:
    """Main RAG system for intelligent query processing."""
    
    def __init__(self):
        # Initialize Google's LLM with API Key from environment variable
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
            
        self.llm = ChatGoogleGenerativeAI(
            model=DEFAULT_LLM_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            #max_tokens=DEFAULT_MAX_TOKENS, # max_tokens is not a direct parameter, use max_output_tokens
            max_output_tokens=DEFAULT_MAX_TOKENS,
            google_api_key=GOOGLE_API_KEY
        )

        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        
        # Define the RAG prompt template
        self.prompt_template = ChatPromptTemplate.from_template("""
You are an expert AI assistant specializing in insurance, legal, HR, and compliance domains. 
Your task is to answer queries based on the provided document context.

## Context:
The following documents have been retrieved based on the user's query:
{documents}

## User Query:
{query}

## Instructions:
1. Analyze the provided documents carefully
2. Extract relevant information that directly addresses the query
3. If the documents don't contain sufficient information, state this clearly
4. Provide specific references to document sections when possible
5. Classify the domain (insurance, legal, hr, compliance)
6. Explain your reasoning process

## Response Format:
Provide a clear, concise answer based on the document context. If the documents don't contain relevant information, state this clearly.

## Response:
"""
)
        
        self.chain = self.prompt_template | self.llm
    
    def add_documents(self, directory_path: str):
        """Process and add documents to the system."""
        print(f"Processing documents from: {directory_path}")
        
        # Process documents
        chunks = self.document_processor.process_directory(directory_path)
        print(f"Processed {len(chunks)} document chunks")
        
        # Add to vector store
        self.vector_store.add_documents(chunks)
        print("Documents added to vector store")
    
    def query(self, user_query: str, top_k: int = 5, threshold: float = 0.01) -> QueryResponse:
        """Process a user query and return structured response."""
        
        # Perform semantic search using the correct method name and parameter
        search_results = self.vector_store.search(
            query=user_query, 
            k=top_k, 
            threshold=threshold
        )
        
        # Extract chunks from search results (search returns tuples of (chunk, score))
        relevant_chunks = [chunk for chunk, score in search_results]
        
        if not relevant_chunks:
            return QueryResponse(
                answer="No relevant documents found to answer your query.",
                confidence=0.0,
                sources=[],
                reasoning="No documents matched the query criteria.",
                relevant_clauses=[],
                domain="unknown",
                timestamp=datetime.now().isoformat()
            )
        
        # Prepare context from retrieved chunks
        context = self._prepare_context(relevant_chunks)
        
        # Generate response using LLM
        try:
            llm_response = self.chain.invoke({
                "documents": context,
                "query": user_query
            })
            
            # Extract the answer from the LLM response
            answer = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            
            # Determine confidence based on number of relevant chunks
            confidence = min(len(relevant_chunks) / top_k, 1.0)
            
            # Determine domain based on content analysis
            domain = self._classify_domain(user_query, answer)
            
            # Extract relevant clauses (first few words of each chunk)
            relevant_clauses = [chunk.content[:100] + "..." for chunk in relevant_chunks[:3]]
            
            return QueryResponse(
                answer=answer,
                confidence=confidence,
                sources=[chunk.source for chunk in relevant_chunks],
                reasoning=f"Found {len(relevant_chunks)} relevant document chunks that match the query.",
                relevant_clauses=relevant_clauses,
                domain=domain,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return QueryResponse(
                answer="Error processing your query. Please try again.",
                confidence=0.0,
                sources=[chunk.source for chunk in relevant_chunks],
                reasoning=f"Error occurred during response generation: {str(e)}",
                relevant_clauses=[],
                domain="unknown",
                timestamp=datetime.now().isoformat()
            )
    
    def _prepare_context(self, chunks: List[DocumentChunk]) -> str:
        """Prepare context string from document chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks):
            # Build the context for each chunk piece by piece for clarity
            part_lines = []
            part_lines.append(f"Document {i+1}: {chunk.source}")
            part_lines.append(f"{'='*50}")
            part_lines.append(f"Content: {chunk.content}")

            if chunk.page_number:
                part_lines.append(f"Page: {chunk.page_number}")
            if chunk.section:
                part_lines.append(f"Section: {chunk.section}")
            if chunk.metadata:
                # Safely handle metadata in case it's not JSON serializable
                try:
                    metadata_str = json.dumps(chunk.metadata)
                    part_lines.append(f"Metadata: {metadata_str}")
                except TypeError:
                    part_lines.append("Metadata: [Not serializable]")

            context_parts.append("".join(part_lines))

        # Join the parts for different chunks with a clear separator
        return "---".join(context_parts)

    def save_system(self, filepath: str):
        """Save the entire system state."""
        self.vector_store.save(filepath)
        print(f"System saved to: {filepath}")
    
    def load_system(self, filepath: str):
        """Load the system state."""
        self.vector_store.load(filepath)
        print(f"System loaded from: {filepath}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "vector_store": self.vector_store.get_statistics(),
            "llm_model": self.llm.model,
            "document_processor": {
                "chunk_size": self.document_processor.chunk_size,
                "chunk_overlap": self.document_processor.chunk_overlap
            }
        }
    
    def batch_query(self, queries: List[str]) -> List[QueryResponse]:
        """Process multiple queries in batch."""
        responses = []
        for query in queries:
            response = self.query(query)
            responses.append(response)
        return responses 

    def _classify_domain(self, query: str, answer: str) -> str:
        """Classify the domain based on query and answer content."""
        content = (query + " " + answer).lower()
        
        if any(word in content for word in ["policy", "coverage", "claim", "premium", "insurance"]):
            return "insurance"
        elif any(word in content for word in ["contract", "agreement", "clause", "legal", "compliance"]):
            return "legal"
        elif any(word in content for word in ["employment", "employee", "benefits", "salary", "hr"]):
            return "hr"
        elif any(word in content for word in ["regulation", "compliance", "audit", "safety", "training"]):
            return "compliance"
        else:
            return "unknown"