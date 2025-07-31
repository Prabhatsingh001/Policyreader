import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# System Configuration
SYSTEM_NAME = "Intelligent Query-Retrieval System"
VERSION = "1.0.0"

# Document Processing Configuration
DEFAULT_CHUNK_SIZE = 1500  
DEFAULT_CHUNK_OVERLAP = 300 
SUPPORTED_FORMATS = ['.pdf', '.docx', '.txt', '.eml', '.html']  

# Vector Store Configuration
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSION = 384
DEFAULT_SEARCH_THRESHOLD = 0.3
DEFAULT_TOP_K = 5

# LLM Configuration
# Use environment variable for Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEFAULT_LLM_MODEL = "gemini-2.5-pro" # Or another suitable Google model
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 4096  # Increased for larger context handling

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Intelligent Query-Retrieval System API"
API_DESCRIPTION = "LLM-Powered system for processing documents and answering queries in insurance, legal, HR, and compliance domains"

# File Paths
UPLOAD_DIR = Path("uploads")
SAMPLE_DOCS_DIR = Path("sample_docs")
SYSTEM_BACKUP_DIR = Path("backups")

# Create necessary directories
UPLOAD_DIR.mkdir(exist_ok=True)
SAMPLE_DOCS_DIR.mkdir(exist_ok=True)
SYSTEM_BACKUP_DIR.mkdir(exist_ok=True)

# Domain Classification
DOMAINS = {
    "insurance": ["policy", "coverage", "claim", "premium", "deductible", "insurance"],
    "legal": ["contract", "agreement", "clause", "legal", "compliance", "regulation"],
    "hr": ["employment", "employee", "benefits", "salary", "workplace", "hr"],
    "compliance": ["regulation", "compliance", "audit", "safety", "training", "penalty"]
}

# Query Templates for Different Domains
QUERY_TEMPLATES = {
    "insurance": [
        "What is the coverage limit for {coverage_type}?",
        "What is the claims process?",
        "What are the policy exclusions?",
        "What is the deductible amount?",
        "What is the premium cost?"
    ],
    "legal": [
        "What are the termination conditions?",
        "What are the confidentiality obligations?",
        "What are the non-compete restrictions?",
        "What are the legal obligations?",
        "What are the compliance requirements?"
    ],
    "hr": [
        "What is the salary structure?",
        "What benefits are provided?",
        "What is the work schedule?",
        "What training is required?",
        "What are the employment terms?"
    ],
    "compliance": [
        "What safety training is required?",
        "What are the data protection requirements?",
        "What are the penalty structures?",
        "What regulatory compliance is needed?",
        "What audit requirements exist?"
    ]
}

# Performance Configuration
MAX_DOCUMENT_SIZE = 50 * 1024 * 1024  # 50MB
MAX_BATCH_SIZE = 100
CACHE_ENABLED = True
CACHE_TTL = 3600  # 1 hour

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Security Configuration
ENABLE_AUTHENTICATION = False
ENABLE_RATE_LIMITING = True
MAX_REQUESTS_PER_MINUTE = 60

# Model Configuration
ENABLE_GPU_ACCELERATION = False
MODEL_CACHE_DIR = Path("model_cache")
MODEL_CACHE_DIR.mkdir(exist_ok=True)

# Response Configuration
MAX_RESPONSE_LENGTH = 2000
ENABLE_SOURCE_CITING = True
ENABLE_CONFIDENCE_SCORING = True
ENABLE_DOMAIN_CLASSIFICATION = True

# Error Handling
RETRY_ATTEMPTS = 3
RETRY_DELAY = 1  # seconds
TIMEOUT_SECONDS = 30

# Development Configuration
DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true"
ENABLE_PROFILING = False
ENABLE_METRICS = True