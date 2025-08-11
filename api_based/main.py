import json
import os
import asyncio
from datetime import datetime
from fastapi import FastAPI, Header, HTTPException, Depends, APIRouter
from pydantic import BaseModel, HttpUrl
from typing import List
from document_processor import DocumentProcessor
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants
API_PREFIX = "/api/v1"
API_VERSION = "1.0.0"
TEAM_TOKEN = os.getenv("TEAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Get the large docs API keys as a comma-separated string
LARGE_DOC_GEMINI_API_KEYS_STR = os.getenv("LARGE_DOC_GEMINI_API_KEYS", "")
# Split the string by comma to create a list of keys
LARGE_DOC_GEMINI_API_KEYS = [key.strip() for key in LARGE_DOC_GEMINI_API_KEYS_STR.split(',') if key.strip()]

PROCESSING_TIMEOUT = 100
LOG_FILE = "requests_log.log"

# Initialize FastAPI app
app = FastAPI(
    title="HackRX API",
    version=API_VERSION,
    openapi_url=f"{API_PREFIX}/openapi.json",
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=f"{API_PREFIX}/redoc"
)

# Create an APIRouter with the prefix
router = APIRouter(prefix=API_PREFIX)

# Create a single shared DocumentProcessor instance
processor = DocumentProcessor(GEMINI_API_KEY,LARGE_DOC_GEMINI_API_KEYS)

# Pydantic models for request and response
class SubmissionRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class SubmissionResponse(BaseModel):
    answers: List[str]

# Dependency to validate the API key
def verify_token(authorization: str = Header(...)):
    """Verifies the bearer token in the Authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")
    
    token = authorization.split(" ")[1]
    if token != TEAM_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
    
    return True

# API endpoint
@router.post("/hackrx/run", response_model=SubmissionResponse)
async def run_hackrx(request: SubmissionRequest, token_valid: bool = Depends(verify_token)):
    """
    Endpoint to run submissions, processing questions against a provided document URL.
    Falls back to placeholder answers if an error or timeout occurs.
    Saves request and response to a file before sending back.
    """
    fallback_answers = ["Cannot answer right now. Please try later..."] * len(request.questions)

    try:
        # Run the processor with a timeout
        answers = await asyncio.wait_for(
            processor.process_answers(request.documents, request.questions),
            timeout=PROCESSING_TIMEOUT
        )
    except asyncio.TimeoutError:
        answers = fallback_answers
    except Exception as e:
        answers = fallback_answers

    # Save request & answer to a file
    try:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "document": str(request.documents),
            "questions": request.questions,
            "answers": answers
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        # Don't block API response on logging failure
        pass

    return SubmissionResponse(answers=answers)

# Include the router in the app
app.include_router(router)
