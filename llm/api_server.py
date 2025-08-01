from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import tempfile
import requests
import os
import logging
from rag_system import IntelligentQuerySystem
from document_processor import DocumentProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize the FastAPI app
app = FastAPI(title="HackRx API", version="1.0")

# Load system
query_system = IntelligentQuerySystem()

# Request and response models
class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponseModel(BaseModel):
    answers: List[str]

# Team Token for validation
TEAM_TOKEN = "bcea5c269cdc675080afa5ff5f2a475d0a29f5b845d4c8db5c142c0be4267fe7"

@app.post("/api/v1/hackrx/run", response_model=QueryResponseModel)
async def run_submission(
    request: Request,
    payload: QueryRequest,
    authorization: Optional[str] = Header(None)
):
    # Validate Bearer token
    if authorization != f"Bearer {TEAM_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    tmp_pdf_path = None
    try:
        # Download the document
        logging.info(f"Downloading document from: {payload.documents}")
        response = requests.get(str(payload.documents))
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download document")

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_pdf_path = tmp_file.name

        # Ensure the file is fully written and closed
        # The file is automatically closed when exiting the context manager

        # Clear old documents if needed
        query_system.vector_store.clear()
        logging.info("Cleared vector store")

        # Create a new document processor with lower min_section_length
        # This will allow processing of smaller sections
        doc_processor = DocumentProcessor(min_section_length=10)
        
        # Process the PDF with a lower threshold for better matching
        logging.info("Processing the downloaded document...")
        chunks = doc_processor.process_pdf(tmp_pdf_path)
        logging.info(f"Processed {len(chunks)} chunks from PDF")
        
        if not chunks:
            raise HTTPException(status_code=422, detail="No valid content found in PDF")

        query_system.vector_store.add_documents(chunks)
        logging.info(f"{len(chunks)} chunks added to vector store")
        
        # Check vector store stats
        stats = query_system.vector_store.get_statistics()
        logging.info(f"Vector store stats: {stats}")

        # Run queries with a lower threshold to improve matching
        logging.info("Running user queries...")
        answers = []
        for question in payload.questions:
            # Use a lower threshold to improve matching
            result = query_system.query(question, threshold=0.01)
            logging.info(f"Query: {question}")
            logging.info(f"Answer: {result.answer}")
            logging.info(f"Confidence: {result.confidence}")
            answers.append(result.answer)

        return QueryResponseModel(answers=answers)

    except Exception as e:
        logging.exception("Error during processing")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temp file
        if tmp_pdf_path and os.path.exists(tmp_pdf_path):
            try:
                os.unlink(tmp_pdf_path)
            except Exception as e:
                logging.warning(f"Failed to delete temporary file: {e}")