#To run server :  uvicorn mock_server:app --host 0.0.0.0 --port 8000 --reload

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "mock_uploads"
PARENT_UPLOAD_DIR = "mock_parent_uploads"

class BatchQueryRequest(BaseModel):
    queries: List[str]
    top_k: int = 5
    threshold: float = 0.3

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    saved_files = [f.filename for f in files]
    print("Mock uploaded:", saved_files)
    return {
        "message": f"Successfully processed {len(saved_files)} documents",
        "files": saved_files,
        "status": "success"
    }

@app.post("/query")
async def query_documents(
    query: str = Form(...),
    top_k: int = Form(5),
    threshold: float = Form(0.3),
    files: List[UploadFile] = File(None)
):
    print("Mock query:", query)
    print("Mock uploaded files:", [f.filename for f in files] if files else "No files uploaded")
    return {
        "answer": f"Mock answer for query: {query}",
        "files": [f.filename for f in files] if files else []
    }

@app.post("/batch-query")
async def batch_query_documents(request: BatchQueryRequest):
    return {
        "responses": [
            {
                "answer": f"Mock answer for query: {q}",
                "files": []
            }
            for q in request.queries
        ],
        "total_queries": len(request.queries)
    }

@app.post("/load-documents")
async def load_documents():
    return {
        "message": "Mock document loading complete.",
        "status": "success",
        "stats": {
            "total_documents": 5,
            "index_size": "2.3MB"
        }
    }