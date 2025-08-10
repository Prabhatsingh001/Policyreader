# 📄 Document Processing Approaches

This repository demonstrates **three distinct approaches** for processing documents — optimized for:
- Handling links inside documents
- Processing very large files efficiently
- Running with a local vector search engine (RAG)

---

## 🚀 API-Based Approaches

These methods run as FastAPI services and use Gemini for question answering.

---

### 1️⃣ Link-Aware, Multi-Stage Processing
**Files:**  
- `document_processor.py` (link-aware version)  
- `main.py` (first version)  

**Workflow:**
1. **File Size Check** → If too large, fall back to general knowledge.  
2. **File Type Detection** → Identify PDF, image, GIF, or text.  
3. **Stage 1** – Ask Gemini whether answers can be extracted directly from the document.  
   - If **no fetching needed** → Return direct answers.  
   - If **fetching needed** → AI generates Python code to download all URLs found in the document.  
4. **Stage 2** – Execute AI-provided fetch code to get link data.  
5. **Stage 3** – Send fetched data + original document to Gemini for final answers.

**Strengths:**
- Automatically detects and fetches **all document links**.
- Works for PDFs, images, plain text, and HTML.

**Limitations:**
- Less efficient for **very large documents**.
- Single-threaded.

---

### 2️⃣ Chunked, Multi-Format, High-Concurrency Processing
**Files:**  
- `main.py` (second version)

**Workflow:**
1. **Download & Detect Type** → Supports PDF, DOCX, PPTX, JPEG, PNG.  
2. **Format-Specific Processing**:
   - **PDF** → Split into ~5MB chunks.
   - **DOCX/PPTX** → Extract all text.
   - **Images** → Send directly to Gemini.  
3. **Concurrent API Calls** → Parallel Gemini requests for faster results.  
4. **Parallel Fallback** → Non-document (general knowledge) call runs in parallel.  
5. **Answer Selection** → Choose per-question answers based on confidence and `source_type`.

**Strengths:**
- Handles **large, multi-format documents** efficiently.
- Parallel execution with multiple API keys.
- Confidence-based result selection.

**Limitations:**
- Does not auto-fetch document links.
- Slightly more setup complexity.

---

## 💡 RAG-Based Local API Server (No External AI Calls)
**Files:**  
- `api_server.py` (RAG server)  
- `document_processor.py` (local chunking + OCR support)  

**Workflow:**
1. **Download PDF** from the given URL.  
2. **Clear Vector Store** to reset context.  
3. **Chunking**:
   - Extracts text from PDF (with OCR fallback).
   - Splits into small, token-limited chunks.
4. **Add Chunks to Vector Store** (`IntelligentQuerySystem` handles retrieval).  
5. **Query Handling**:
   - Each question is matched against stored chunks using vector similarity.
   - Returns the most relevant answer based on similarity threshold.

**Strengths:**
- Fully **local** — no AI API costs.
- Works offline once documents are downloaded.
- Fine-grained chunking with OCR for scanned PDFs.

**Limitations:**
- Only as good as the vector search quality.
- No generative reasoning — purely extractive.

---

## 🛠 How to Run

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run API-Based (Link-Aware or Large-Doc):
```bash
# Link-aware
uvicorn main:app --reload  

# Large-doc
uvicorn main:app --reload  
```
*(Switch `main.py` to the desired version before running.)*

### Run RAG-Based Local Server:
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📂 Project Structure
```
document_processor.py       # Link-aware or local chunking logic
main.py (v1)                # API link-aware pipeline
main.py (v2)                # API large-doc concurrent pipeline
api_server.py               # RAG-based local document query API
```
