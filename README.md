# 📄 Document Processing Approaches

This repository contains **two main categories** of document question-answering pipelines:

1. **Integrated API-Based Processing** — combines link-aware fetching and large-document chunking into a single FastAPI service.
2. **Local RAG-Based Processing** — uses a vector store for offline retrieval.

---

## 🚀 Integrated API-Based Processing (Default)

The API-based pipeline now merges the best parts of:

- **Link-Aware, Multi-Stage Processing** (fetches and processes document links)
- **Chunked, High-Concurrency Processing** (handles large and multi-format files efficiently)

### **Workflow**
1. **File Size & Type Detection**
   - Detects file type (PDF, DOCX, PPTX, JPEG, PNG, GIF, plain text).
   - If file is **too large** (> 20 MB), automatically falls back to chunked processing.
2. **Link-Aware Stage**
   - For smaller files, first asks Gemini whether answers can be extracted directly.
   - If the document contains links, AI generates Python fetch code to download **all URLs** from the file.
3. **Data Fetch & Merge**
   - Executes fetch code to retrieve linked data.
   - Combines document + fetched data for final answers.
4. **Large Document Handling**
   - For large or complex documents:
     - PDFs are split into ~5 MB chunks.
     - Non-PDF formats are converted to PDF where possible.
     - Multiple Gemini clients are used for **parallel API calls**.
   - Runs a **non-document fallback** call in parallel.
5. **Answer Selection**
   - Per-question confidence scoring and source type filtering (`document_specific`, `general_knowledge`, `prohibited`).

**✅ Strengths**
- Automatically detects and fetches **document links**.
- Supports large, multi-format documents with parallel processing.
- Confidence-based best-answer selection.
- Single integrated service — no need to choose between two APIs.

---

## 💡 RAG-Based Local API Server (Optional)

An alternative **offline** approach that uses a vector database for retrieval.

### **Workflow**
1. Download PDF from URL.
2. Clear the local vector store.
3. Extract and chunk text (OCR fallback for scanned PDFs).
4. Add chunks to the store.
5. For each question:
   - Retrieve the most relevant chunks.
   - Return extracted text snippets as answers (no generative reasoning).

**✅ Strengths**
- Works entirely offline once documents are downloaded.
- No API costs.
- OCR support.

**⚠ Limitations**
- Purely extractive — no reasoning or synthesis like the API-based version.

---

## 🛠 Running the Services

### Install dependencies
```bash
pip install -r requirements.txt
```

```bash
uvicorn main:app --reload
```

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```
