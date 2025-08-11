import itertools
import asyncio
import json
import httpx
from datetime import datetime
from io import BytesIO
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# -------- Models --------
class AnswerWithConfidence(BaseModel):
    answer: str
    confidence: int = Field(..., ge=0, le=100)
    source_type: str = Field(..., description="One of 'document_specific', 'general_knowledge', or 'prohibited'")

class InternalGeminiResponse(BaseModel):
    answers: List[AnswerWithConfidence]

class HackRxResponse(BaseModel):
    answers: List[str]

# -------- Configuration (for the function) --------
TIMEOUT_SECONDS = 55
FALLBACK_ANSWER = "I'm sorry, I can't answer that question."
MAX_SPLIT_SIZE_MB = 5
HTTP_CLIENT = httpx.AsyncClient()

# -------- PDF Splitting Helper --------
async def _split_pdf_by_size(pdf_bytes: bytes) -> List[bytes]:
    """Splits a PDF into smaller chunks of approximately 5MB."""
    pdf_reader = PdfReader(BytesIO(pdf_bytes))
    total_pages = len(pdf_reader.pages)
    
    # Calculate average page size to estimate how many pages to put in a chunk
    estimated_page_size = len(pdf_bytes) / total_pages if total_pages > 0 else 1
    pages_per_chunk = int((MAX_SPLIT_SIZE_MB * 1024 * 1024) / estimated_page_size)
    if pages_per_chunk == 0:
        pages_per_chunk = 1
    
    split_pdfs = []
    for start_page in range(0, total_pages, pages_per_chunk):
        pdf_writer = PdfWriter()
        end_page = min(start_page + pages_per_chunk, total_pages)
        for page_num in range(start_page, end_page):
            pdf_writer.add_page(pdf_reader.pages[page_num])
        
        output_stream = BytesIO()
        pdf_writer.write(output_stream)
        split_pdfs.append(output_stream.getvalue())

    logging.info(f"📄 PDF split into {len(split_pdfs)} parts.")
    return split_pdfs

# -------- PDF Conversion Helper --------
def text_to_pdf_bytes(text: str) -> bytes:
    """Converts a string of text into a PDF in bytes."""
    logging.debug("Converting text to PDF bytes...")
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    text_object = c.beginText(40, height - 50)
    text_object.setFont("Helvetica", 12)

    max_chars_per_line = 90
    for line in text.splitlines():
        while len(line) > max_chars_per_line:
            text_object.textLine(line[:max_chars_per_line])
            line = line[max_chars_per_line:]
        text_object.textLine(line)
    c.drawText(text_object)
    c.showPage()
    c.save()
    return buffer.getvalue()

# -------- Helper to call Gemini (with document) --------
async def _call_gemini_with_document(client: genai.Client, pdf_bytes: bytes, questions: List[str]) -> Optional[List[AnswerWithConfidence]]:
    """Calls Gemini with a PDF document and a list of questions."""
    instruction = (
        "You are an expert insurance policy assistant. Your primary function is to analyze the provided PDF document and answer a list of user questions. "
        "For each question, you must provide a JSON object with three fields: 'answer', 'confidence' (0-100), and 'source_type' ('document_specific', 'general_knowledge', or 'prohibited'). "
        "Use 'document_specific' if the answer is in the PDF, with high confidence. "
        "Use 'general_knowledge' if the question is not in the PDF but can be answered from your knowledge; confidence must be 0. "
        "Use 'prohibited' for illegal/unethical questions; the answer must be 'I cannot assist with this request as it involves illegal or prohibited actions.' and confidence must be 0."
    )

    contents = [
        types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
        instruction,
        f"Questions: {json.dumps(questions)}"
    ]

    def sync_generate():
        return client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=InternalGeminiResponse,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        )

    try:
        response = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(None, sync_generate),
            timeout=TIMEOUT_SECONDS
        )
    except Exception as e:
        logging.error(f"❌ Gemini call with document failed or timed out: {e}")
        return None

    try:
        parsed: InternalGeminiResponse = response.parsed
        if not isinstance(parsed, InternalGeminiResponse) or len(parsed.answers) != len(questions):
            raise ValueError("Invalid or mismatched response schema")
        return parsed.answers
    except Exception as e:
        logging.error(f"❌ Failed to parse Gemini response with document: {e}")
        return None

# -------- Helper to call Gemini (without document) --------
async def _call_gemini_without_document(client: genai.Client, questions: List[str]) -> Optional[List[str]]:
    """A direct Gemini call with just the questions."""
    instruction = (
        "You are an expert assistant. Answer each question in order, under 50 words, using your general knowledge. "
        "Return ONLY a JSON object: {\"answers\": [\"answer1\", \"answer2\", ...]}."
    )
    
    contents = [
        instruction,
        f"Questions: {json.dumps(questions)}"
    ]

    def sync_generate():
        return client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=HackRxResponse,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        )

    try:
        response = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(None, sync_generate),
            timeout=TIMEOUT_SECONDS
        )
        parsed: HackRxResponse = response.parsed
        return parsed.answers
    except Exception as e:
        logging.error(f"❌ Gemini call without document failed or timed out: {e}")
        return None

# -------- Main Processing Function --------
async def process_document_questions(
    document_url: str,
    questions: List[str],
    gemini_clients: List[genai.Client],
    non_doc_client: genai.Client
) -> List[str]:
    """
    Processes a list of questions against a document from a URL.
    
    This function handles both PDF and non-PDF documents by converting non-PDFs to PDF first.
    It then splits the PDF, calls Gemini concurrently on the parts, and selects the best
    answers based on confidence scores and source type. It also runs a fallback non-document
    call in parallel to handle cases where the document is unreadable or the primary
    document-based calls fail.

    Args:
        document_url: The URL of the document to be processed.
        questions: A list of questions to answer.
        gemini_clients: A list of Gemini clients to use for concurrent API calls.
        non_doc_client: A single Gemini client for the non-document fallback call.

    Returns:
        A list of strings, where each string is the answer to the corresponding question.
    """
    logging.info(f"🚀 Starting processing for URL: {document_url}")
    
    # Start the non-document call immediately as a fallback
    non_document_answers_task = asyncio.create_task(_call_gemini_without_document(non_doc_client, questions))
    
    pdf_bytes = None
    try:
        r = await HTTP_CLIENT.get(document_url, timeout=10.0)
        r.raise_for_status()
        
        content_type = r.headers.get("Content-Type", "")
        if "pdf" in content_type:
            pdf_bytes = r.content
            logging.info("✅ PDF downloaded successfully.")
        else:
            logging.info("📄 Document is not a PDF, converting to PDF...")
            text_content = r.text
            pdf_bytes = text_to_pdf_bytes(text_content)
            logging.info("✅ Document converted to PDF successfully.")
            
    except Exception as e:
        logging.error(f"❌ Failed to download or convert document: {e}")
        
    # If downloading/conversion failed, wait for the non-document call and return its result.
    if pdf_bytes is None:
        try:
            fallback_answers = await asyncio.wait_for(non_document_answers_task, timeout=5)
            logging.info("Returning answers from non-document call due to primary document failure.")
            return fallback_answers if fallback_answers is not None else [FALLBACK_ANSWER for _ in questions]
        except Exception:
            logging.error("Non-document fallback also failed.")
            return [FALLBACK_ANSWER for _ in questions]

    # Split the PDF
    split_pdfs = await _split_pdf_by_size(pdf_bytes)
    
    # Use a generator to cycle through the provided clients
    client_cycle = itertools.cycle(gemini_clients)
    
    # Truncate splits if they exceed the number of available API keys
    num_clients = len(gemini_clients)
    if len(split_pdfs) > num_clients:
        split_pdfs = split_pdfs[:num_clients]
        logging.warning(f"⚠️ Truncated PDF splits to match the number of clients ({num_clients}).")
    
    # Concurrently call Gemini for each split
    logging.info(f"✨ Making {len(split_pdfs)} concurrent API calls.")
    tasks = []
    for i, pdf_part in enumerate(split_pdfs):
        client = next(client_cycle)
        tasks.append(_call_gemini_with_document(client, pdf_part, questions))
    
    all_responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Await the non-document call if it hasn't finished already
    non_document_answers = await non_document_answers_task
    
    logging.info("✅ All API calls completed (or failed).")

    # Select the best answers
    final_answers = []
    num_questions = len(questions)
    
    for i in range(num_questions):
        prohibited_keywords = ["fraudulent", "illegal", "forged", "manipulate"]
        q_lower = questions[i].lower()
        if any(word in q_lower for word in prohibited_keywords):
            final_answers.append("I cannot assist with this request as it involves illegal or prohibited actions.")
            continue

        best_answer = FALLBACK_ANSWER
        highest_confidence = -1
        doc_specific_count = 0
        general_knowledge_count = 0
        
        for response in all_responses:
            if isinstance(response, Exception) or response is None:
                continue
            
            answer_item = response[i]
            
            if answer_item.source_type == "document_specific":
                doc_specific_count += 1
                if answer_item.confidence > highest_confidence:
                    highest_confidence = answer_item.confidence
                    best_answer = answer_item.answer
            elif answer_item.source_type == "general_knowledge":
                general_knowledge_count += 1
        
        if doc_specific_count > general_knowledge_count and highest_confidence > 0:
            final_answers.append(best_answer)
        elif non_document_answers and i < len(non_document_answers):
            final_answers.append(non_document_answers[i])
        else:
            final_answers.append(FALLBACK_ANSWER)

    logging.info("✅ Final answers selected based on confidence scores and source type.")
    
    return final_answers

