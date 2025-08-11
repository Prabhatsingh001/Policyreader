import httpx
import logging
import os
from typing import List, Optional
from pydantic import BaseModel, HttpUrl, RootModel
from google import genai
from google.genai import types
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import asyncio
import json
from document_processor_2 import process_document_questions

# --- Logging setup ---
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")


async def detect_file_type(url: str) -> str:
    logging.debug(f"Detecting file type for URL: {url}")
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        file_start = await resp.aread()
        file_start = file_start[:8]

    if file_start.startswith(b"%PDF-"):
        return "pdf"
    elif file_start.startswith(b"\xFF\xD8\xFF"):
        return "jpeg"
    elif file_start.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    elif file_start.startswith(b"GIF87a") or file_start.startswith(b"GIF89a"):
        return "gif"
    else:
        return "unknown"


def text_to_pdf_bytes(text: str) -> bytes:
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


class ProcessingResult(BaseModel):
    needs_fetching: bool
    links: Optional[List[str]] = None
    answers: Optional[List[str]] = None


class AnswerList(RootModel[List[str]]):
    pass


class DocumentProcessor:
    def __init__(self, api_key: str, key_list):
        self.GEMINI_MODEL = "gemini-2.5-flash"
        self.GEMINI_FILE_LIMIT_MB = 20
        self.client = genai.Client(api_key=api_key)
        self.key_list = [genai.Client(api_key=i) for i in key_list]

    async def process_answers(self, document_url: HttpUrl, questions: List[str]) -> List[str]:
        try:
            logging.info(f"=== Starting processing for document: {document_url} ===")
            async with httpx.AsyncClient() as client:
                # Step 0: File size
                head_resp = await client.head(str(document_url), follow_redirects=True)
                head_resp.raise_for_status()
                file_size_bytes = int(head_resp.headers.get("Content-Length", 0))
                file_size_mb = file_size_bytes / (1024 * 1024)
                logging.info(f"File size: {file_size_mb:.2f} MB")

                questions_str = "\n".join(f"- {q}" for q in questions)

                if file_size_mb > self.GEMINI_FILE_LIMIT_MB:
                    return await process_document_questions(str(document_url), questions, self.key_list, self.client)
                else:
                    file_type = await detect_file_type(str(document_url))
                    logging.info(f"File type detected: {file_type}")
                    if file_type in ["jpeg", "png", "gif"]:
                        resp = await client.get(str(document_url), follow_redirects=True)
                        file_bytes = await resp.aread()
                        content_part = types.Part.from_bytes(data=file_bytes, mime_type=f"image/{file_type}")
                    elif file_type == "pdf":
                        resp = await client.get(str(document_url), follow_redirects=True)
                        file_bytes = await resp.aread()
                        content_part = types.Part.from_bytes(data=file_bytes, mime_type="application/pdf")
                    else:
                        resp = await client.get(str(document_url), follow_redirects=True)
                        file_text = resp.text
                        pdf_bytes = text_to_pdf_bytes(file_text)
                        content_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

            # Stage 1 — Ask AI if fetching is needed, and only return links
            logging.info("=== Stage 1: Asking AI if fetching is needed ===")
            stage1_prompt = f"""
You are processing a document and answering questions.
If all answers can be obtained directly from the given document, return them in "answers".

If the document contains one or more HTTP or HTTPS links needed to answer the questions,
return needs_fetching=true and provide all the links in document in a JSON list called "links".

Do not include any code, explanations, or duplicates — just the links in the order they appear.

Questions:
{questions_str}
"""
            response1 = self.client.models.generate_content(
                model=self.GEMINI_MODEL,
                contents=[content_part, stage1_prompt] if file_size_mb <= self.GEMINI_FILE_LIMIT_MB else stage1_prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    response_mime_type="application/json",
                    response_schema=ProcessingResult,
                )
            )

            logging.debug("AI raw Stage 1 JSON response:")
            logging.debug(response1.text)

            stage1_result = ProcessingResult.model_validate_json(response1.text)
            logging.info(f"AI says: needs_fetching={stage1_result.needs_fetching}")

            if not stage1_result.needs_fetching:
                logging.info("AI provided final answers in Stage 1, no fetch needed.")
                logging.debug(f"Answers: {stage1_result.answers}")
                return stage1_result.answers or []

            # Stage 2 — Fetch links ourselves
            fetched_data = []
            if stage1_result.links:
                logging.info(f"Fetching {len(stage1_result.links)} link(s) from document...")
                async with httpx.AsyncClient() as client:
                    for link in stage1_result.links:
                        try:
                            resp = await client.get(link, follow_redirects=True, timeout=30)
                            content_type = resp.headers.get("Content-Type", "")
                            if "application/json" in content_type:
                                try:
                                    data = resp.json()
                                except Exception:
                                    data = resp.text
                            else:
                                data = resp.text
                            fetched_data.append((link, data))
                        except Exception as e:
                            logging.error(f"Error fetching {link}: {e}")
                            fetched_data.append((link, f"Error: {e}"))
            else:
                logging.warning("AI said fetching is needed but provided no links.")
                fetched_data = []

            # Stage 3 — Ask AI final answers using document + fetched data
            logging.info("=== Stage 3: Sending fetched data to AI for final answers ===")
            stage2_prompt = f"""
You are given fetched data from the document's referenced link(s) and also the document.
Now answer the questions based on both sources.

Questions:
{questions_str}

Fetched data:
{json.dumps(fetched_data, ensure_ascii=False, indent=2)}
"""
            response2 = self.client.models.generate_content(
                model=self.GEMINI_MODEL,
                contents=[content_part, stage2_prompt],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    response_mime_type="application/json",
                    response_schema=AnswerList,
                )
            )

            logging.debug("AI raw Stage 3 JSON response:")
            logging.debug(response2.text)

            final_answers = AnswerList.model_validate_json(response2.text).root
            logging.info("AI provided final answers after seeing fetched data.")
            logging.debug(f"Final answers: {final_answers}")

            return final_answers

        except httpx.RequestError as e:
            logging.error(f"Request error: {e}")
            return [f"Error: Unable to access document. Details: {e}"]
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return [f"Error: Unexpected error. Details: {e}"]
