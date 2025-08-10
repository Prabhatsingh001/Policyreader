import requests
import logging
from typing import List, Optional
from pydantic import BaseModel, HttpUrl, RootModel
from google import genai
from google.genai import types
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import subprocess
import tempfile
import json

# --- Logging setup ---
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")


def detect_file_type(url: str) -> str:
    logging.debug(f"Detecting file type for URL: {url}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    file_start = resp.raw.read(8)

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
    fetch_code: Optional[str] = None
    answers: Optional[List[str]] = None


class AnswerList(RootModel[List[str]]):
    pass


class DocumentProcessor:
    def __init__(self, api_key: str):
        self.GEMINI_MODEL = "gemini-2.5-flash"
        self.GEMINI_FILE_LIMIT_MB = 20
        self.client = genai.Client(api_key=api_key)

    def process_answers(self, document_url: HttpUrl, questions: List[str]) -> List[str]:
        try:
            logging.info(f"=== Starting processing for document: {document_url} ===")

            # Step 0: File size
            head_resp = requests.head(document_url, allow_redirects=True)
            head_resp.raise_for_status()
            file_size_bytes = int(head_resp.headers.get("Content-Length", 0))
            file_size_mb = file_size_bytes / (1024 * 1024)
            logging.info(f"File size: {file_size_mb:.2f} MB")

            questions_str = "\n".join(f"- {q}" for q in questions)

            if file_size_mb > self.GEMINI_FILE_LIMIT_MB:
                logging.warning(f"File too large, skipping document parsing.")
                content_part = f"Document too large ({file_size_mb:.2f} MB). Using general knowledge."
            else:
                file_type = detect_file_type(document_url)
                logging.info(f"File type detected: {file_type}")
                if file_type in ["jpeg", "png", "gif"]:
                    file_bytes = requests.get(document_url).content
                    content_part = types.Part.from_bytes(data=file_bytes, mime_type=f"image/{file_type}")
                elif file_type == "pdf":
                    file_bytes = requests.get(document_url).content
                    content_part = types.Part.from_bytes(data=file_bytes, mime_type="application/pdf")
                else:
                    file_text = requests.get(document_url).text
                    pdf_bytes = text_to_pdf_bytes(file_text)
                    content_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

            # Stage 1
            logging.info("=== Stage 1: Asking AI if fetching is needed ===")
            stage1_prompt = f"""
You are processing a document and answering questions. 
If all answers can be obtained directly from the given document, return them in "answers". The answers should be to the point covering all aspects of the question. Or if the document contains a link or multiple links, 
return needs_fetching=True and provide Python code in fetch_code like this for **all links present in the document**
List every single HTTP/HTTPS link in the document — even if they are conditional or depend on prior results. Do not filter, deduplicate, or decide execution order. Your only job is to output all links exactly as they appear, in the same order as in the document.

import requests
url1 = ""
url2 = ""
url3 = ""
response1 = requests.get(url1)
response2 = requests.get(url2)
response3 = requests.get(url3)
print([(url1,response1.json()),(url2,response2.json()),(url3,response3.json())])

Don't try to access anything inside the the responses like response.text,etc

 .
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

            logging.info("AI says fetching is needed. Extracting fetch_code...")
            logging.debug(f"Fetch code provided by AI:\n{stage1_result.fetch_code}")

            # Stage 2
            logging.info("=== Stage 2: Running AI-provided fetch code ===")
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmpfile:
                tmpfile.write(stage1_result.fetch_code)
                tmpfile.flush()
                try:
                    fetch_output = subprocess.check_output(["python", tmpfile.name], text=True, timeout=100)
                    logging.debug("=== Fetch code output (to send back to AI) ===")
                    logging.debug(fetch_output)
                except subprocess.CalledProcessError as e:
                    logging.error(f"Error running fetch code: {e}")
                    return [f"Error executing fetch code: {e}"]

            # Stage 3
            logging.info("=== Stage 3: Sending fetched data to AI for final answers ===")
            stage2_prompt = f"""
You are given the fetched data from the document's referenced link and also the document. Check documents content properly then answer.
Now answer the questions based on this fetched data:
Questions:
{questions_str}

Fetched data:
{fetch_output}
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

        except requests.exceptions.RequestException as e:
            logging.error(f"Request error: {e}")
            return [f"Error: Unable to access document. Details: {e}"]
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return [f"Error: Unexpected error. Details: {e}"]
