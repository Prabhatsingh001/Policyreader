import pdfplumber
from pdfminer.pdfparser import PDFSyntaxError
import re
import tiktoken
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pytesseract
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class DocumentChunk:
    content: str
    source: str
    chunk_id: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    section_level: Optional[int] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_keywords: int = 5,
        min_section_length: int = 30  # Reduced for potentially smaller sections
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_keywords = max_keywords
        self.min_section_length = min_section_length
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Hierarchical section patterns (ordered by priority)
        # Hierarchical section patterns (ordered by priority)
        self.section_patterns = [
            (r'\n(ARTICLE|SECTION)\s+[IVXLCDM]+[.:]?\s+', 1),  # Roman numerals
            (r'\n\d+\.\d+(\.\d+)?\s+[A-Z]', 2),  # 1.1, 1.1.1
            (r'\n\([a-z]\)\s+', 3),  # (a), (b)
            (r'\n\d+\.\s+[A-Z]', 2),  # 1. Section
            (r'\n[A-Z]\.\s+[A-Z]', 3),  # A. Subsection
            ]

        

    def process_directory(self, directory_path: str) -> List[DocumentChunk]:
        """Process all documents in a directory."""
        directory = Path(directory_path)
        if not directory.exists():
            logging.warning(f"Directory {directory_path} does not exist")
            return []
        
        all_chunks = []
        supported_extensions = {'.pdf', '.txt', '.doc', '.docx', '.csv', '.json'}
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    logging.info(f"Processing file: {file_path}")
                    if file_path.suffix.lower() == '.pdf':
                        chunks = self.process_pdf(str(file_path))
                    elif file_path.suffix.lower() == '.txt':
                        chunks = self.process_text_file(str(file_path))
                    elif file_path.suffix.lower() in {'.doc', '.docx'}:
                        chunks = self.process_word_document(str(file_path))
                    elif file_path.suffix.lower() == '.csv':
                        chunks = self.process_csv_file(str(file_path))
                    elif file_path.suffix.lower() == '.json':
                        chunks = self.process_json_file(str(file_path))
                    else:
                        continue
                    
                    all_chunks.extend(chunks)
                    logging.info(f"Processed {len(chunks)} chunks from {file_path.name}")
                    
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {e}")
                    continue
        
        logging.info(f"Total chunks processed: {len(all_chunks)}")
        return all_chunks

    def process_text_file(self, file_path: str) -> List[DocumentChunk]:
        """Process plain text files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple chunking for text files
            chunks = self._chunk_text(
                text=content,
                source=Path(file_path).stem,
                page_number=1,
                section="Document",
                section_level=0,
                metadata={
                    'document_title': Path(file_path).stem,
                    'section_type': 'document',
                    'pages': (1, 1),
                    'keywords': self._extract_keywords(content),
                    'source_file': file_path
                }
            )
            return chunks
        except Exception as e:
            logging.error(f"Error processing text file {file_path}: {e}")
            return []

    def process_word_document(self, file_path: str) -> List[DocumentChunk]:
        """Process Word documents (basic implementation)."""
        try:
            # For now, treat as text file - in production, use python-docx
            logging.warning(f"Word document processing not fully implemented for {file_path}")
            return self.process_text_file(file_path)
        except Exception as e:
            logging.error(f"Error processing Word document {file_path}: {e}")
            return []

    def process_csv_file(self, file_path: str) -> List[DocumentChunk]:
        """Process CSV files."""
        try:
            import csv
            content_parts = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:  # Header
                        content_parts.append(f"Headers: {', '.join(row)}")
                    else:
                        content_parts.append(f"Row {i}: {', '.join(row)}")
            
            content = "".join(content_parts)
            chunks = self._chunk_text(
                text=content,
                source=Path(file_path).stem,
                page_number=1,
                section="CSV Data",
                section_level=0,
                metadata={
                    'document_title': Path(file_path).stem,
                    'section_type': 'csv',
                    'pages': (1, 1),
                    'keywords': self._extract_keywords(content),
                    'source_file': file_path
                }
            )
            return chunks
        except Exception as e:
            logging.error(f"Error processing CSV file {file_path}: {e}")
            return []

    def process_json_file(self, file_path: str) -> List[DocumentChunk]:
        """Process JSON files."""
        try:
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert JSON to readable text
            content = json.dumps(data, indent=2)
            chunks = self._chunk_text(
                text=content,
                source=Path(file_path).stem,
                page_number=1,
                section="JSON Data",
                section_level=0,
                metadata={
                    'document_title': Path(file_path).stem,
                    'section_type': 'json',
                    'pages': (1, 1),
                    'keywords': self._extract_keywords(content),
                    'source_file': file_path
                }
            )
            return chunks
        except Exception as e:
            logging.error(f"Error processing JSON file {file_path}: {e}")
            return []

    def process_pdf(self, file_path: str) -> List[DocumentChunk]:
        """Process PDF document with enhanced text extraction and section detection"""
        chunks = []
        try:
            with pdfplumber.open(file_path) as pdf:
                full_text = ""
                char_pos_to_page = {}
                current_pos = 0
                
                # Extract text with precise page tracking
                for page_num, page in enumerate(pdf.pages, 1):
                    logging.info(f"Processing page {page_num} with pdfplumber")
                    text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
                    
                    # If no text extracted by pdfplumber, try OCR
                    if not text.strip():
                        logging.info(f"No text extracted by pdfplumber from page {page_num}, attempting OCR...")
                        try:
                            # Render page as an image
                            page_image = page.to_image(resolution=300)
                            img = page_image.original
                            
                            # Use pytesseract to extract text from image
                            text = pytesseract.image_to_string(img) or ""
                            if text.strip():
                                logging.info(f"OCR extracted {len(text)} characters from page {page_num}")
                            else:
                                logging.warning(f"OCR extracted no text from page {page_num}")
                        except Exception as ocr_e:
                            logging.error(f"OCR failed for page {page_num} in {file_path}: {ocr_e}")
                            text = "" # Ensure text is empty if OCR fails

                    text += ""  # Preserve page separation
                    full_text += text
                    
                    # Map character positions to page numbers
                    for _ in range(len(text)):
                        char_pos_to_page[current_pos] = page_num
                        current_pos += 1
                
                if not full_text.strip():
                    logging.warning(f"No text extracted from the entire PDF {file_path}")
                    return []
                    
                # Extract sections with hierarchy
                sections = self._extract_sections(full_text, char_pos_to_page)
                
                if not sections:
                    logging.warning(f"No sections detected in {file_path}, treating as a single document")
                    # Fallback: Treat the entire text as one section if no sections are found
                    start_page = char_pos_to_page.get(0, 1)
                    end_page = char_pos_to_page.get(len(full_text) - 1, start_page)
                    sections = [{
                        'id': 0,
                        'title': 'Document',
                        'content': full_text,
                        'level': 0,
                        'start_page': start_page,
                        'end_page': end_page
                    }]
                
                logging.info(f"Extracted {len(sections)} sections from {file_path}")

                # Process each section
                for section in sections:
                    logging.info(f"Processing section: {section['title']} (Level {section['level']})")
                    if len(section['content'].strip()) < self.min_section_length:
                        logging.warning(f"Section '{section['title']}' is too short ({len(section['content'].strip())} chars), skipping.")
                        continue
                        
                    section_chunks = self._chunk_text(
                        text=section['content'],
                        source=f"{Path(file_path).stem}_section_{section['id']}",
                        page_number=section['start_page'],
                        section=section['title'],
                        section_level=section['level'],
                        metadata={
                            'document_title': Path(file_path).stem,
                            'section_type': self._classify_section(section['title']),
                            'pages': (section['start_page'], section['end_page']),
                            'keywords': self._extract_keywords(section['content']),
                            'source_file': file_path
                        }
                    )
                    chunks.extend(section_chunks)
                    logging.info(f"Generated {len(section_chunks)} chunks for section '{section['title']}'")
                    
        except pdfplumber.utils.exceptions.PdfminerException as e:
            logging.error(f"Malformed PDF structure in {file_path}: {e}")
        except Exception as e:
            logging.exception(f"Critical failure processing {file_path}: {e}")
        
        return chunks

    def _extract_sections(
        self, 
        text: str, 
        char_pos_to_page: Dict[int, int]
    ) -> List[Dict[str, Any]]:
        """Hierarchical section detection with precise page tracking"""
        sections = []
        last_end = 0
        section_id = 0
        
        # Find all potential section headers
        matches = []
        for pattern, level in self.section_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                start, end = match.span()
                matches.append({
                    'start': start,
                    'end': end,
                    'title': text[start:end].strip(),
                    'level': level
                })
        
        # Sort matches by position
        matches.sort(key=lambda x: x['start'])
        
        # Create sections from matches
        if not matches:
            # Fallback: Entire document as one section
            start_page = char_pos_to_page.get(0, 1)
            end_page = char_pos_to_page.get(len(text) - 1, start_page)
            return [{
                'id': 0,
                'title': 'Document',
                'content': text,
                'level': 0,
                'start_page': start_page,
                'end_page': end_page
            }]
        
        # Process text before the first section as an introductory section
        if matches[0]['start'] > 0:
             intro_content = text[:matches[0]['start']].strip()
             if intro_content:
                start_page = char_pos_to_page.get(0, 1)
                end_page = char_pos_to_page.get(matches[0]['start'] - 1, start_page)
                sections.append({
                    'id': section_id,
                    'title': 'Introduction',
                    'content': intro_content,
                    'level': 0,
                    'start_page': start_page,
                    'end_page': end_page
                })
                section_id += 1
                last_end = matches[0]['start']
        
        # Process each match
        for i, match in enumerate(matches):
            # Add content between current match and the next match (or end of text)
            section_start = match['end']
            section_end = matches[i+1]['start'] if i+1 < len(matches) else len(text)
            content = text[section_start:section_end].strip()
            
            # Only create a section if there is content after the header
            if content:
                 # Calculate page range
                start_page = char_pos_to_page.get(match['start'], 1)
                end_page = char_pos_to_page.get(section_end - 1, start_page)
                
                # Create section
                sections.append({
                    'id': section_id,
                    'title': match['title'],
                    'content': content,
                    'level': match['level'],
                    'start_page': start_page,
                    'end_page': end_page
                })
                section_id += 1
            last_end = section_end
        
        return sections

    def _chunk_text(
        self,
        text: str,
        source: str,
        page_number: int,
        section: str,
        section_level: int,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Sentence-aware chunking with token limits"""
        # Split into sentences while preserving delimiters
        sentences = re.split(r'(?<!w.w.)(?<![A-Z][a-z].)(?<=[.?!])s+', text) # Improved sentence splitting
        chunks = []
        current_chunk = []
        current_token_count = 0
        chunk_id_base = hashlib.md5((source + section).encode()).hexdigest()[:8] # Include section in ID base
        
        for i, sentence in enumerate(sentences):
            # Handle encoding issues
            try:
                sentence_tokens = self.tokenizer.encode(sentence)
            except UnicodeEncodeError:
                sentence = sentence.encode('utf-8', 'ignore').decode()
                sentence_tokens = self.tokenizer.encode(sentence)
                
            token_count = len(sentence_tokens)
            
            # Check if sentence fits in current chunk
            if current_token_count + token_count > self.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_content = " ".join(current_chunk)
                chunk_id = f"{chunk_id_base}-{len(chunks)}"
                chunks.append(DocumentChunk(
                    content=chunk_content,
                    source=source,
                    chunk_id=chunk_id,
                    page_number=page_number,
                    section=section,
                    section_level=section_level,
                    timestamp=datetime.now(),
                    metadata=metadata.copy()
                ))
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - int(self.chunk_overlap / 10)) # Approximate sentence overlap
                current_chunk = current_chunk[overlap_start:]
                current_token_count = sum(len(self.tokenizer.encode(s)) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_token_count += token_count
        
        # Add final chunk
        if current_chunk:
            chunk_content = " ".join(current_chunk)
            chunk_id = f"{chunk_id_base}-{len(chunks)}"
            chunks.append(DocumentChunk(
                content=chunk_content,
                source=source,
                chunk_id=chunk_id,
                page_number=page_number,
                section=section,
                section_level=section_level,
                timestamp=datetime.now(),
                metadata=metadata.copy()
            ))
        
        return chunks

    def _extract_keywords(self, content: str) -> List[str]:
        """Dynamic keyword extraction using TF-IDF"""
        try:
            # Handle small text inputs
            if len(content.split()) < 5:
                return []
                
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=50
            )
            X = vectorizer.fit_transform([content])
            feature_array = np.array(vectorizer.get_feature_names_out())
            # Convert sparse matrix to dense array and get indices of top features
            tfidf_sorting = np.argsort(X.toarray().flatten())[::-1]
            
            # Get top keywords, but don't exceed the number of available features
            top_keywords = feature_array[tfidf_sorting][:min(self.max_keywords, len(feature_array))]
            return top_keywords.tolist()
        except Exception as e:
            logging.warning(f"Keyword extraction failed: {e}")
            # Fallback to static keywords
            static_keywords = ['definition', 'policy', 'coverage', 'exclusion', 'claim']
            return [kw for kw in static_keywords if kw in content.lower()]

    def _classify_section(self, title: str) -> str:
        """Categorize section based on title content"""
        title_lower = title.lower()
        if 'article' in title_lower:
            return 'article'
        elif 'section' in title_lower:
            return 'section'
        elif 'definition' in title_lower:
            return 'definitions'
        elif 'exhibit' in title_lower:
            return 'exhibit'
        elif 'schedule' in title_lower:
            return 'schedule'
        elif any(term in title_lower for term in ['term', 'termination']):
            return 'term'
        return 'other'

# Usage Example
if __name__ == "__main__":
    processor = DocumentProcessor()
    # Replace with a test PDF file path
    test_pdf_path = r"D:\Policyreader\llm\sample_docs\ICIHLIP22012V012223.pdf"  # Using raw string for Windows path
    if Path(test_pdf_path).exists():
        chunks = processor.process_pdf(test_pdf_path)
        
        print(f"Generated {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3]):
            print(f"Chunk {i+1}:")
            print(f"ID: {chunk.chunk_id}")
            print(f"Section: {chunk.section} (Level {chunk.section_level})")
            print(f"Pages: {chunk.metadata.get('pages', 'N/A') if hasattr(chunk, 'metadata') and chunk.metadata else 'N/A'}")
            print(f"Keywords: {', '.join(chunk.metadata.get('keywords', [])) if hasattr(chunk, 'metadata') and chunk.metadata else 'N/A'}")
            print(f"Content: {chunk.content[:200] if hasattr(chunk, 'content') else 'No content'}...")
    else:
        print(f"Test PDF file not found at {test_pdf_path}")
