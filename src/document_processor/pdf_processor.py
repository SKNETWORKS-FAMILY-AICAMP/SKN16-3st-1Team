from abc import ABC, abstractmethod
import time
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from .core import DocumentProcessor, ProcessingResult, DocumentChunk

# Import for PDF processing
try:
    import PyPDF2
    from PyPDF2 import PdfReader
except ImportError:
    PyPDF2 = None
    PdfReader = None

# Alternative PDF processing library
try:
    import PyMuPDF as fitz
except ImportError:
    fitz = None

class PDFProcessor(DocumentProcessor):
    """PDF document processor"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        if PyPDF2 is None and fitz is None:
            raise ImportError("Either PyPDF2 or PyMuPDF is required for PDF processing")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_pymupdf = fitz is not None  # Prefer PyMuPDF if available

    def get_supported_formats(self) -> List[str]:
        """Get supported PDF formats"""
        return ['.pdf']

    def validate_file(self, file_path: str) -> bool:
        """Validate if PDF file can be processed"""
        try:
            self._validate_file_exists(file_path)

            # Check file extension
            if not file_path.lower().endswith('.pdf'):
                return False

            # Try to open the PDF to verify it's valid
            if self.use_pymupdf:
                try:
                    doc = fitz.open(file_path)
                    page_count = len(doc)
                    doc.close()
                    return page_count > 0
                except:
                    return False
            else:
                try:
                    with open(file_path, 'rb') as file:
                        reader = PdfReader(file)
                        return len(reader.pages) > 0
                except:
                    return False
        except:
            return False
    
    def process_file(self, file_path: str) -> ProcessingResult:
        """Process PDF file and extract text content"""
        start_time = time.time()

        try:
            # Validate file first
            if not self.validate_file(file_path):
                return ProcessingResult(
                    success=False,
                    chunks=[],
                    error_message="Invalid PDF file or file does not exist",
                    processing_time=time.time() - start_time,
                    file_path=file_path
                )

            # Get file info
            file_size = os.path.getsize(file_path)
            file_name = Path(file_path).name

            # Extract text using preferred library
            if self.use_pymupdf:
                text_content, total_pages = self._extract_text_pymupdf(file_path)
            else:
                text_content, total_pages = self._extract_text_pypdf2(file_path)

            if not text_content:
                return ProcessingResult(
                    success=False,
                    chunks=[],
                    error_message="No text content found in PDF",
                    processing_time=time.time() - start_time,
                    file_path=file_path,
                    file_size=file_size,
                    total_pages=total_pages
                )

            # Create chunks from extracted text
            chunks = self._create_chunks(text_content, file_path, file_name)

            processing_time = time.time() - start_time

            return ProcessingResult(
                success=True,
                chunks=chunks,
                error_message=None,
                processing_time=processing_time,
                file_path=file_path,
                file_size=file_size,
                total_pages=total_pages
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                chunks=[],
                error_message=f"Error processing PDF: {str(e)}",
                processing_time=time.time() - start_time,
                file_path=file_path
            )

    def _extract_text_pymupdf(self, file_path: str) -> tuple[str, int]:
        """Extract text using PyMuPDF (more reliable)"""
        full_text = ""
        doc = fitz.open(file_path)

        try:
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                if text.strip():  # Only add non-empty pages
                    full_text += f"\n\n--- Page {page_num} ---\n\n{text}"

            return full_text.strip(), len(doc)
        finally:
            doc.close()

    def _extract_text_pypdf2(self, file_path: str) -> tuple[str, int]:
        """Extract text using PyPDF2 (fallback)"""
        full_text = ""

        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            total_pages = len(reader.pages)

            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    full_text += f"\n\n--- Page {page_num} ---\n\n{text}"

        return full_text.strip(), total_pages

    def _create_chunks(self, text: str, file_path: str, file_name: str) -> List[DocumentChunk]:
        """Create document chunks from extracted text"""
        chunks = []

        # Split by pages first if page markers exist
        if "--- Page" in text:
            page_sections = text.split("--- Page")

            for i, section in enumerate(page_sections):
                if not section.strip():
                    continue

                # Extract page number from section header
                lines = section.split('\n')
                page_num = i  # fallback
                clean_text = section

                if lines and "---" in lines[0]:
                    try:
                        page_num = int(lines[0].split()[0]) if lines[0].split() else i
                        # Remove the page header line
                        clean_text = '\n'.join(lines[1:]).strip()
                    except (ValueError, IndexError):
                        page_num = i
                        clean_text = section.strip()

                # Further chunk the page content if it's too long
                page_chunks = self._chunk_text(clean_text, self.chunk_size, self.chunk_overlap)

                for chunk_idx, chunk_text in enumerate(page_chunks):
                    if chunk_text.strip():
                        chunk = DocumentChunk(
                            content=chunk_text.strip(),
                            content_type='text',
                            source_location=f"page_{page_num}_chunk_{chunk_idx}",
                            metadata={
                                'file_name': file_name,
                                'file_path': file_path,
                                'page_number': page_num,
                                'chunk_index': chunk_idx,
                                'source_type': 'pdf'
                            }
                        )
                        chunks.append(chunk)
        else:
            # No page markers, chunk the entire text
            text_chunks = self._chunk_text(text, self.chunk_size, self.chunk_overlap)

            for chunk_idx, chunk_text in enumerate(text_chunks):
                if chunk_text.strip():
                    chunk = DocumentChunk(
                        content=chunk_text.strip(),
                        content_type='text',
                        source_location=f"chunk_{chunk_idx}",
                        metadata={
                            'file_name': file_name,
                            'file_path': file_path,
                            'chunk_index': chunk_idx,
                            'source_type': 'pdf'
                        }
                    )
                    chunks.append(chunk)

        return chunks

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            # Find chunk end position
            end = start + chunk_size

            # If this isn't the last chunk, try to break at a sentence or paragraph
            if end < len(text):
                # Look for sentence endings
                for break_char in ['.\n', '!\n', '?\n', '.', '!', '?']:
                    break_pos = text.rfind(break_char, start, end)
                    if break_pos > start + chunk_size // 2:  # Don't break too early
                        end = break_pos + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = max(start + chunk_size - overlap, end)

            # Prevent infinite loop
            if start >= len(text):
                break

        return chunks