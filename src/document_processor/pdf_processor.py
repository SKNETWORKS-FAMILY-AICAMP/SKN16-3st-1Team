from abc import ABC, abstractmethod
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from document_processor.core import DocumentProcessor, ProcessingResult

# Import for PDF processing
try:
    import PyPDF2
    from PyPDF2 import PdfReader
except ImportError:
    PyPDF2 = None
    PdfReader = None

class PDFProcessor(DocumentProcessor):
    """PDF document processor"""

    def __init__(self):
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required for PDF processing")

    def get_supported_formats(self) -> List[str]:
        """Get supported PDF formats"""
        return ['.pdf']
    
    def process_file(self, file_path: str) -> ProcessingResult:
        """Process PDF file and extract text content"""
        start_time = time.time()
        fail_result = ProcessingResult(
                    success=False,
                    chunks=[],
                    error_message="Invalid PDF file",
                    processing_time=time.time() - start_time,
                    file_path=file_path)

        return fail_result