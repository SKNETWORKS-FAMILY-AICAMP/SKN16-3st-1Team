from abc import ABC, abstractmethod
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from document_processor.core import DocumentProcessor, ProcessingResult

 # Import for notebook processing
try:
    import nbformat
except ImportError:
    nbformat = None

class NotebookProcessor(DocumentProcessor):
    """Jupyter notebook processor"""

    def __init__(self):
        if nbformat is None:
            raise ImportError("nbformat is required for notebook processing")

    def get_supported_formats(self) -> List[str]:
        """Get supported notebook formats"""
        return ['.ipynb']
    
    def process_file(self, file_path: str) -> ProcessingResult:
        """Process notebook file and extract content"""
        start_time = time.time()
        return ProcessingResult(
                    success=False,
                    chunks=[],
                    error_message="Invalid notebook file",
                    processing_time=time.time() - start_time,
                    file_path=file_path)
                