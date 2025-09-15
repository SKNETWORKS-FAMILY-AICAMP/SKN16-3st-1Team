"""
Core document processor classes and interfaces
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import os
import time
from pathlib import Path

from .errors import UnsupportedFileFormatError


@dataclass
class DocumentChunk:
    """Represents a processed chunk of document content"""
    content: str
    content_type: str  # 'text', 'code', 'output', 'markdown'
    source_location: str  # page number, cell index, etc.
    metadata: Dict[str, Any]

    def __post_init__(self):
        if not self.content:
            raise ValueError("Content cannot be empty")
        if self.content_type not in ['text', 'code', 'output', 'markdown']:
            raise ValueError(f"Invalid content type: {self.content_type}")


@dataclass
class ProcessingResult:
    """Result of document processing operation"""
    success: bool
    chunks: List[DocumentChunk]
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    total_pages: Optional[int] = None

    def __post_init__(self):
        if not self.success and not self.error_message:
            raise ValueError("Error message required when success is False")
        
class DocumentProcessor(ABC):
    """Abstract base class for document processors"""

    @abstractmethod
    def process_file(self, file_path: str) -> ProcessingResult:
        """Process a document file and return chunks"""
        pass

    @abstractmethod
    def validate_file(self, file_path: str) -> bool:
        """Validate if file can be processed"""
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        pass

    def _validate_file_exists(self, file_path: str) -> None:
        """Helper method to validate file existence"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not os.path.isfile(file_path):
            raise ValueError(f"Path is not a file: {file_path}")


    
class DocumentProcessorFactory:
    """Factory for creating document processors"""

    _processors = {}

    @classmethod
    def _get_processors(cls):
        if not cls._processors:
            # Import here to avoid circular imports
            from .pdf_processor import PDFProcessor
            from .notebook_processor import NotebookProcessor
            cls._processors = {
                '.pdf': PDFProcessor,
                '.ipynb': NotebookProcessor
            }
        return cls._processors

    @classmethod
    def create_processor(cls, file_path: str) -> DocumentProcessor:
        """Create appropriate processor for file"""
        processors = cls._get_processors()
        file_ext = Path(file_path).suffix.lower()

        if file_ext not in processors:
            raise UnsupportedFileFormatError(f"Unsupported file format: {file_ext}")

        processor_class = processors[file_ext]
        return processor_class()

    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """Get all supported file formats"""
        processors = cls._get_processors()
        return list(processors.keys())

    @classmethod
    def register_processor(cls, file_extension: str, processor_class: type):
        """Register a new processor for a file extension"""
        processors = cls._get_processors()
        processors[file_extension] = processor_class    