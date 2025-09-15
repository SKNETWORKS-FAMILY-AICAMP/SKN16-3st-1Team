"""
Document processing error classes
"""


class DocumentProcessingError(Exception):
    """Base exception for document processing errors"""
    pass


class UnsupportedFileFormatError(DocumentProcessingError):
    """Raised when file format is not supported"""
    pass


class FileAccessError(DocumentProcessingError):
    """Raised when file cannot be accessed"""
    pass


class ProcessingTimeoutError(DocumentProcessingError):
    """Raised when processing times out"""
    pass


class InvalidDocumentStructureError(DocumentProcessingError):
    """Raised when document structure is invalid"""
    pass


class MissingDependencyError(DocumentProcessingError):
    """Raised when required dependency is missing"""
    pass


class ContentExtractionError(DocumentProcessingError):
    """Raised when content extraction fails"""
    pass