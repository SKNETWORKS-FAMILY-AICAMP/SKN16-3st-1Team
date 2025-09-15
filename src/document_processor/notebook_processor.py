from abc import ABC, abstractmethod
import time
import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from .core import DocumentProcessor, ProcessingResult, DocumentChunk

# Import for notebook processing
try:
    import nbformat
except ImportError:
    nbformat = None

class NotebookProcessor(DocumentProcessor):
    """Jupyter notebook processor"""

    def __init__(self, include_outputs: bool = True, include_code: bool = True, include_markdown: bool = True):
        if nbformat is None:
            raise ImportError("nbformat is required for notebook processing")

        self.include_outputs = include_outputs
        self.include_code = include_code
        self.include_markdown = include_markdown

    def get_supported_formats(self) -> List[str]:
        """Get supported notebook formats"""
        return ['.ipynb']

    def validate_file(self, file_path: str) -> bool:
        """Validate if notebook file can be processed"""
        try:
            self._validate_file_exists(file_path)

            # Check file extension
            if not file_path.lower().endswith('.ipynb'):
                return False

            # Try to read the notebook to verify it's valid JSON and has notebook structure
            with open(file_path, 'r', encoding='utf-8') as f:
                notebook_data = json.load(f)

            # Check if it has the basic notebook structure
            return ('cells' in notebook_data and
                    'metadata' in notebook_data and
                    isinstance(notebook_data['cells'], list))

        except (json.JSONDecodeError, FileNotFoundError, KeyError, TypeError):
            return False
        except Exception:
            return False
    
    def process_file(self, file_path: str) -> ProcessingResult:
        """Process notebook file and extract content"""
        start_time = time.time()

        try:
            # Validate file first
            if not self.validate_file(file_path):
                return ProcessingResult(
                    success=False,
                    chunks=[],
                    error_message="Invalid notebook file or file does not exist",
                    processing_time=time.time() - start_time,
                    file_path=file_path
                )

            # Get file info
            file_size = os.path.getsize(file_path)
            file_name = Path(file_path).name

            # Read and parse notebook
            with open(file_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)

            # Process cells and create chunks
            chunks = self._process_notebook_cells(notebook, file_path, file_name)

            if not chunks:
                return ProcessingResult(
                    success=False,
                    chunks=[],
                    error_message="No processable content found in notebook",
                    processing_time=time.time() - start_time,
                    file_path=file_path,
                    file_size=file_size
                )

            processing_time = time.time() - start_time

            return ProcessingResult(
                success=True,
                chunks=chunks,
                error_message=None,
                processing_time=processing_time,
                file_path=file_path,
                file_size=file_size,
                total_pages=len(notebook.cells)  # Use cell count as "pages"
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                chunks=[],
                error_message=f"Error processing notebook: {str(e)}",
                processing_time=time.time() - start_time,
                file_path=file_path
            )

    def _process_notebook_cells(self, notebook: 'nbformat.NotebookNode', file_path: str, file_name: str) -> List[DocumentChunk]:
        """Process notebook cells and create document chunks"""
        chunks = []

        for cell_idx, cell in enumerate(notebook.cells):
            cell_type = cell.cell_type

            # Process markdown cells
            if cell_type == 'markdown' and self.include_markdown:
                if hasattr(cell, 'source') and cell.source.strip():
                    chunk = DocumentChunk(
                        content=cell.source.strip(),
                        content_type='markdown',
                        source_location=f"cell_{cell_idx}",
                        metadata={
                            'file_name': file_name,
                            'file_path': file_path,
                            'cell_index': cell_idx,
                            'cell_type': cell_type,
                            'source_type': 'notebook'
                        }
                    )
                    chunks.append(chunk)

            # Process code cells
            elif cell_type == 'code' and self.include_code:
                if hasattr(cell, 'source') and cell.source.strip():
                    # Add code chunk
                    chunk = DocumentChunk(
                        content=cell.source.strip(),
                        content_type='code',
                        source_location=f"cell_{cell_idx}",
                        metadata={
                            'file_name': file_name,
                            'file_path': file_path,
                            'cell_index': cell_idx,
                            'cell_type': cell_type,
                            'source_type': 'notebook',
                            'execution_count': getattr(cell, 'execution_count', None)
                        }
                    )
                    chunks.append(chunk)

                    # Process outputs if enabled
                    if self.include_outputs and hasattr(cell, 'outputs'):
                        for output_idx, output in enumerate(cell.outputs):
                            output_text = self._extract_output_text(output)
                            if output_text.strip():
                                output_chunk = DocumentChunk(
                                    content=output_text.strip(),
                                    content_type='output',
                                    source_location=f"cell_{cell_idx}_output_{output_idx}",
                                    metadata={
                                        'file_name': file_name,
                                        'file_path': file_path,
                                        'cell_index': cell_idx,
                                        'output_index': output_idx,
                                        'cell_type': cell_type,
                                        'source_type': 'notebook',
                                        'output_type': output.output_type if hasattr(output, 'output_type') else 'unknown'
                                    }
                                )
                                chunks.append(output_chunk)

            # Process raw cells (treat as text)
            elif cell_type == 'raw':
                if hasattr(cell, 'source') and cell.source.strip():
                    chunk = DocumentChunk(
                        content=cell.source.strip(),
                        content_type='text',
                        source_location=f"cell_{cell_idx}",
                        metadata={
                            'file_name': file_name,
                            'file_path': file_path,
                            'cell_index': cell_idx,
                            'cell_type': cell_type,
                            'source_type': 'notebook'
                        }
                    )
                    chunks.append(chunk)

        return chunks

    def _extract_output_text(self, output: Dict[str, Any]) -> str:
        """Extract text content from notebook cell output"""
        output_text = ""

        try:
            # Handle different output types
            if hasattr(output, 'output_type'):
                output_type = output.output_type

                if output_type == 'stream':
                    # Stream output (stdout, stderr)
                    if hasattr(output, 'text'):
                        if isinstance(output.text, list):
                            output_text = ''.join(output.text)
                        else:
                            output_text = str(output.text)

                elif output_type == 'execute_result' or output_type == 'display_data':
                    # Execution results or display data
                    if hasattr(output, 'data'):
                        # Prefer text/plain representation
                        if 'text/plain' in output.data:
                            data = output.data['text/plain']
                            if isinstance(data, list):
                                output_text = ''.join(data)
                            else:
                                output_text = str(data)
                        # Fallback to text/html (simplified)
                        elif 'text/html' in output.data:
                            html_data = output.data['text/html']
                            if isinstance(html_data, list):
                                output_text = ''.join(html_data)
                            else:
                                output_text = str(html_data)
                            # Simple HTML tag removal (basic)
                            import re
                            output_text = re.sub(r'<[^>]+>', '', output_text)

                elif output_type == 'error':
                    # Error output
                    if hasattr(output, 'traceback'):
                        if isinstance(output.traceback, list):
                            output_text = '\n'.join(output.traceback)
                        else:
                            output_text = str(output.traceback)
                    elif hasattr(output, 'ename') and hasattr(output, 'evalue'):
                        output_text = f"{output.ename}: {output.evalue}"

        except Exception:
            # If extraction fails, return empty string
            output_text = ""

        return output_text
                