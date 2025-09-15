"""
Embedding generation service using sentence-transformers
"""

import logging
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
import numpy as np

# Sentence transformers imports
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Document models import
from ..document_processor.core import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_name: str = "google/embeddinggemma-300m"
    device: str = "cpu"  # or "cuda" if available
    batch_size: int = 32
    max_seq_length: int = 512
    normalize_embeddings: bool = True


class EmbeddingGenerator:
    """Generate embeddings for text using sentence-transformers"""

    def __init__(self, config: EmbeddingConfig):
        """Initialize embedding generator with configuration"""
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )

        self.config = config
        self._model = None
        self._is_loaded = False

    def load_model(self) -> bool:
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.config.model_name}")

            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device
            )

            # Set max sequence length if specified
            if self.config.max_seq_length:
                self._model.max_seq_length = self.config.max_seq_length

            self._is_loaded = True
            logger.info(f"Successfully loaded model: {self.config.model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self._is_loaded = False
            return False

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._is_loaded and self._model is not None

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        return self._model.get_sentence_embedding_dimension()

    def encode_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            # Generate embedding
            embedding = self._model.encode(
                text,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=False
            )

            return embedding

        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not texts:
            raise ValueError("Text list cannot be empty")

        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            raise ValueError("No valid texts found")

        try:
            # Generate embeddings in batches
            embeddings = self._model.encode(
                valid_texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=len(valid_texts) > 10  # Show progress for large batches
            )

            return embeddings

        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise

    def encode_document_chunk(self, chunk: DocumentChunk) -> np.ndarray:
        """Generate embedding for a document chunk"""
        if not isinstance(chunk, DocumentChunk):
            raise TypeError("Input must be a DocumentChunk instance")

        # Create text representation of the chunk
        text_content = chunk.content

        # Optionally include metadata context
        if chunk.metadata:
            # Add some metadata context that might be useful for retrieval
            context_parts = []

            if 'file_name' in chunk.metadata:
                context_parts.append(f"File: {chunk.metadata['file_name']}")

            if 'content_type' in chunk.metadata:
                context_parts.append(f"Type: {chunk.content_type}")

            if context_parts:
                # Prepend context to content
                text_content = " ".join(context_parts) + " " + text_content

        return self.encode_text(text_content)

    def encode_document_chunks(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Generate embeddings for multiple document chunks"""
        if not chunks:
            raise ValueError("Chunks list cannot be empty")

        # Extract text content from chunks
        texts = []
        for chunk in chunks:
            if not isinstance(chunk, DocumentChunk):
                raise TypeError("All items must be DocumentChunk instances")

            text_content = chunk.content

            # Add metadata context
            if chunk.metadata:
                context_parts = []

                if 'file_name' in chunk.metadata:
                    context_parts.append(f"File: {chunk.metadata['file_name']}")

                if 'content_type' in chunk.metadata:
                    context_parts.append(f"Type: {chunk.content_type}")

                if context_parts:
                    text_content = " ".join(context_parts) + " " + text_content

            texts.append(text_content)

        return self.encode_texts(texts)

    # LangChain compatible embedding
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents and return as list of lists of floats"""
        if not texts:
            return []

        try:
            # Generate embeddings using our embedding generator
            embeddings = self.encode_texts(texts)
            # Convert numpy array to list of lists
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise

    # LangChain compatible embedding
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query and return as list of floats"""
        if not text:
            return []

        try:
            # Generate embedding for single text
            embedding = self.encode_text(text)
            # Convert numpy array to list
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        if embedding1.shape != embedding2.shape:
            raise ValueError("Embeddings must have the same shape")

        # Compute cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_loaded():
            return {"loaded": False, "model_name": self.config.model_name}

        try:
            return {
                "loaded": True,
                "model_name": self.config.model_name,
                "embedding_dimension": self.get_embedding_dimension(),
                "max_seq_length": self.config.max_seq_length,
                "device": self.config.device,
                "batch_size": self.config.batch_size,
                "normalize_embeddings": self.config.normalize_embeddings
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"loaded": False, "error": str(e)}


# Factory function for easy generator creation
def create_embedding_generator(
    model_name: str = "google/embeddinggemma-300m",
    device: str = "cpu",
    batch_size: int = 32
) -> EmbeddingGenerator:
    """Factory function to create and configure EmbeddingGenerator"""
    config = EmbeddingConfig(
        model_name=model_name,
        device=device,
        batch_size=batch_size
    )
    generator = EmbeddingGenerator(config)

    # Auto-load the model
    if not generator.load_model():
        raise RuntimeError(f"Failed to load embedding model: {model_name}")

    return generator