"""
LangChain Chroma vector store client for educational documents
"""

import os
import logging
from typing import Optional, Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain.embeddings.base import Embeddings
from dataclasses import dataclass
from datetime import datetime

# LangChain Chroma imports
try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    Chroma = None

logger = logging.getLogger(__name__)


@dataclass
class ChromaConfig:
    """Configuration for LangChain Chroma vector store"""
    persist_directory: str
    collection_name: str = "educational_documents"
    embedding_function: Optional["Embeddings"] = None
    max_results: int = 10


class ChromaClient:
    """Client for managing LangChain Chroma vector store"""

    def __init__(self, config: ChromaConfig):
        """Initialize LangChain Chroma client with configuration"""
        if Chroma is None:
            raise ImportError("langchain is required. Install with: pip install langchain chromadb")

        self.config = config
        self._vectorstore = None
        self._is_connected = False

    def connect(self, embedding_function: Optional["Embeddings"] = None) -> bool:
        """Connect to LangChain Chroma vector store"""
        try:
            # Ensure persist directory exists and is absolute path
            persist_path = os.path.abspath(self.config.persist_directory)
            os.makedirs(persist_path, exist_ok=True)

            # Use provided embedding function or config one
            embeddings = embedding_function or self.config.embedding_function
            if embeddings is None:
                raise ValueError("Embedding function is required")

            # Create LangChain Chroma vector store
            self._vectorstore = Chroma(
                collection_name=self.config.collection_name,
                embedding_function=embeddings,
                persist_directory=persist_path
            )

            self._is_connected = True
            logger.info(f"Connected to Chroma vector store: {self.config.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Chroma vector store: {e}")
            self._is_connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Chroma vector store"""
        try:
            if self._vectorstore:
                # Persist the vector store
                self._vectorstore.persist()
                self._vectorstore = None
                self._is_connected = False
                logger.info("Disconnected from Chroma vector store")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    def is_connected(self) -> bool:
        """Check if connected to Chroma vector store"""
        return self._is_connected and self._vectorstore is not None

    def get_vectorstore(self) -> Optional[Chroma]:
        """Get the current vector store"""
        if not self.is_connected():
            raise ConnectionError("Not connected to Chroma vector store")
        return self._vectorstore

    def get_client(self):
        """Get the underlying ChromaDB client"""
        if not self.is_connected():
            raise ConnectionError("Not connected to Chroma vector store")
        return self._vectorstore._client if self._vectorstore else None

    def reset_collection(self) -> bool:
        """Reset (delete and recreate) the collection"""
        try:
            if not self.is_connected():
                raise ConnectionError("Not connected to Chroma vector store")

            # Delete existing collection through the underlying client
            client = self.get_client()
            if client:
                try:
                    client.delete_collection(name=self.config.collection_name)
                    logger.info(f"Deleted collection: {self.config.collection_name}")
                except Exception:
                    # Collection doesn't exist, that's fine
                    pass

            # Reconnect to create new collection
            embedding_function = self._vectorstore._embedding_function if self._vectorstore else None
            self._vectorstore = None
            self._is_connected = False

            # Reconnect with same embedding function
            success = self.connect(embedding_function)
            if success:
                logger.info(f"Reset collection: {self.config.collection_name}")
            return success

        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        if not self.is_connected():
            raise ConnectionError("Not connected to Chroma vector store")

        try:
            # Get collection info through the underlying client
            client = self.get_client()
            collection = client.get_collection(name=self.config.collection_name) if client else None

            count = collection.count() if collection else 0
            metadata = collection.metadata if collection else {}

            return {
                "name": self.config.collection_name,
                "count": count,
                "metadata": metadata,
                "config": {
                    "persist_directory": self.config.persist_directory,
                    "max_results": self.config.max_results
                }
            }

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Chroma vector store connection"""
        try:
            health_status = {
                "connected": self.is_connected(),
                "vectorstore_available": self._vectorstore is not None,
                "timestamp": datetime.now().isoformat()
            }

            if self.is_connected():
                collection_info = self.get_collection_info()
                health_status.update({
                    "collection_name": collection_info.get("name"),
                    "document_count": collection_info.get("count", 0),
                    "persist_directory_exists": os.path.exists(self.config.persist_directory)
                })

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "connected": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Factory function for easy client creation
def create_chroma_client(
    persist_directory: str,
    collection_name: str = "educational_documents",
    embedding_function: Optional["Embeddings"] = None
) -> ChromaClient:
    """Factory function to create and configure ChromaClient"""
    config = ChromaConfig(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embedding_function
    )
    return ChromaClient(config)