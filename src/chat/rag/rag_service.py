"""
RAG (Retrieval Augmented Generation) service for the chat application.
This service handles document management, text chunking, and semantic search.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

from vector_rag.api import VectorRAGAPI
from vector_rag.config import Config as VectorRAGConfig
from vector_rag.db import DBFileHandler
from vector_rag.embeddings import SentenceTransformersEmbedder
from vector_rag.model import Chunk, ChunkResult, ChunkResults, File, Project

logger = logging.getLogger(__name__)


class RAGService:
    """Service for managing RAG functionality."""

    def __init__(self, config_overrides: Optional[Dict] = None):
        """Initialize the RAG service.

        Args:
            config_overrides: Optional overrides for the vector_rag configuration.
        """
        self.config = VectorRAGConfig(**(config_overrides or {}))
        logger.info(f"Initializing RAG service with config: {self.config.as_dict()}")

        self.api = VectorRAGAPI(config=self.config)
        self.handler = self.api.handler

        # Initialize current project
        self._current_project = None

    @property
    def current_project(self) -> Optional[Project]:
        """Get the currently selected project."""
        return self._current_project

    def get_or_create_project(self, name: str, description: Optional[str] = None) -> Project:
        """Get an existing project or create a new one.

        Args:
            name: Name of the project
            description: Optional description of the project

        Returns:
            Project object
        """
        project = self.handler.get_or_create_project(name, description)
        self._current_project = project
        return project

    def list_projects(self) -> List[Project]:
        """List all available projects.

        Returns:
            List of projects
        """
        return self.handler.get_projects()

    def add_document(self,
                     project_id: int,
                     filename: str,
                     content: str,
                     metadata: Optional[Dict] = None) -> Optional[File]:
        """Add a document to a project.

        Args:
            project_id: ID of the project
            filename: Name of the file
            content: Content of the file
            metadata: Optional metadata for the file

        Returns:
            File object if successful, None otherwise
        """
        import hashlib

        # Create a unique path for the file
        file_path = f"/chat_app/documents/{filename}"

        # Generate CRC
        crc = hashlib.md5(content.encode()).hexdigest()

        # Create file model
        file_model = File(
            name=filename,
            path=file_path,
            crc=crc,
            content=content,
            metadata=metadata or {}
        )

        # Add file to project
        return self.handler.add_file(project_id, file_model)

    def search(self,
               project_id: int,
               query: str,
               top_k: int = 5,
               similarity_threshold: float = 0.7) -> List[ChunkResult]:
        """Search for chunks matching the query.

        Args:
            project_id: ID of the project to search
            query: Search query text
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of chunk results, sorted by similarity score
        """
        results = self.handler.search_chunks_by_text(
            project_id=project_id,
            query_text=query,
            page=1,
            page_size=top_k,
            similarity_threshold=similarity_threshold
        )

        return results.results if results else []

    def get_context_for_query(self,
                              project_id: int,
                              query: str,
                              max_chunks: int = 5,
                              max_tokens: int = 1500) -> str:
        """Get context for a query as a formatted string.

        Args:
            project_id: ID of the project to search
            query: Search query text
            max_chunks: Maximum number of chunks to include
            max_tokens: Approximate maximum number of tokens to include

        Returns:
            Formatted context string for the query
        """
        results = self.search(
            project_id=project_id,
            query=query,
            top_k=max_chunks
        )

        if not results:
            return ""

        context_parts = []
        total_length = 0
        approximate_token_ratio = 4  # Approximate characters per token
        max_chars = max_tokens * approximate_token_ratio

        for i, result in enumerate(results):
            chunk = result.chunk
            source = f"Document: {chunk.metadata.get('filename', 'Unknown')}"
            text = chunk.content

            # Skip if this would exceed our token budget
            if total_length + len(text) > max_chars:
                break

            context_parts.append(f"[{i + 1}] {source}\n{text}\n")
            total_length += len(text)

        return "\n".join(context_parts)

    def list_files(self, project_id: int) -> List[File]:
        """List all files in a project.

        Args:
            project_id: ID of the project

        Returns:
            List of files
        """
        return self.handler.list_files(project_id)