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
               similarity_threshold: float = 0.7,
               metadata_filter: Optional[Dict] = None) -> List[ChunkResult]:
        """Search for chunks matching the query.

        Args:
            project_id: ID of the project to search
            query: Search query text
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            metadata_filter: Optional metadata filters (key-value pairs)

        Returns:
            List of chunk results, sorted by similarity score
        """
        results = self.api.search_text(
            project_id=project_id,
            query_text=query,
            page=1,
            page_size=top_k,
            similarity_threshold=similarity_threshold,
            metadata_filter=metadata_filter
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
    
    def search_bm25(self,
                    project_id: int,
                    query: str,
                    top_k: int = 10,
                    rank_threshold: float = 0.0,
                    metadata_filter: Optional[Dict] = None) -> ChunkResults:
        """Search using BM25 (keyword/lexical) matching.
        
        Args:
            project_id: ID of the project to search
            query: Search query text
            top_k: Number of results to return
            rank_threshold: Minimum BM25 rank score
            metadata_filter: Optional metadata filters (key-value pairs)
            
        Returns:
            ChunkResults object with BM25-ranked results
        """
        return self.api.search_bm25(
            project_id=project_id,
            query_text=query,
            page=1,
            page_size=top_k,
            rank_threshold=rank_threshold,
            metadata_filter=metadata_filter
        )
    
    def search_hybrid(self,
                      project_id: int,
                      query: str,
                      top_k: int = 10,
                      vector_weight: float = 0.5,
                      bm25_weight: float = 0.5,
                      similarity_threshold: float = 0.0,
                      rank_threshold: float = 0.0,
                      metadata_filter: Optional[Dict] = None) -> ChunkResults:
        """Search using hybrid approach (vector + BM25).
        
        Args:
            project_id: ID of the project to search
            query: Search query text
            top_k: Number of results to return
            vector_weight: Weight for semantic similarity (0-1)
            bm25_weight: Weight for BM25 score (0-1)
            similarity_threshold: Minimum vector similarity score
            rank_threshold: Minimum BM25 rank score
            metadata_filter: Optional metadata filters (key-value pairs)
            
        Returns:
            ChunkResults object with hybrid-scored results
        """
        return self.api.search_hybrid(
            project_id=project_id,
            query_text=query,
            page=1,
            page_size=top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            similarity_threshold=similarity_threshold,
            rank_threshold=rank_threshold,
            metadata_filter=metadata_filter
        )
    
    def query_metadata(self,
                       project_id: int,
                       metadata_filter: Dict,
                       query_text: Optional[str] = None,
                       file_id: Optional[int] = None,
                       top_k: int = 50) -> ChunkResults:
        """Query chunks by metadata only (no semantic or BM25 scoring).
        
        Args:
            project_id: ID of the project to search
            metadata_filter: Dictionary of metadata filters (required)
            query_text: Optional simple text search in content (uses ILIKE)
            file_id: Optional file ID to limit search to specific file
            top_k: Number of results to return (Note: query API doesn't support pagination)
            
        Returns:
            ChunkResults object with metadata-filtered results
        """
        # Note: The query method doesn't support pagination, so we get all results
        # and the API will handle any internal limits
        return self.api.query(
            project_id=project_id,
            file_id=file_id,
            query_text=query_text,
            metadata_filter=metadata_filter
        )