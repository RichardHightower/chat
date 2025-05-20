"""Utility functions for integrating RAG with chat functionality."""

import logging
from typing import Dict, List, Optional, Tuple

from .rag_service import RAGService

logger = logging.getLogger(__name__)


def enhance_prompt_with_context(
        prompt: str,
        rag_service: RAGService,
        project_id: int,
        max_chunks: int = 3,
        similarity_threshold: float = 0.7
) -> str:
    """Enhance a user prompt with context from RAG.

    Args:
        prompt: Original user prompt
        rag_service: RAG service instance
        project_id: ID of the project to search
        max_chunks: Maximum number of chunks to include
        similarity_threshold: Minimum similarity score (0.0 to 1.0)

    Returns:
        Enhanced prompt with context
    """
    context = rag_service.get_context_for_query(
        project_id=project_id,
        query=prompt,
        max_chunks=max_chunks
    )

    if not context:
        return prompt

    # Create a prompt with context
    enhanced_prompt = (
        f"I'll provide you with some context information followed by a question. "
        f"Please use this context to help answer the question accurately.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {prompt}\n\n"
        f"Please answer based on the provided context. If the context doesn't contain "
        f"relevant information, you can rely on your general knowledge, but prioritize "
        f"the context information when available."
    )

    return enhanced_prompt


def format_citation(chunk_result: Dict) -> str:
    """Format a citation for a chunk result.

    Args:
        chunk_result: Dictionary with chunk result data

    Returns:
        Formatted citation string
    """
    metadata = chunk_result.get("metadata", {})
    filename = metadata.get("filename", "Unknown source")

    return f"[Source: {filename}]"