"""
RAG integration with chat functionality.

This module contains functions for integrating RAG with chat components,
including context enhancement and result formatting.
"""
import logging

import streamlit as st
from typing import Dict, List, Optional

from chat.rag.rag_service import RAGService
from chat.util.logging_util import logger as llm_logger


def get_rag_context_for_prompt(prompt: str) -> Optional[str]:
    """
    Get RAG context for a prompt if RAG is enabled.

    Args:
        prompt: User prompt

    Returns:
        Enhanced prompt with RAG context if enabled, otherwise None
    """
    # Check if RAG is enabled
    if not st.session_state.get("rag_enabled", False):
        return None

    llm_logger.info("Getting RAG context for prompt...")
    # Check if a project is selected
    project_id = st.session_state.get("current_rag_project_id")
    if not project_id:
        return None

    llm_logger.info(f"Using Project ID: {project_id}")

    try:
        # Get RAG service
        rag_service = st.session_state.rag_service

        # Get settings
        similarity_threshold = st.session_state.get("rag_similarity_threshold", 0.7)
        max_chunks = st.session_state.get("rag_max_chunks", 3)

        llm_logger.info(f"Using Similarity Threshold: {similarity_threshold} and Max Chunks: {max_chunks}")

        # Search for relevant context
        results = rag_service.search(
            project_id=project_id,
            query=prompt,
            top_k=max_chunks,
            similarity_threshold=similarity_threshold
        )

        if not results:
            llm_logger.info("No relevant context found")
            return None

        # Format context
        context_parts = []
        for i, result in enumerate(results):
            chunk = result.chunk
            metadata = chunk.metadata or {}
            filename = metadata.get("filename", "Unknown")
            content = chunk.content

            context_parts.append(f"[Document {i + 1}: {filename}]\n{content}\n")

        context_text = "\n".join(context_parts)

        # Create enhanced prompt
        enhanced_prompt = (
            f"I'll provide you with some context information followed by a question. "
            f"Please use this context to help answer the question accurately.\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION: {prompt}\n\n"
            f"Please answer based on the provided context. If the context doesn't contain "
            f"relevant information, you can rely on your general knowledge, but prioritize "
            f"the context information when available. Include citations like [Document X] in your answer."
        )

        llm_logger.info(f"Enhanced prompt with {len(results)} context chunks")
        return enhanced_prompt

    except Exception as e:
        llm_logger.error(f"Error getting RAG context: {str(e)}", exc_info=True)
        return None


def format_rag_response(response: str) -> str:
    """
    Format a response from the LLM to highlight citations.

    Args:
        response: The response from the LLM

    Returns:
        Formatted response with highlighted citations
    """
    # Simple formatting for now - in a real implementation, you might want to
    # use regex to identify citations and style them
    return response