"""
Provider management module for the chat application.

This module contains functions for initializing and managing LLM providers.
"""

from typing import Dict, Any, Optional, Tuple

from chat.ai.open_ai import OpenAIProvider
from chat.ai.google_gemini import GoogleGeminiProvider
from chat.ai.perplexity import PerplexityProvider
from chat.ai.anthropic import AnthropicProvider
from chat.ai.ollama import OllamaProvider
from chat.ai.llm_provider import LLMProvider
from chat.util.logging_util import logger as llm_logger


# Define available providers and their default models
PROVIDERS = {
    "OpenAI": {
        "class": OpenAIProvider,
        "models": ["gpt-4o-2024-08-06", "gpt-4.1-2025-04-14",
                   "gpt-4o", "gpt-4.1", "o4-mini", "o3", "o3-mini",
                   "chatgpt-4o-latest"]
    },
    "Google Gemini": {
        "class": GoogleGeminiProvider,
        "models": ["gemini-2.5-pro-preview-05-06",
                   "gemini-2.0-flash-001", "gemini-2.0-flash-lite-001",
                   "gemini-2.5-flash-preview-04-17",
                   "gemini-2.0-flash-live-preview-04-09"]
    },
    "Perplexity": {
        "class": PerplexityProvider,
        "models": ["sonar-pro", "sonar", "sonar-deep-research",
                   "sonar-reasoning-pro", "sonar-reasoning", "r1-1776"
                   ]
    },
    "Anthropic": {
        "class": AnthropicProvider,
        "models": ["claude-3-7-sonnet-latest",
                   "claude-3-5-haiku-latest", "claude-3-opus-latest"]
    },
    "Ollama": {
        "class": OllamaProvider,
        "models": ["gemma3:27b", "qwen3:32b", "qwen:72b", "deepseek-r1:70b", "llama3.3:latest", "llama4:scout",
                   "mistral", "mixtral", "phi3"]
    }
}


def initialize_provider(provider_name: str, model_name: str) -> Tuple[Optional[LLMProvider], Optional[str]]:
    """
    Initialize an LLM provider with the specified model.

    Args:
        provider_name: Name of the provider to initialize
        model_name: Name of the model to use

    Returns:
        Tuple containing (provider_instance, error_message)
        If initialization is successful, error_message will be None
        If initialization fails, provider_instance will be None and error_message will contain the error
    """
    try:
        # Get the provider class from the PROVIDERS dictionary
        provider_class = PROVIDERS[provider_name]["class"]

        # Initialize the provider with the selected model
        provider_instance = provider_class(model=model_name)

        llm_logger.info(f"Provider initialized: {provider_name} with model: {model_name}")
        return provider_instance, None

    except ValueError as e:
        error_message = f"Error initializing provider: {e}"
        llm_logger.error(error_message)
        return None, error_message

    except Exception as e:
        error_message = f"An unexpected error occurred during provider initialization: {e}"
        llm_logger.error(error_message, exc_info=True)
        return None, error_message


def get_available_providers() -> Dict[str, Dict[str, Any]]:
    """
    Get the dictionary of available providers and their models.

    Returns:
        Dictionary of providers and their configuration
    """
    return PROVIDERS