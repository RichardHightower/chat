import os
from typing import Optional, Dict, Any, List

import litellm

from chat.ai.llm_provider import LLMProvider
from chat.conversation.conversation import Conversation, MessageType
from chat.util.logging_util import logger


class OllamaProvider(LLMProvider):
    """Integration with Ollama models using LiteLLM."""

    def __init__(self, api_key: Optional[str] = None, model: str = "llama3.3:latest"):
        # Ollama doesn't require an API key, but we'll keep this parameter for consistency
        self.api_key = api_key

        # LiteLLM's naming convention for Ollama models depends on the model name format
        # If the model name contains a colon (e.g., "gemma3:27b"), we need to keep that
        # Otherwise, we use the format "ollama/model_name"
        self.original_model_name = model

        if ":" in model:
            # Models with versions/variants like gemma3:27b should be formatted
            # as ollama/gemma3:27b for LiteLLM
            self.model = f"ollama/{model}"
        elif not model.startswith("ollama/"):
            self.model = f"ollama/{model}"
        else:
            self.model = model

        # Default Ollama base URL
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        os.environ["OLLAMA_API_BASE"] = self.base_url

        try:
            self.client = litellm
            logger.info(f"OllamaProvider initialized with model: {self.model} at {self.base_url}")
        except ImportError:
            logger.error("litellm package not installed. Please install it (e.g., pip install litellm)")
            raise

    async def generate_completion(
            self,
            prompt: str,
            output_format: str = "text",
            options: Optional[Dict[str, Any]] = None,
            conversation: Optional[Conversation] = None
    ) -> str:
        """Generate a completion from an Ollama model using LiteLLM."""
        options = options or {}

        # Adjust parameters based on model size and type
        if "70b" in self.original_model_name or "72b" in self.original_model_name:
            default_max_tokens = 2048  # Most conservative for the largest models
        elif "27b" in self.original_model_name or "32b" in self.original_model_name:
            default_max_tokens = 2560  # Conservative for medium-large models
        elif "llama4:scout" in self.original_model_name:
            default_max_tokens = 4096  # Scout should be efficient enough for this
        else:
            default_max_tokens = 4096  # Standard value for smaller models

        max_tokens = options.get("max_tokens", default_max_tokens)
        temperature = options.get("temperature", 0.7)
        system_prompt = options.get("system_prompt",
                                    "You are a helpful assistant specializing in providing accurate and relevant information.")

        logger.info(f"Sending request to Ollama with model: {self.model}")

        try:
            # Convert conversation history to messages format expected by the LLM
            if conversation and conversation.messages:
                # Add the new prompt if it's not already the last user message
                messages = conversation.to_llm_messages()

                # Ensure a system message exists at the beginning
                if not any(msg["role"] == "system" for msg in messages):
                    messages.insert(0, {"role": "system", "content": system_prompt})

                # Add the new prompt if it's not already the last user message
                if not (messages and messages[-1]["role"] == "user" and messages[-1]["content"] == prompt):
                    messages.append({"role": "user", "content": prompt})
            else:
                # Standard message format without conversation history
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]

            logger.info(f"Using {len(messages)} messages in conversation history")

            # For very large models, we might need to limit context size
            if len(messages) > 10:
                if "70b" in self.original_model_name or "72b" in self.original_model_name:
                    logger.info(f"Trimming conversation history for large model ({self.original_model_name})")
                    # Keep system message and most recent messages
                    messages = [messages[0]] + messages[-7:]  # More aggressive trimming for largest models
                elif "27b" in self.original_model_name or "32b" in self.original_model_name:
                    logger.info(f"Trimming conversation history for medium-large model ({self.original_model_name})")
                    # Keep system message and most recent messages
                    messages = [messages[0]] + messages[-9:]  # Less aggressive trimming

            # Set up additional parameters for the request
            extra_params = {}
            if "stop_sequences" in options and options["stop_sequences"]:
                extra_params["stop"] = options["stop_sequences"]

            # Add timeout parameters based on model size
            if "70b" in self.original_model_name or "72b" in self.original_model_name:
                extra_params["request_timeout"] = 180  # 3 minute timeout for largest models
            elif "27b" in self.original_model_name or "32b" in self.original_model_name:
                extra_params["request_timeout"] = 120  # 2 minute timeout for medium-large models
            else:
                extra_params["request_timeout"] = 60  # 1 minute timeout for standard models

            # Get context window size if set in session state
            import streamlit as st
            if "ollama_context_size" in st.session_state and (
                    "70b" in self.original_model_name or
                    "72b" in self.original_model_name or
                    "27b" in self.original_model_name or
                    "32b" in self.original_model_name
            ):
                # Adjust max_tokens based on user-set context size
                max_tokens = min(max_tokens, st.session_state.ollama_context_size)
                logger.info(f"Using user-defined context size: {max_tokens}")

            response = await self.client.acompletion(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **extra_params
            )

            output = response.choices[0].message.content
            reason = response.choices[0].finish_reason or "unknown"
            logger.info(f"Received response from Ollama. Finish reason: {reason}, Output length: {len(output or '')}")

            # Update conversation if provided
            if conversation and output:
                conversation.add_message(output, MessageType.OUTPUT)

            return output or ""

        except Exception as e:
            logger.error(f"Error generating completion from Ollama via LiteLLM: {e}", exc_info=True)
            if "connection" in str(e).lower() and "refused" in str(e).lower():
                return "Error: Could not connect to Ollama server. Please ensure Ollama is running and the base URL is correct."
            elif "model not found" in str(e).lower():
                return f"Error: Model '{self.original_model_name}' not found. Please make sure you've pulled this model using 'ollama pull {self.original_model_name}'."
            elif "timeout" in str(e).lower():
                return f"Error: Request timed out. The model '{self.original_model_name}' might be loading or require more resources than available."
            elif "out of memory" in str(e).lower() or "oom" in str(e).lower():
                return f"Error: Out of memory error. The model '{self.original_model_name}' requires more RAM than currently available. Try reducing the context size in the settings."
            else:
                raise

    async def generate_json(
            self,
            prompt: str,
            schema: Dict[str, Any],
            options: Optional[Dict[str, Any]] = None,
            conversation: Optional[Conversation] = None
    ) -> Dict[str, Any]:
        """Generate a JSON response from Ollama using the parent class implementation."""
        return await super().generate_json(prompt, schema, options, conversation)