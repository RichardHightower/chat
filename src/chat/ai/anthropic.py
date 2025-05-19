import os
from typing import Optional, Dict, Any

import litellm

from chat.ai.llm_provider import LLMProvider
from chat.conversation.conversation import Conversation, MessageType
from chat.util.logging_util import logger


class AnthropicProvider(LLMProvider):
    """Integration with Anthropic Claude models using LiteLLM."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-7-sonnet-latest"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Set it in .env or as an environment variable.")

        self.model = model
        self.original_model_name = model

        # Use LiteLLM's model naming convention for Anthropic
        if not self.model.startswith("anthropic/"):
            self.model = f"anthropic/{model}"

        os.environ["ANTHROPIC_API_KEY"] = self.api_key

        try:
            self.client = litellm
            logger.info(f"AnthropicProvider initialized with model: {self.model}")
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
        """Generate a completion from Claude using LiteLLM."""
        options = options or {}
        
        # Set appropriate max_tokens based on the model
        if "haiku" in self.original_model_name.lower():
            default_max_tokens = 8000  # Slightly below the 8192 limit for safety
        elif "sonnet" in self.original_model_name.lower():
            default_max_tokens = 32000
        else:  # opus or other models
            default_max_tokens = 64000
            
        max_tokens = options.get("max_tokens", default_max_tokens)
        temperature = options.get("temperature", 0.7)
        system_prompt = options.get("system_prompt",
                                    "You are a helpful assistant specializing in providing accurate and relevant information.")

        logger.info(f"Sending request to Claude with model: {self.model}, max_tokens: {max_tokens}")

        try:
            # Determine message format based on conversation history
            if conversation and conversation.messages:
                # Add the new prompt if it's not already the last user message
                # This prevents duplicating the prompt if it was already added to the conversation
                # by a previous call to this method
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

            response = await self.client.acompletion(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            output = response.choices[0].message.content
            reason = response.choices[0].finish_reason or "unknown"
            logger.info(f"Received response from Claude. Finish reason: {reason}, Output length: {len(output or '')}")

            # Update conversation if provided
            if conversation and output:
                conversation.add_message(output, MessageType.OUTPUT)

            return output or ""

        except Exception as e:
            logger.error(f"Error generating completion from Anthropic via LiteLLM: {e}", exc_info=True)
            raise

    async def generate_json(
            self,
            prompt: str,
            schema: Dict[str, Any],
            options: Optional[Dict[str, Any]] = None,
            conversation: Optional[Conversation] = None
    ) -> Dict[str, Any]:
        """Generate a JSON response from Claude using the parent class implementation."""
        return await super().generate_json(prompt, schema, options, conversation)