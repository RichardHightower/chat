import os
from typing import Optional, Dict, Any

import litellm

from chat.llm_provider import LLMProvider
from chat.conversation import Conversation, MessageType
from chat.logging_util import logger


class GoogleGeminiProvider(LLMProvider):
    """Integration with Google Gemini models using LiteLLM."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2-flash"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set it in .env or as an environment variable.")

        # Use LiteLLM's model naming convention for Gemini
        self.model = model
        if not self.model.startswith("gemini/"):
            self.model = f"gemini/{model}"

        os.environ["GEMINI_API_KEY"] = self.api_key

        try:
            self.client = litellm
            logger.info(f"GoogleGeminiProvider initialized with model: {self.model}")
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
        """Generate a completion from Gemini using LiteLLM."""
        options = options or {}
        max_tokens = options.get("max_tokens", 64000)
        temperature = options.get("temperature", 0.7)
        system_prompt = options.get("system_prompt",
                                    "You are a helpful assistant specializing in providing accurate and relevant information.")

        logger.info(f"Sending request to Gemini with model: {self.model}")

        try:
            # Determine message format based on conversation history
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

            response = await self.client.acompletion(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            output = response.choices[0].message.content
            reason = response.choices[0].finish_reason or "unknown"
            logger.info(f"Received response from Gemini. Finish reason: {reason}, Output length: {len(output or '')}")

            # Update conversation if provided
            if conversation and output:
                conversation.add_message(output, MessageType.OUTPUT)

            return output or ""

        except Exception as e:
            logger.error(f"Error generating completion from Google Gemini via LiteLLM: {e}", exc_info=True)
            raise

    async def generate_json(
            self,
            prompt: str,
            schema: Dict[str, Any],
            options: Optional[Dict[str, Any]] = None,
            conversation: Optional[Conversation] = None
    ) -> Dict[str, Any]:
        """Generate a JSON response from Gemini using the parent class implementation."""
        return await super().generate_json(prompt, schema, options, conversation)
