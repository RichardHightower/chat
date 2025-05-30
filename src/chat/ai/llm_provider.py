import json
from typing import Optional, Dict, Any, AsyncGenerator, Callable
from abc import ABC, abstractmethod

# Assuming jsonschema and litellm are installed
import jsonschema

from chat.util.json_util import JsonUtil
from chat.conversation.conversation import Conversation
from chat.util.logging_util import logger


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate_completion(
            self,
            prompt: str,
            output_format: str = "text",
            options: Optional[Dict[str, Any]] = None,
            conversation: Optional[Conversation] = None
    ) -> str:
        """
        Generate a completion from the LLM for the given prompt.

        Args:
            prompt: The user's input prompt
            output_format: The desired output format (text or json)
            options: Additional options for the completion
            conversation: Optional conversation history to provide context

        Returns:
            The generated completion as a string
        """
        pass

    async def generate_completion_stream(
            self,
            prompt: str,
            output_format: str = "text",
            options: Optional[Dict[str, Any]] = None,
            conversation: Optional[Conversation] = None,
            callback: Optional[Callable[[str], None]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming completion from the LLM for the given prompt.
        
        Note: This method should be implemented by providers that support streaming.
        The default implementation raises NotImplementedError.

        Args:
            prompt: The user's input prompt
            output_format: The desired output format (text or json)
            options: Additional options for the completion
            conversation: Optional conversation history to provide context
            callback: Optional callback function to update UI with each chunk

        Yields:
            Text chunks as they become available
        """
        raise NotImplementedError("This provider does not support streaming completions")
        
    async def generate_json(
            self,
            prompt: str,
            schema: Dict[str, Any],
            options: Optional[Dict[str, Any]] = None,
            conversation: Optional[Conversation] = None
    ) -> Dict[str, Any]:
        """
        Generate a completion from the LLM for the given prompt, expecting JSON output matching the schema.

        Args:
            prompt: The user's input prompt
            schema: JSON schema that the response should conform to
            options: Additional options for the completion
            conversation: Optional conversation history to provide context

        Returns:
            The generated completion as a validated JSON dictionary
        """
        options = options or {}  # Ensure options is a dict
        # Store schema in options if generate_completion needs it for specific modes
        options_with_schema = {**options, "schema": schema}

        json_str = await self.generate_completion(
            prompt,
            output_format="json_object",
            options=options_with_schema,
            conversation=conversation
        )

        # Use the JsonUtil from this module
        json_dict = JsonUtil.extract_json(json_str)
        try:
            jsonschema.validate(instance=json_dict, schema=schema)
        except jsonschema.ValidationError as e:
            logger.error(
                f"JSON validation error: {e.message}\nSchema: {json.dumps(schema, indent=2)}\nJSON String: {json_str[:500]}...\nParsed Dict: {json_dict}")
            raise ValueError(f"JSON validation error: {e.message}\nJSON received: {json_str}") from e
        return json_dict