import os
from typing import Optional, Dict, Any, AsyncGenerator, Callable
import asyncio

import litellm

from chat.ai.llm_provider import LLMProvider
from chat.conversation.conversation import Conversation, MessageType
from chat.util.logging_util import logger
from chat.util.streaming_util import stream_response


class PerplexityProvider(LLMProvider):
    """Integration with Perplexity's research-focused LLM using LiteLLM."""

    def __init__(self, api_key: Optional[str] = None, model: str = "sonar-pro"):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("Perplexity API key is required. Set it in .env or as an environment variable.")

        # Use Perplexity's online models for web search capabilities
        self.model = model

        # Ensure we're using an online model for search capabilities
        if not self._is_online_model(model):
            logger.warning(
                f"Model {model} may not have online search capabilities. Recommended models: sonar-pro, sonar-small-online")

        # Use LiteLLM's model naming convention for Perplexity
        if not model.startswith("perplexity/"):
            self.model = f"perplexity/{model}"

        os.environ["PERPLEXITY_API_KEY"] = self.api_key

        try:
            self.client = litellm
            logger.info(f"PerplexityProvider initialized with model: {self.model}")
        except ImportError:
            logger.error("litellm package not installed. Please install it (e.g., pip install litellm)")
            raise

    def _is_online_model(self, model_name: str) -> bool:
        """Check if the model has online search capabilities."""
        online_indicators = ["online", "search", "sonar"]
        return any(indicator in model_name.lower() for indicator in online_indicators)

    async def generate_completion(
            self,
            prompt: str,
            output_format: str = "text",
            options: Optional[Dict[str, Any]] = None,
            conversation: Optional[Conversation] = None
    ) -> str:
        """Generate a completion from Perplexity using LiteLLM."""
        options = options or {}
        max_tokens = options.get("max_tokens", 20000)
        temperature = options.get("temperature", 0.7)
        system_prompt = options.get("system_prompt",
                                    "You are a helpful assistant specializing in providing accurate and relevant information.")

        logger.info(f"Sending request to Perplexity with model: {self.model}")

        try:
            # Create a properly formatted message list for Perplexity
            # Always start with a system message
            messages = [{"role": "system", "content": system_prompt}]
            
            # Then ensure strict user/assistant alternation
            if conversation and conversation.messages:
                # Get raw messages from conversation
                raw_messages = []
                for msg in conversation.messages:
                    role = "user" if msg.message_type == MessageType.INPUT else "assistant"
                    raw_messages.append({"role": role, "content": msg.content})
                
                # Ensure strict alternation
                formatted_messages = []
                expected_role = "user"  # First message after system should be user
                
                for msg in raw_messages:
                    if not formatted_messages:
                        # First message must be from user
                        if msg["role"] == "user":
                            formatted_messages.append(msg)
                            expected_role = "assistant"
                        # If first message is assistant, skip it (we'll add the prompt as user message later)
                    else:
                        # For subsequent messages, ensure alternation
                        if msg["role"] == expected_role:
                            formatted_messages.append(msg)
                            expected_role = "user" if expected_role == "assistant" else "assistant"
                        else:
                            # If we have consecutive messages with the same role, combine their content
                            if formatted_messages and formatted_messages[-1]["role"] == msg["role"]:
                                formatted_messages[-1]["content"] += "\n\n" + msg["content"]
                
                # Add formatted messages to our message list
                messages.extend(formatted_messages)
                
                # Ensure the last message is from user with the current prompt
                if not formatted_messages or formatted_messages[-1]["role"] == "assistant":
                    # If last message was assistant or no messages, add the prompt as user
                    messages.append({"role": "user", "content": prompt})
                elif formatted_messages[-1]["role"] == "user" and formatted_messages[-1]["content"] != prompt:
                    # If last message was user but with different content, add an assistant response and then the new prompt
                    messages.append({"role": "assistant", "content": "I understand. Please continue."})
                    messages.append({"role": "user", "content": prompt})
            else:
                # No conversation history, just add a user message with the prompt
                messages.append({"role": "user", "content": prompt})

            # Add detailed logging of the message sequence
            logger.info(f"Using {len(messages)} messages in conversation history")
            logger.info(f"Detailed message sequence:")
            for i, msg in enumerate(messages):
                logger.info(f"  Message {i}: role={msg['role']}, content_preview=\"{msg['content'][:50]}...\"")
            
            # Log the full sequence of roles for easier debugging
            role_sequence = [msg['role'] for msg in messages]
            logger.info(f"Role sequence: {role_sequence}")

            # Validate message sequence before sending
            self._validate_message_sequence(messages)

            response = await self.client.acompletion(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            output = response.choices[0].message.content
            reason = response.choices[0].finish_reason or "unknown"
            logger.info(
                f"Received response from Perplexity. Finish reason: {reason}, Output length: {len(output or '')}")

            # Update conversation if provided
            if conversation and output:
                conversation.add_message(output, MessageType.OUTPUT)

            return output or ""

        except Exception as e:
            logger.error(f"Error generating completion from Perplexity via LiteLLM: {e}", exc_info=True)
            # Log the message sequence that caused the error
            if 'messages' in locals():
                logger.error(f"Error occurred with this message sequence: {[msg['role'] for msg in messages]}")
            raise

    def _validate_message_sequence(self, messages):
        """Validate that the message sequence follows Perplexity's requirements."""
        if not messages:
            return
            
        # First message(s) can be system
        i = 0
        while i < len(messages) and messages[i]["role"] == "system":
            i += 1
            
        # After system messages, roles must alternate between user and assistant
        if i < len(messages):
            # First non-system message must be user
            if messages[i]["role"] != "user":
                raise ValueError("First message after system must be from user")
                
            expected_role = "assistant"
            i += 1
            
            # Check remaining messages
            while i < len(messages):
                if messages[i]["role"] != expected_role:
                    raise ValueError(f"Expected {expected_role} message at position {i}, but got {messages[i]['role']}")
                expected_role = "user" if expected_role == "assistant" else "assistant"
                i += 1

    async def generate_json(
            self,
            prompt: str,
            schema: Dict[str, Any],
            options: Optional[Dict[str, Any]] = None,
            conversation: Optional[Conversation] = None
    ) -> Dict[str, Any]:
        """Generate a JSON response from Perplexity using the parent class implementation."""
        return await super().generate_json(prompt, schema, options, conversation)
        
    async def generate_completion_stream(
            self,
            prompt: str,
            output_format: str = "text",
            options: Optional[Dict[str, Any]] = None,
            conversation: Optional[Conversation] = None,
            callback: Optional[Callable[[str], None]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming completion from Perplexity using LiteLLM."""
        options = options or {}
        max_tokens = options.get("max_tokens", 20000)
        temperature = options.get("temperature", 0.7)
        system_prompt = options.get("system_prompt",
                                   "You are a helpful assistant specializing in providing accurate and relevant information.")

        logger.info(f"Starting streaming request to Perplexity with model: {self.model}")

        try:
            # Create a properly formatted message list for Perplexity
            # Always start with a system message
            messages = [{"role": "system", "content": system_prompt}]
            
            # Then ensure strict user/assistant alternation
            if conversation and conversation.messages:
                # Get raw messages from conversation
                raw_messages = []
                for msg in conversation.messages:
                    role = "user" if msg.message_type == MessageType.INPUT else "assistant"
                    raw_messages.append({"role": role, "content": msg.content})
                
                # Ensure strict alternation
                formatted_messages = []
                expected_role = "user"  # First message after system should be user
                
                for msg in raw_messages:
                    if not formatted_messages:
                        # First message must be from user
                        if msg["role"] == "user":
                            formatted_messages.append(msg)
                            expected_role = "assistant"
                        # If first message is assistant, skip it (we'll add the prompt as user message later)
                    else:
                        # For subsequent messages, ensure alternation
                        if msg["role"] == expected_role:
                            formatted_messages.append(msg)
                            expected_role = "user" if expected_role == "assistant" else "assistant"
                        else:
                            # If we have consecutive messages with the same role, combine their content
                            if formatted_messages and formatted_messages[-1]["role"] == msg["role"]:
                                formatted_messages[-1]["content"] += "\n\n" + msg["content"]
                
                # Add formatted messages to our message list
                messages.extend(formatted_messages)
                
                # Ensure the last message is from user with the current prompt
                if not formatted_messages or formatted_messages[-1]["role"] == "assistant":
                    # If last message was assistant or no messages, add the prompt as user
                    messages.append({"role": "user", "content": prompt})
                elif formatted_messages[-1]["role"] == "user" and formatted_messages[-1]["content"] != prompt:
                    # If last message was user but with different content, add an assistant response and then the new prompt
                    messages.append({"role": "assistant", "content": "I understand. Please continue."})
                    messages.append({"role": "user", "content": prompt})
            else:
                # No conversation history, just add a user message with the prompt
                messages.append({"role": "user", "content": prompt})

            # Log the message sequence
            logger.info(f"Using {len(messages)} messages in conversation history for streaming")
            role_sequence = [msg['role'] for msg in messages]
            logger.info(f"Role sequence: {role_sequence}")

            # Validate message sequence before sending
            self._validate_message_sequence(messages)
            
            # Set up streaming options
            stream_options = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            full_response = ""
            # Use the streaming utility with LiteLLM
            async for chunk in stream_response(
                client=self.client,
                messages=messages,
                stream_options=stream_options,
                callback=callback
            ):
                full_response += chunk
                yield chunk
            
            # Update conversation with complete response
            if conversation and full_response:
                conversation.add_message(full_response, MessageType.OUTPUT)
                
        except Exception as e:
            error_msg = f"Error generating streaming completion from Perplexity: {e}"
            logger.error(error_msg, exc_info=True)
            # Log the message sequence that caused the error
            if 'messages' in locals():
                logger.error(f"Error occurred with this message sequence: {[msg['role'] for msg in messages]}")
            if callback:
                callback(f"\nError: {str(e)}")
            yield f"\nError: {str(e)}"