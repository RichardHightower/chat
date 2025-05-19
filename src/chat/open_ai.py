import os
from typing import Optional, Dict, Any, Literal, List

import litellm

from chat.llm_provider import LLMProvider
from chat.conversation import Conversation, MessageType
from chat.logging_util import logger


class OpenAIProvider(LLMProvider):
    """Integration with OpenAI GPT models using LiteLLM."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-2024-08-06"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it in .env or as an environment variable.")

        self.model = model
        os.environ["OPENAI_API_KEY"] = self.api_key  # LiteLLM expects this

        try:
            self.client = litellm
            logger.info(f"OpenAIProvider initialized with model: {self.model}")
        except ImportError:
            logger.error("litellm package not installed. Please install it (e.g., pip install litellm)")
            raise

    async def _generate_completion_gpt4_series(
            self,
            prompt: str,
            response_format: Dict[str, Any],
            options: Optional[Dict[str, Any]] = None,  # Made options optional
            conversation: Optional[Conversation] = None
    ) -> str:
        options = options or {}

        if self.model.startswith("gpt-4.1"):  # Matches "gpt-4.1-2025-04-14"
            max_tokens = options.get("max_tokens", 32768)
        elif self.model.startswith("gpt-4o"):
            max_tokens = options.get("max_tokens", 4096)
        else:
            max_tokens = options.get("max_tokens", 16384)  # Default for other gpt-4

        temperature = options.get("temperature", 0.5)
        reasoning_effort = self.get_reasoning_effort(options)
        system_prompt = options.get("system_prompt",
                                    "You are a helpful assistant specializing in technical writing and software engineering.")

        try:
            # Prepare the messages - either from conversation history or create new
            if conversation and conversation.messages:
                # Convert conversation history to messages format
                messages = conversation.to_llm_messages()

                # Add the new prompt as a user message if not already present
                if not (messages and messages[-1]["role"] == "user" and messages[-1]["content"] == prompt):
                    messages.append({"role": "user", "content": prompt})

                # Ensure the system prompt is set
                if not any(msg["role"] == "system" for msg in messages):
                    messages.insert(0, {"role": "system", "content": system_prompt})
            else:
                # Standard message format without conversation history
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]

            llm_params: Dict[str, Any] = {  # Explicitly type llm_params
                "model": self.model,
                "response_format": response_format,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }

            llm_params["allowed_openai_params"] = llm_params.get("allowed_openai_params", [])

            if self._supports_reasoning_effort(self.model) and not self.model.startswith("gpt-4o"):
                llm_params["reasoning_effort"] = reasoning_effort
                if "reasoning_effort" not in llm_params["allowed_openai_params"]:
                    llm_params["allowed_openai_params"].append("reasoning_effort")

            # Log params without messages for brevity
            loggable_params = {k: v for k, v in llm_params.items() if k != "messages"}
            logger.info(f"Sending request to LiteLLM (GPT-4 series) with params: {loggable_params}")
            logger.info(f"Using {len(messages)} messages in conversation history")

            response = await self.client.acompletion(**llm_params)

            output = response.choices[0].message.content
            reason = response.choices[0].finish_reason or "unknown"
            logger.info(
                f"Received response (GPT-4 series). Finish reason: {reason}, Output length: {len(output or '')}")

            # Update conversation if provided
            if conversation and output:
                conversation.add_message(output, MessageType.OUTPUT)

            if reason == "stop":
                return output or ""  # Ensure string return
            elif reason == "length":
                logger.info("Output truncated due to length (GPT-4 series). Attempting to continue.")
                return await self._generate_continue(prompt, output or "", options, llm_params,
                                                     conversation=conversation)
            else:  # Includes other reasons like 'tool_calls', 'content_filter', etc.
                logger.warning(f"Unexpected finish_reason (GPT-4 series): {reason}. Returning output as is.")
                return output or ""

        except Exception as e:
            logger.error(f"Error generating completion from OpenAI {self.model} (GPT-4 series) via LiteLLM: {e}",
                         exc_info=True)
            raise

    async def _generate_completion_o_series(
            self,
            prompt: str,
            response_format: Dict[str, Any],
            options: Optional[Dict[str, Any]] = None,
            conversation: Optional[Conversation] = None
    ) -> str:
        options = options or {}
        # Your class uses 'max_completion_tokens' for o-series
        # LiteLLM typically uses 'max_tokens' for OpenAI models.
        # We'll use 'max_tokens' in the call to LiteLLM but source it from 'max_completion_tokens' if present.
        if self.model.startswith("gpt-4o"):
            max_completion_tokens = options.get("max_completion_tokens", options.get("max_tokens", 16384))
        else:
            max_completion_tokens = options.get("max_completion_tokens", options.get("max_tokens", 100000))

        reasoning_effort = self.get_reasoning_effort(options)
        system_prompt = options.get("system_prompt",
                                    "You are a helpful assistant specializing in technical writing and software engineering.")

        try:
            # Prepare the messages - either from conversation history or create new
            if conversation and conversation.messages:
                # Convert conversation history to messages format
                messages = conversation.to_llm_messages()

                # Add the new prompt as a user message if not already present
                if not (messages and messages[-1]["role"] == "user" and messages[-1]["content"] == prompt):
                    messages.append({"role": "user", "content": prompt})

                # Ensure the system prompt is set
                if not any(msg["role"] == "system" for msg in messages):
                    messages.insert(0, {"role": "system", "content": system_prompt})
            else:
                # Standard message format without conversation history
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]

            llm_params: Dict[str, Any] = {  # Explicitly type llm_params
                "model": self.model,
                "response_format": response_format,
                "max_tokens": max_completion_tokens,  # This is what LiteLLM's acompletion expects for OpenAI
                "messages": messages
            }

            # Handle temperature for O-series models
            is_o_series = "o" in self.model.lower()
            if is_o_series:
                # O-series models only support temperature=1.0
                llm_params["temperature"] = 1.0
                logger.info("Using temperature=1.0 for O-series model (only supported value)")
            elif "temperature" in options:
                # For non-O-series models, use the provided temperature
                llm_params["temperature"] = options.get("temperature", 0.5)

            llm_params["allowed_openai_params"] = llm_params.get("allowed_openai_params", [])

            if self._supports_reasoning_effort(self.model) and not self.model.startswith("gpt-4o"):
                llm_params["reasoning_effort"] = reasoning_effort
                if "reasoning_effort" not in llm_params["allowed_openai_params"]:
                    llm_params["allowed_openai_params"].append("reasoning_effort")

            loggable_params = {k: v for k, v in llm_params.items() if k != "messages"}
            logger.info(f"Sending request to LiteLLM (o-series) with params: {loggable_params}")
            logger.info(f"Using {len(messages)} messages in conversation history")

            response = await self.client.acompletion(**llm_params)

            output = response.choices[0].message.content
            reason = response.choices[0].finish_reason or "unknown"
            logger.info(f"Received response (o-series). Finish reason: {reason}, Output length: {len(output or '')}")

            # Update conversation if provided
            if conversation and output:
                conversation.add_message(output, MessageType.OUTPUT)

            if reason == "stop":
                return output or ""
            elif reason == "length":
                logger.info("Output truncated due to length (o-series). Attempting to continue.")
                return await self._generate_continue(prompt, output or "", options, llm_params,
                                                     conversation=conversation)
            else:
                logger.warning(f"Unexpected finish_reason (o-series): {reason}. Returning output as is.")
                return output or ""

        except Exception as e:
            logger.error(f"Error generating completion from OpenAI {self.model} (o-series) via LiteLLM: {e}",
                         exc_info=True)
            raise

    @staticmethod
    def _supports_reasoning_effort(model: str) -> bool:
        return model.startswith("gpt-4o") or model.startswith("o3")

    async def generate_completion(
            self,
            prompt: str,
            output_format: str = "text",
            options: Optional[Dict[str, Any]] = None,
            conversation: Optional[Conversation] = None
    ) -> str:
        options = options or {}
        response_format = await self._create_response_format(options, output_format)

        # Add the new user message to the conversation if provided
        if conversation:
            conversation.add_message(prompt, MessageType.INPUT)

        if self.model.startswith("o") or self.model.startswith("gpt-4o"):  # Matching gpt-4o to o-series logic
            logger.info(f"Routing to o-series completion for model {self.model}")
            return await self._generate_completion_o_series(prompt, response_format, options, conversation)
        else:  # Handles gpt-4, gpt-4.1, etc.
            logger.info(f"Routing to GPT-4 series completion for model {self.model}")
            return await self._generate_completion_gpt4_series(prompt, response_format, options, conversation)

    @staticmethod
    async def _create_response_format(options: Dict[str, Any], output_format: str) -> Dict[str, Any]:
        normalized_format = "text"  # Default for chat
        if output_format.lower() == "json_object" or (
                output_format.lower() == "json" and "schema" in options):  # More specific for json_object
            normalized_format = "json_object"
        elif output_format.lower() == "json":  # Simple JSON, could be text if schema not present
            normalized_format = "json_object"  # Assume intent is JSON mode

        response_format_dict = {"type": normalized_format}

        logger.info(f"Response format created: {response_format_dict} for input format '{output_format}'")
        return response_format_dict

    # generate_json is inherited from LLMProvider ABC and uses the above generate_completion

    async def _generate_continue(
            self,
            original_prompt: str,
            current_output: str,
            options: Dict[str, Any],
            llm_params: Dict[str, Any],  # Original params from the first call
            max_calls: int = 3,
            conversation: Optional[Conversation] = None
    ) -> str:
        accumulated_output = current_output
        continuation_instruction = "Continue exactly where you left off, providing the next part of the response. Do not repeat any part of the previous response. Start directly with the new content."

        # Start with a copy of the original messages from the first call
        messages_history = list(llm_params.get("messages", []))
        # Add the assistant's partial response so far
        messages_history.append({"role": "assistant", "content": current_output})
        # Add the user's instruction to continue
        messages_history.append({"role": "user", "content": continuation_instruction})

        # Determine max_tokens for continuation based on original params
        # This should be the max_tokens for *each continuation chunk*, not the total.
        # The original llm_params['max_tokens'] or llm_params['max_completion_tokens']
        # was for the first chunk. We can reuse it for subsequent chunks.
        continue_max_tokens = llm_params.get("max_tokens") or llm_params.get("max_completion_tokens")

        for attempt in range(1, max_calls + 1):
            try:
                logger.info(
                    f"Continuation attempt {attempt}/{max_calls}. Current accumulated length: {len(accumulated_output)}")

                continuation_api_params = {
                    "model": self.model,
                    "messages": messages_history,
                    "max_tokens": continue_max_tokens,  # Max tokens for this specific continuation call
                    "temperature": llm_params.get("temperature"),
                    "response_format": llm_params.get("response_format"),  # Preserve original response format
                    "allowed_openai_params": llm_params.get("allowed_openai_params", [])
                }
                if "reasoning_effort" in llm_params and self._supports_reasoning_effort(self.model):
                    continuation_api_params["reasoning_effort"] = llm_params["reasoning_effort"]

                response = await self.client.acompletion(**continuation_api_params)

                new_chunk = response.choices[0].message.content or ""  # Ensure it's a string
                finish_reason = response.choices[0].finish_reason or "unknown"
                logger.info(
                    f"Continuation attempt {attempt} - Finish reason: {finish_reason}, New chunk length: {len(new_chunk)}")

                accumulated_output += new_chunk

                # Note: we don't add continuation chunks to the conversation history
                # as they're part of the same logical message

                if finish_reason == "stop":
                    logger.info("Continuation successful, finish_reason is 'stop'.")
                    return accumulated_output
                elif finish_reason == "length":
                    if not new_chunk:  # Safety break if it returns empty but says length
                        logger.warning(f"Attempt {attempt}: finish_reason 'length' but no new content. Stopping.")
                        return accumulated_output
                    logger.warning(
                        f"Attempt {attempt}: output cut off again during continuation. Preparing next attempt...")
                    messages_history.append({"role": "assistant", "content": new_chunk})
                    messages_history.append({"role": "user", "content": continuation_instruction})
                else:
                    logger.warning(
                        f"Attempt {attempt}: Unexpected finish_reason during continuation: {finish_reason}. Stopping.")
                    return accumulated_output

            except Exception as e:
                logger.error(f"Error continuing generation at attempt {attempt}: {e}", exc_info=True)
                return accumulated_output  # Return what we have so far on error

        logger.warning("Maximum continuation attempts reached. Returning accumulated output.")
        return accumulated_output

    @staticmethod
    def get_reasoning_effort(options: Dict[str, Any]) -> Literal["low", "medium", "high"]:
        reasoning_effort_value = options.get("reasoning_effort", "high")
        if reasoning_effort_value not in ["low", "medium", "high"]:
            logger.warning(f"Invalid reasoning_effort value '{reasoning_effort_value}'. Defaulting to 'high'.")
            reasoning_effort_value = "high"
        return reasoning_effort_value  # type: ignore