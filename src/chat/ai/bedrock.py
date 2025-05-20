# src/chat/ai/bedrock.py
import os
import json
import uuid
from typing import Optional, Dict, Any, AsyncGenerator, Callable
import boto3
from botocore.exceptions import ClientError
import asyncio
import litellm

from chat.ai.llm_provider import LLMProvider
from chat.conversation.conversation import Conversation, MessageType
from chat.util.logging_util import logger
from chat.util.streaming_util import stream_response


class BedrockProvider(LLMProvider):
    """Integration with Amazon Bedrock foundation models using both direct API and LiteLLM."""

    # The full ARN of the inference profile from your AWS console
    CLAUDE_INFERENCE_PROFILE_ARN = "arn:aws:bedrock:us-west-2:043309360196:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"

    def __init__(self, api_key: Optional[str] = None, model: str = "anthropic.claude-3-sonnet-20240229-v1:0",
                 inference_profile: Optional[str] = None):
        # For Bedrock, we don't use the api_key parameter directly, but use AWS credentials
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_session_token = os.getenv("AWS_SESSION_TOKEN")  # Optional session token for temp credentials
        self.aws_region = os.getenv("AWS_REGION", "us-west-2")  # Default to us-west-2

        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ValueError("AWS credentials are required. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env")

        # Store the original model name
        self.original_model = model

        # Initialize boto3 client for direct API calls
        bedrock_client_kwargs = {
            "service_name": "bedrock-runtime",
            "region_name": self.aws_region,
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key
        }

        # Add session token if available
        if self.aws_session_token:
            bedrock_client_kwargs["aws_session_token"] = self.aws_session_token
            logger.info("Using AWS session token for temporary credentials")

        self.bedrock_runtime = boto3.client(**bedrock_client_kwargs)
        logger.info(f"Direct boto3 bedrock-runtime client initialized for region {self.aws_region}")

        # Set the inference profile ARN
        self.inference_profile = inference_profile or self.CLAUDE_INFERENCE_PROFILE_ARN

        # Special handling for Claude 3.7 - we'll use direct boto3 calls
        self.use_direct_api = model == "anthropic.claude-3-7-sonnet-20250219-v1:0"

        if self.use_direct_api:
            logger.info(f"Using direct boto3 API for Claude 3.7 model with inference profile: {self.inference_profile}")
            self.model = model  # Store original model ID for direct API
        else:
            # For non-Claude 3.7 models, set up LiteLLM
            if not model.startswith("bedrock/"):
                self.model = f"bedrock/{model}"
            else:
                self.model = model

            # Set environment variables for LiteLLM to use
            os.environ["AWS_ACCESS_KEY_ID"] = self.aws_access_key_id
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_access_key
            os.environ["AWS_REGION_NAME"] = self.aws_region

            if self.aws_session_token:
                os.environ["AWS_SESSION_TOKEN"] = self.aws_session_token

            # Initialize LiteLLM client for non-Claude 3.7 models
            self.client = litellm
            logger.info(f"LiteLLM initialized for regular model: {self.model}")

    async def generate_completion(
            self,
            prompt: str,
            output_format: str = "text",
            options: Optional[Dict[str, Any]] = None,
            conversation: Optional[Conversation] = None
    ) -> str:
        """Generate a completion from AWS Bedrock models."""
        options = options or {}

        # Add the new user message to the conversation if provided
        if conversation:
            conversation.add_message(prompt, MessageType.INPUT)

        # Default system prompt
        system_prompt = options.get("system_prompt",
                                    "You are a helpful assistant specializing in technical writing and software engineering.")

        # Determine max tokens based on model
        if "claude" in self.model.lower() or "claude" in self.original_model.lower():
            if "claude-3-7" in self.model.lower() or "claude-3-7" in self.original_model.lower():
                max_tokens = options.get("max_tokens", 32000)  # Claude 3.7 has larger context
            elif "opus" in self.model.lower():
                max_tokens = options.get("max_tokens", 16000)
            elif "sonnet" in self.model.lower():
                max_tokens = options.get("max_tokens", 8000)
            elif "haiku" in self.model.lower():
                max_tokens = options.get("max_tokens", 4000)
            else:
                max_tokens = options.get("max_tokens", 4000)  # Default for older Claude models
        elif "llama" in self.model.lower():
            max_tokens = options.get("max_tokens", 4096)  # Llama models
        else:
            max_tokens = options.get("max_tokens", 1024)  # Default for other models

        # Get temperature parameter with a reasonable default
        temperature = options.get("temperature", 0.7)

        # Prepare the messages
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

        # Decide which API to use
        if self.use_direct_api:
            return await self._generate_with_boto3(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                output_format=output_format,
                conversation=conversation
            )
        else:
            return await self._generate_with_litellm(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                output_format=output_format,
                conversation=conversation
            )

    async def _generate_with_litellm(
            self,
            messages: list,
            max_tokens: int,
            temperature: float,
            output_format: str,
            conversation: Optional[Conversation] = None
    ) -> str:
        """Generate completion using LiteLLM (for regular models)."""
        llm_params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "aws_region_name": self.aws_region,
        }

        # Handle response format for models that support it
        if output_format.lower() == "json" and "claude" in self.model.lower():
            llm_params["response_format"] = {"type": "json_object"}
            logger.info("Using JSON response format")

        logger.info(f"Sending request to AWS Bedrock ({self.model}) with LiteLLM")
        logger.info(f"Using {len(messages)} messages in conversation history")

        # Make the API call
        response = await self.client.acompletion(**llm_params)

        output = response.choices[0].message.content
        reason = response.choices[0].finish_reason or "unknown"
        logger.info(f"Received response from Bedrock. Finish reason: {reason}, "
                    f"Output length: {len(output or '')}")

        # Update conversation if provided
        if conversation and output:
            conversation.add_message(output, MessageType.OUTPUT)

        return output or ""

    async def _generate_with_boto3(
            self,
            messages: list,
            max_tokens: int,
            temperature: float,
            output_format: str,
            conversation: Optional[Conversation] = None
    ) -> str:
        """Generate completion using direct boto3 API for Claude 3.7."""
        # Extract system message
        system_content = None
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}]
                })

        # Create request payload
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_messages
        }

        # Add system message if present
        if system_content:
            request_body["system"] = system_content

        # Add response format for JSON if needed
        if output_format.lower() == "json":
            request_body["response_format"] = {"type": "json_object"}
            logger.info("Using JSON response format for direct API call")

        # Convert to JSON and encode
        body_bytes = json.dumps(request_body).encode('utf-8')

        logger.info(f"Sending direct API request to Bedrock for Claude 3.7")
        logger.info(
            f"Request includes {len(anthropic_messages)} messages and system prompt: {system_content is not None}")

        try:
            # Make the API call with the body containing our whole request
            # Note: No inferenceProfileArn parameter here - we'll directly use the inference profile
            # as the modelId according to AWS documentation
            response = self.bedrock_runtime.invoke_model(
                modelId=self.inference_profile,  # Use the inference profile ARN as the modelId
                body=body_bytes,
                accept="application/json",
                contentType="application/json"
            )

            # Parse the response body
            response_body = json.loads(response['body'].read())

            # Extract the text from the response
            if 'content' in response_body and len(response_body['content']) > 0:
                output = response_body['content'][0]['text']
                logger.info(f"Received response from direct Bedrock API, length: {len(output)}")

                # Update conversation if provided
                if conversation and output:
                    conversation.add_message(output, MessageType.OUTPUT)

                return output
            else:
                logger.warning(f"Unexpected response format: {response_body}")
                return "Error: Unexpected response format from AWS Bedrock"

        except (ClientError, Exception) as e:
            # Log the error details
            logger.error(f"Bedrock error: {str(e)}")

            # Check for specific error types
            if hasattr(e, 'response') and 'Error' in getattr(e, 'response', {}):
                error_code = e.response['Error'].get('Code', 'Unknown')
                error_message = e.response['Error'].get('Message', str(e))
                logger.error(f"AWS Error {error_code}: {error_message}")

                if "AccessDenied" in error_code:
                    raise ValueError(f"Access denied. Please check your AWS credentials and permissions.")
                elif "ValidationException" in error_code:
                    raise ValueError(f"Validation error: {error_message}")
                elif "ResourceNotFoundException" in error_code:
                    raise ValueError(f"Resource not found: {error_message}")

            # Generic error
            raise ValueError(f"AWS Bedrock error: {str(e)}")

    async def generate_completion_stream(
            self,
            prompt: str,
            output_format: str = "text",
            options: Optional[Dict[str, Any]] = None,
            conversation: Optional[Conversation] = None,
            callback: Optional[Callable[[str], None]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming completion from AWS Bedrock models."""
        # For Claude 3.7, we'll use non-streaming approach and return in one go
        if self.use_direct_api:
            logger.info(f"Using non-streaming approach for Claude 3.7 model with direct boto3")

            try:
                # Generate the full response
                full_response = await self.generate_completion(
                    prompt=prompt,
                    output_format=output_format,
                    options=options,
                    conversation=conversation
                )

                # If we have a callback, call it with the entire output
                if callback:
                    callback(full_response)

                # Yield the entire output
                yield full_response

                return

            except Exception as e:
                error_message = str(e)
                logger.error(f"Error in non-streaming boto3 approach: {error_message}")

                if callback:
                    callback(f"\nError: {error_message}")
                yield f"\nError: {error_message}"

                return

        # For other models, use standard LiteLLM streaming
        options = options or {}

        # Add the new user message to the conversation if provided
        if conversation:
            conversation.add_message(prompt, MessageType.INPUT)

        # Default system prompt
        system_prompt = options.get("system_prompt",
                                    "You are a helpful assistant specializing in technical writing and software engineering.")

        # Determine max tokens based on model
        if "claude" in self.model.lower() or "claude" in self.original_model.lower():
            if "claude-3-7" in self.model.lower() or "claude-3-7" in self.original_model.lower():
                max_tokens = options.get("max_tokens", 32000)  # Claude 3.7 has larger context
            elif "opus" in self.model.lower():
                max_tokens = options.get("max_tokens", 16000)
            elif "sonnet" in self.model.lower():
                max_tokens = options.get("max_tokens", 8000)
            elif "haiku" in self.model.lower():
                max_tokens = options.get("max_tokens", 4000)
            else:
                max_tokens = options.get("max_tokens", 4000)  # Default for older Claude models
        elif "llama" in self.model.lower():
            max_tokens = options.get("max_tokens", 4096)  # Llama models
        else:
            max_tokens = options.get("max_tokens", 1024)  # Default for other models

        # Get temperature parameter with a reasonable default
        temperature = options.get("temperature", 0.7)

        # Prepare messages
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

        # Set up streaming options
        stream_options = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "aws_region_name": self.aws_region,
        }

        # Handle response format for models that support it
        if output_format.lower() == "json" and "claude" in self.model.lower():
            stream_options["response_format"] = {"type": "json_object"}
            logger.info("Using JSON response format for streaming")

        logger.info(f"Starting streaming request to AWS Bedrock with model: {self.model}")
        logger.info(f"Using {len(messages)} messages in conversation history for streaming")

        try:
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

            logger.info("Streaming completed successfully")

        except Exception as e:
            error_message = str(e)
            logger.error(f"Error in streaming: {error_message}")

            # Create a user-friendly error message
            if "You don't have access to the model" in error_message:
                error_msg = f"Access denied to model: {self.model}. Check your AWS credentials and permissions."
            elif "Streaming not supported for" in error_message:
                error_msg = f"Streaming is not supported for this model. Try using non-streaming mode instead."
            elif "failed to authenticate request" in error_message.lower():
                error_msg = "Authentication failed. Please check your AWS credentials and region configuration."
            else:
                error_msg = f"Error during streaming: {e}"

            if callback:
                callback(f"\n{error_msg}")
            yield f"\n{error_msg}"