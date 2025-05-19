import asyncio
from typing import AsyncGenerator, Callable, Optional, Dict, Any
import logging

# Use the application's logger
from chat.util.logging_util import logger

async def stream_response(
    client,
    messages: list,
    stream_options: Dict[str, Any],
    callback: Optional[Callable[[str], None]] = None
) -> AsyncGenerator[str, None]:
    """
    Stream a response from any LLM provider using LiteLLM.
    
    Args:
        client: The LiteLLM client
        messages: The messages to send
        stream_options: Additional options for the streaming call
        callback: Optional callback function to update UI
        
    Yields:
        Text chunks as they become available
    """
    try:
        # Set up streaming request with LiteLLM
        full_text = ""
        
        # Extract common options
        model = stream_options.get("model")
        max_tokens = stream_options.get("max_tokens", 4096)
        temperature = stream_options.get("temperature", 0.7)
        
        # Add system message if provided
        if "system" in stream_options and stream_options["system"]:
            # Check if a system message already exists
            if not any(msg.get("role") == "system" for msg in messages):
                messages.insert(0, {"role": "system", "content": stream_options["system"]})
        
        logger.info(f"Starting streaming with model {model}, {len(messages)} messages")
        
        # Prepare parameters for LiteLLM
        completion_params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True  # Important: enable streaming
        }
        
        # Handle response_format if present (for OpenAI models)
        if "response_format" in stream_options:
            completion_params["response_format"] = stream_options["response_format"]
            logger.info(f"Using response_format: {stream_options['response_format']}")
        
        # Handle reasoning_effort if present (for some OpenAI models)
        # Note: gpt-4o doesn't support reasoning_effort in streaming mode
        if "reasoning_effort" in stream_options and not model.startswith("gpt-4o"):
            completion_params["reasoning_effort"] = stream_options["reasoning_effort"]
            # Add to allowed params for OpenAI
            completion_params["allowed_openai_params"] = completion_params.get("allowed_openai_params", [])
            if "reasoning_effort" not in completion_params["allowed_openai_params"]:
                completion_params["allowed_openai_params"].append("reasoning_effort")
            logger.info(f"Using reasoning_effort: {stream_options['reasoning_effort']}")
        elif "reasoning_effort" in stream_options:
            logger.info(f"Skipping reasoning_effort parameter for {model} as it's not supported in streaming mode")
        
        # Use LiteLLM's streaming interface
        logger.info(f"Setting up streaming with LiteLLM for model: {model}")
        response = await client.acompletion(**completion_params)
        logger.info(f"Got streaming response object of type: {type(response)}")
        
        async for chunk in response:
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                
                # Extract content from delta
                if hasattr(delta, 'content') and delta.content:
                    text_chunk = delta.content
                    full_text += text_chunk
                    
                    # Call the callback if provided
                    if callback:
                        callback(text_chunk)
                    
                    # Yield the chunk
                    yield text_chunk
        
        logger.info(f"Streaming completed, total length: {len(full_text)}")
        # In an async generator, we don't return a value
            
    except Exception as e:
        error_msg = f"Error in LLM streaming: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if callback:
            callback(f"\nError: {str(e)}")
        yield f"\nError: {str(e)}"