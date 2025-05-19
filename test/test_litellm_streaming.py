"""
Test script for streaming functionality with LiteLLM.

This script tests the streaming functionality independently from the main application.
"""

import os
import asyncio
import litellm
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

async def test_streaming():
    """Test the streaming functionality with LiteLLM."""
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment variables.")
        return

    # Set up the environment variable for LiteLLM
    os.environ["ANTHROPIC_API_KEY"] = api_key
    
    print("Testing streaming functionality with LiteLLM")
    try:
        ver = getattr(litellm, "__version__", "unknown")
        print(f"LiteLLM version: {ver}")
    except:
        print("Could not determine LiteLLM version")
    
    # Test prompt
    prompt = "Write a short poem about streaming data, one line at a time."
    
    # Set up messages
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Model to use
    model = "anthropic/claude-3-7-sonnet-latest"
    
    print(f"Sending prompt to {model} via LiteLLM: {prompt}")
    print("\nResponse:")
    
    # Set up streaming
    full_response = ""
    try:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
            stream=True
        )
        
        async for chunk in response:
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                
                # Extract content from delta
                if hasattr(delta, 'content') and delta.content:
                    text_chunk = delta.content
                    full_response += text_chunk
                    
                    # Print each chunk
                    print(text_chunk, end="", flush=True)
        
        # Print a newline at the end
        print("\n")
        print("-" * 50)
        print(f"Complete response ({len(full_response)} characters):")
        print(full_response)
            
    except Exception as e:
        print(f"\n\nError during streaming: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_streaming())