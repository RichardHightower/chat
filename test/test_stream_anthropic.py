"""
Simplified test for Anthropic streaming using our utility functions.
"""

import os
import asyncio
from dotenv import load_dotenv
import litellm
from src.chat.util.streaming_util import stream_response
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def simple_stream_test():
    """Test streaming with our utility function."""
    # Ensure API key is set
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment variables")
        return
    
    os.environ["ANTHROPIC_API_KEY"] = api_key
    
    # Set up the client
    client = litellm
    
    # Test messages
    messages = [
        {"role": "user", "content": "Write a short poem about data streaming, line by line."}
    ]
    
    # Stream options
    stream_options = {
        "model": "anthropic/claude-3-7-sonnet-latest",
        "max_tokens": 1000,
        "temperature": 0.7,
        "system": "You are a helpful assistant."
    }
    
    print(f"Testing streaming with model: {stream_options['model']}")
    
    # Define callback for printing chunks
    def print_chunk(chunk):
        print(chunk, end="", flush=True)
    
    # Run streaming test
    full_text = ""
    try:
        async for chunk in stream_response(
            client=client,
            messages=messages,
            stream_options=stream_options,
            callback=print_chunk
        ):
            full_text += chunk
        
        print("\n\n" + "-" * 40)
        print(f"Streaming completed. Total length: {len(full_text)} characters")
        
    except Exception as e:
        print(f"\n\nError occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(simple_stream_test())