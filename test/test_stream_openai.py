"""
Test specifically for OpenAI streaming using our utility functions.
"""

import os
import asyncio
from dotenv import load_dotenv
import litellm
from src.chat.util.streaming_util import stream_response
from src.chat.ai.open_ai import OpenAIProvider
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def test_openai_streaming():
    """Test streaming with OpenAI provider."""
    # Ensure API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
    
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Initialize provider
    provider = OpenAIProvider(api_key=api_key, model="gpt-4o-2024-08-06")
    
    # Test prompt
    prompt = "Write a short poem about code streaming, with each line building on the previous one."
    
    print(f"\nTest prompt: '{prompt}'\n")
    print("Response:")
    
    # Define callback for printing chunks
    def print_chunk(chunk):
        print(chunk, end="", flush=True)
    
    # Run streaming test directly with provider
    full_text = ""
    try:
        # Get the generator
        generator = provider.generate_completion_stream(
            prompt=prompt,
            output_format="text",
            callback=print_chunk
        )
        
        # Manually iterate over the generator
        while True:
            try:
                chunk = await anext(generator)
                full_text += chunk
            except StopAsyncIteration:
                break
        
        print("\n\n" + "-" * 40)
        print(f"Streaming completed. Total length: {len(full_text)} characters")
        
    except Exception as e:
        print(f"\n\nError during OpenAI streaming: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_openai_streaming())