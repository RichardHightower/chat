"""
Test script to verify streaming functionality for all providers.
"""

import os
import asyncio
from dotenv import load_dotenv
import logging

# Use the application's logger
from src.chat.util.logging_util import logger

# Import provider classes
from src.chat.ai.anthropic import AnthropicProvider
from src.chat.ai.google_gemini import GoogleGeminiProvider
from src.chat.ai.open_ai import OpenAIProvider
from src.chat.ai.perplexity import PerplexityProvider
from src.chat.ai.ollama import OllamaProvider

# Load environment variables
load_dotenv()

async def test_provider_streaming(provider_class, model_name, prompt):
    """Test streaming with a specific provider."""
    print(f"\n\n{'=' * 50}")
    print(f"Testing {provider_class.__name__} with model {model_name}")
    print(f"{'=' * 50}")
    
    try:
        # Initialize provider
        provider = provider_class(model=model_name)
        print(f"Provider initialized: {provider.__class__.__name__}")
        
        print(f"\nPrompt: '{prompt}'\n")
        print("Response:")
        
        # Define callback to print chunks
        def print_chunk(text):
            print(text, end="", flush=True)
        
        # Get the streaming generator
        generator = provider.generate_completion_stream(
            prompt=prompt,
            callback=print_chunk
        )
        
        # Manually iterate over the generator
        full_text = ""
        while True:
            try:
                chunk = await anext(generator)
                full_text += chunk
            except StopAsyncIteration:
                break
        
        print("\n\n" + "-" * 40)
        print(f"Streaming completed. Total length: {len(full_text)} characters")
        
    except Exception as e:
        print(f"\n\nError testing {provider_class.__name__}: {str(e)}")

async def main():
    """Run streaming tests for all providers."""
    # Test prompt
    prompt = "Write a short poem about streaming data, one line at a time."
    
    # Test each provider
    await test_provider_streaming(AnthropicProvider, "claude-3-7-sonnet-latest", prompt)
    await test_provider_streaming(GoogleGeminiProvider, "gemini-2-flash", prompt)
    await test_provider_streaming(OpenAIProvider, "gpt-4o-2024-08-06", prompt)
    await test_provider_streaming(PerplexityProvider, "sonar-pro", prompt)
    
    # For Ollama, we need to ensure the server is running and a model is available
    # Uncomment this line if Ollama is set up
    # await test_provider_streaming(OllamaProvider, "llama3.3:latest", prompt)

if __name__ == "__main__":
    asyncio.run(main())