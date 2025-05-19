"""
Test script for streaming functionality with Anthropic Claude.

This script tests the streaming functionality independently from the main application.
"""

import os
import asyncio
import src.chat.ai.anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

async def test_streaming():
    """Test the streaming functionality with Claude API."""
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment variables.")
        return

    # Initialize the Anthropic client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Test prompt
    prompt = "Write a short poem about streaming data, one line at a time."
    
    print(f"Sending prompt to Claude: {prompt}")
    print("\nResponse:")
    
    # Set up streaming
    full_response = ""
    try:
        with client.messages.stream(
            messages=[{"role": "user", "content": prompt}],
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0.7,
            system="You are a helpful assistant."
        ) as stream:
            # Process each chunk as it arrives
            for chunk in stream:
                if chunk.type == "content_block_delta" and chunk.delta.type == "text":
                    # Extract text from the chunk
                    text_chunk = chunk.delta.text
                    if text_chunk:
                        full_response += text_chunk
                        # Print each chunk with a cursor
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