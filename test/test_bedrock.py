"""
Test script for AWS Bedrock LLM provider.
"""

import os
import asyncio
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the AWS Bedrock provider
from src.chat.ai.bedrock import BedrockProvider

# Load environment variables from .env file
load_dotenv()

async def test_bedrock_provider():
    """Test the AWS Bedrock provider with Claude model."""
    # Check if AWS credentials are set
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_session_token = os.getenv("AWS_SESSION_TOKEN")
    
    if not aws_access_key or not aws_secret_key:
        print("Error: AWS credentials not found in environment variables.")
        print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your .env file.")
        return
        
    # Log if using session token
    if aws_session_token:
        print("Using temporary credentials with AWS session token")
    else:
        print("Using long-term credentials (no session token provided)")
    
    # Default model - Claude 3 Sonnet
    model = "anthropic.claude-3-sonnet-20240229-v1:0"
    
    try:
        # Initialize the provider
        provider = BedrockProvider(model=model)
        print(f"Provider initialized with model: {model}")
        
        # Test prompt
        prompt = "Write a short paragraph explaining what AWS Bedrock is and its benefits."
        print(f"\nSending prompt: '{prompt}'\n")
        
        # Get completion
        response = await provider.generate_completion(prompt=prompt)
        print("Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        # Test streaming (optional)
        print("\nTesting streaming response with the same prompt...")
        
        # Define callback to print chunks
        def print_chunk(text):
            print(text, end="", flush=True)
        
        # Get streaming generator
        print("\nStreaming response:")
        print("-" * 50)
        
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
        
        print("\n" + "-" * 50)
        print(f"Streaming completed. Total length: {len(full_text)} characters")
        
    except Exception as e:
        print(f"\n\nError testing AWS Bedrock provider: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_bedrock_provider())