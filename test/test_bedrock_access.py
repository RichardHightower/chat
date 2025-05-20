"""
Test script to check AWS Bedrock model access.
"""

import os
import boto3
import json
from dotenv import load_dotenv
import logging
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def check_bedrock_access():
    """Check which AWS Bedrock models are accessible with current credentials."""
    # Check if AWS credentials are set
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_session_token = os.getenv("AWS_SESSION_TOKEN")
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    
    if not aws_access_key or not aws_secret_key:
        print("Error: AWS credentials not found in environment variables.")
        print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your .env file.")
        return
    
    # Log credential information
    print(f"AWS Region: {aws_region}")
    print(f"Using Access Key ID: {aws_access_key[:4]}...{aws_access_key[-4:] if len(aws_access_key) > 8 else ''}")
    if aws_session_token:
        print("Using temporary credentials with session token")
    
    try:
        # Create boto3 clients for Bedrock services
        bedrock_client_kwargs = {
            "region_name": aws_region,
            "aws_access_key_id": aws_access_key,
            "aws_secret_access_key": aws_secret_key
        }
        
        # Add session token if available
        if aws_session_token:
            bedrock_client_kwargs["aws_session_token"] = aws_session_token
        
        # Create clients for both the management API and runtime API
        bedrock_mgmt = boto3.client("bedrock", **bedrock_client_kwargs)
        bedrock_runtime = boto3.client("bedrock-runtime", **bedrock_client_kwargs)
        
        print("\n=== Testing AWS Bedrock IAM Permissions ===")
        
        # Test management API permissions
        try:
            print("\nTesting ListFoundationModels permission:")
            response = bedrock_mgmt.list_foundation_models()
            print("✅ Success - You have permission to list foundation models")
        except ClientError as e:
            print(f"❌ Failed - {e.response['Error']['Message']}")
            print("You may need 'bedrock:ListFoundationModels' permission in your IAM policy")
        
        # List available model IDs
        try:
            print("\n=== Available Bedrock Models ===")
            response = bedrock_mgmt.list_foundation_models()
            models = response.get("modelSummaries", [])
            
            if not models:
                print("No models found. You need to request access to models in the AWS Bedrock console.")
                return
            
            print(f"Found {len(models)} accessible models")
            
            # Group models by provider
            providers = {}
            for model in models:
                model_id = model["modelId"]
                provider = model_id.split(".")[0] if "." in model_id else "other"
                if provider not in providers:
                    providers[provider] = []
                providers[provider].append(model)
            
            # Print models by provider
            for provider, provider_models in providers.items():
                print(f"\n{provider.upper()} ({len(provider_models)} models):")
                for model in provider_models:
                    model_id = model["modelId"]
                    
                    # Check if the model has throughputType information available
                    # Note: Not all models return this information
                    model_throughput = model.get("inferenceTypesSupported", [])
                    if model_throughput:
                        supports_on_demand = "ON_DEMAND" in model_throughput
                        throughput_info = "✅ On-demand supported" if supports_on_demand else "⚠️ Requires provisioned throughput"
                        print(f"  - {model_id} ({throughput_info})")
                    else:
                        # If no throughput info, just print the model ID
                        print(f"  - {model_id}")
                    
            # Test access to a specific model
            print("\n=== Testing Model Access ===")
            
            # Try to access a Claude model if available
            claude_models = [m for m in models if "anthropic.claude" in m["modelId"]]
            if claude_models:
                test_model = claude_models[0]["modelId"]
                print(f"Testing access to {test_model}...")
                
                try:
                    # For Claude models, use the appropriate API based on the model version
                    if "claude-3" in test_model.lower():
                        # Claude 3 models use the converse API
                        request_id = "test-" + os.urandom(4).hex()
                        response = bedrock_runtime.converse(
                            modelId=test_model,
                            messages=[
                                {"role": "user", "content": "Hello, can you say hi?"}
                            ],
                            conversationId=request_id
                        )
                    else:
                        # Older Claude models use InvokeModel
                        prompt = "Human: Hello, can you say hi?\nAssistant:"
                        body = json.dumps({"prompt": prompt, "max_tokens_to_sample": 50})
                        response = bedrock_runtime.invoke_model(
                            modelId=test_model,
                            contentType="application/json",
                            body=body
                        )
                    
                    print(f"✅ Success! Model {test_model} is accessible")
                    
                except ClientError as e:
                    error_msg = e.response.get('Error', {}).get('Message', str(e))
                    print(f"❌ Failed - {error_msg}")
                    print(f"You don't have access to use {test_model}")
                    print("Check the 'Model access' section in the AWS Bedrock console")
            else:
                print("No Claude models found to test")
                
            # Try a different provider if Claude isn't available
            if not claude_models:
                if len(models) > 0:
                    test_model = models[0]["modelId"]
                    print(f"Testing access to {test_model}...")
                    try:
                        # Try generic InvokeModel API
                        # Note: different models have different input formats, so this may fail
                        body = json.dumps({"prompt": "Hello", "max_tokens": 50})
                        response = bedrock_runtime.invoke_model(
                            modelId=test_model,
                            contentType="application/json",
                            body=body
                        )
                        print(f"✅ Success! Model {test_model} is accessible")
                    except ClientError as e:
                        error_msg = e.response.get('Error', {}).get('Message', str(e))
                        print(f"❌ Failed - {error_msg}")
                        print(f"You don't have access to use {test_model} or the request format was incorrect")
        
        except ClientError as e:
            print(f"Error listing models: {e.response['Error']['Message']}")
    
    except Exception as e:
        print(f"Error checking AWS Bedrock access: {str(e)}")

if __name__ == "__main__":
    check_bedrock_access()