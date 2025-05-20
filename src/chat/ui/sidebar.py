"""
Sidebar UI components for the chat application.

This module contains functions for rendering the sidebar UI components
including provider selection, model selection, temperature controls,
and conversation management.
"""

import streamlit as st
import uuid
from datetime import datetime
import os
from typing import Dict, Any, Tuple

from chat.conversation.conversation import MessageType
from chat.conversation.conversation_storage import ConversationStorage
from chat.util.logging_util import logger as llm_logger


def render_provider_settings(providers: Dict[str, Dict[str, Any]]) -> Tuple[str, str, float, bool]:
    """
    Render the provider settings section in the sidebar.
    
    Args:
        providers: Dictionary of available providers and their models
        
    Returns:
        Tuple containing (selected_provider, selected_model, temperature, use_streaming)
    """
    st.header("Provider Settings")
    
    # Provider selection
    selected_provider = st.selectbox("Select Provider", list(providers.keys()))
    
    # Model selection for the chosen provider
    provider_info = providers[selected_provider]
    selected_model = st.selectbox("Select Model", provider_info["models"])
    
    # Temperature slider
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    # Streaming option - default to enabled for all providers
    # All providers now support streaming
    use_streaming = st.checkbox("Enable streaming responses", 
                              value=True,
                              help="When enabled, responses will stream in real-time instead of waiting for the complete response.")
    
    # Provider-specific settings
    if selected_provider == "Ollama":
        render_ollama_settings(selected_model)
    elif selected_provider == "AWS Bedrock":
        render_bedrock_settings()

    return selected_provider, selected_model, temperature, use_streaming


def render_bedrock_settings():
    """Render AWS Bedrock-specific settings."""
    st.subheader("AWS Bedrock Settings")
    
    # Get current AWS settings from environment variables
    current_region = os.environ.get("AWS_REGION", "us-east-1")
    
    # Allow the user to select a region
    aws_regions = [
        "us-east-1", "us-east-2", "us-west-1", "us-west-2", 
        "eu-west-1", "eu-central-1", "ap-northeast-1", "ap-southeast-1", 
        "ap-southeast-2", "ap-south-1"
    ]
    
    selected_region = st.selectbox("AWS Region", aws_regions, 
                                  index=aws_regions.index(current_region) if current_region in aws_regions else 0)
    
    # Update the environment variable if it has changed
    if selected_region != current_region:
        os.environ["AWS_REGION"] = selected_region
        llm_logger.info(f"Updated AWS Region to: {selected_region}")
    
    # Check AWS credentials
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    aws_session_token = os.environ.get("AWS_SESSION_TOKEN", "")
    
    if not aws_access_key or not aws_secret_key:
        st.warning("⚠️ AWS credentials not configured. Please add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to your .env file.")
    else:
        # Show masked credentials
        st.success("✅ AWS credentials configured")
        st.text(f"Access Key ID: {aws_access_key[:4]}...{aws_access_key[-4:] if len(aws_access_key) > 8 else ''}")
        
        # Show session token info if present
        if aws_session_token:
            st.info("🔑 Using temporary credentials with session token")
            # Show a very short preview of the token for confirmation
            token_preview = aws_session_token[:4] + "..." + aws_session_token[-4:] if len(aws_session_token) > 8 else ""
            st.text(f"Session Token: {token_preview}")
        
    # Add a button to check available models
    if st.button("Check Available Bedrock Models"):
        try:
            import boto3
            
            # Create a Bedrock client
            bedrock_service_client = boto3.client(
                "bedrock",
                region_name=selected_region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                **({"aws_session_token": aws_session_token} if aws_session_token else {})
            )
            
            try:
                # Try to list models
                response = bedrock_service_client.list_foundation_models()
                accessible_models = [model["modelId"] for model in response.get("modelSummaries", [])]
                
                if accessible_models:
                    st.success(f"✅ Found {len(accessible_models)} accessible models in your AWS account")
                    
                    # Display models by provider
                    provider_models = {}
                    for model_id in accessible_models:
                        provider = model_id.split(".")[0] if "." in model_id else "other"
                        if provider not in provider_models:
                            provider_models[provider] = []
                        provider_models[provider].append(model_id)
                    
                    for provider, models in provider_models.items():
                        with st.expander(f"{provider.capitalize()} Models ({len(models)})"):
                            st.write("\n".join([f"- `{model}`" for model in models]))
                else:
                    st.warning("No models found. You need to request access to models in the AWS Bedrock console.")
            
            except Exception as e:
                if "AccessDeniedException" in str(e):
                    st.error("Access denied. Your IAM user/role needs 'bedrock:ListFoundationModels' permission.")
                else:
                    st.error(f"Error listing models: {str(e)}")
                    
        except Exception as e:
            st.error(f"Error checking models: {str(e)}")
    
    # Information about AWS Bedrock
    with st.expander("About AWS Bedrock"):
        st.markdown("""
        **Amazon Bedrock** is a fully managed service that makes high-performing foundation models (FMs) from leading 
        AI companies available through a unified API.
        
        Available models include:
        - Claude (from Anthropic)
        - Llama 3 (from Meta)
        - Titan (from Amazon)
        - Command (from Cohere)
        
        To use AWS Bedrock with this chat app:
        1. Make sure you have AWS credentials configured in your .env file
        2. Ensure your AWS account has access to the selected models
        3. Choose the appropriate region where the models are available
        
        **Important Requirements:**
        
        1. **Request Model Access**
           - Go to AWS Management Console → Amazon Bedrock → Model access
           - Request access to the models you want to use
           - Some models are approved immediately, others may require a waiting period
        
        2. **On-Demand vs. Provisioned Throughput**
           - This app uses "on-demand throughput" (pay-as-you-go usage)
           - Some newer models (like some Claude 3.7 variants) only work with "provisioned throughput"
           - Provisioned throughput requires creating an inference profile in the AWS console
           - Stick with models that support on-demand throughput unless you set up an inference profile
        
        3. **Model Versions Matter**
           - Use the exact model ID that matches what's available in your AWS account
           - Use the "Check Available Bedrock Models" button above to verify your available models
        
        For more information, visit [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
        """)


def render_ollama_settings(selected_model: str = ""):
    """Render Ollama-specific settings."""
    st.subheader("Ollama Settings")

    # Get the current base URL
    current_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    # Allow the user to change the base URL
    ollama_base_url = st.text_input("Ollama API Base URL", value=current_base_url)

    # Update the environment variable if it has changed
    if ollama_base_url != current_base_url:
        os.environ["OLLAMA_BASE_URL"] = ollama_base_url
        llm_logger.info(f"Updated Ollama base URL to: {ollama_base_url}")

    # Model-specific settings
    if selected_model:
        st.subheader(f"Model: {selected_model}")

        # Show different settings based on model size
        is_large_model = any(size in selected_model for size in ["70b", "72b"])
        is_medium_model = any(size in selected_model for size in ["27b", "32b"])

        if is_large_model:
            st.warning(f"⚠️ {selected_model} is a very large model that requires significant RAM (40-45GB). Responses may take longer and context length is limited.")

            # Context size settings for large models
            context_size = st.slider("Context Tokens", min_value=256, max_value=4096, value=2048, step=256,
                      help="Maximum number of tokens to generate. Lower values reduce memory usage.", key="context_slider_large")
            st.session_state.ollama_context_size = context_size

        elif is_medium_model:
            st.warning(f"⚠️ {selected_model} is a large model that requires significant RAM (15-20GB). Responses may take longer.")

            # Context size settings for medium models
            context_size = st.slider("Context Tokens", min_value=512, max_value=6144, value=2560, step=512,
                      help="Maximum number of tokens to generate. Lower values reduce memory usage.", key="context_slider_medium")
            st.session_state.ollama_context_size = context_size

        elif "llama4:scout" in selected_model:
            st.success(f"✅ {selected_model} is Meta's newest model, optimized for efficiency and performance.")

            # Scout-specific settings
            st.info("💡 Llama 4 Scout is designed for efficient operation with good performance.")
            context_size = st.slider("Context Tokens", min_value=1024, max_value=8192, value=4096, step=1024,
                      help="Maximum number of tokens to generate.", key="context_slider_scout")
            st.session_state.ollama_context_size = context_size

        else:
            st.success(f"✅ {selected_model} is optimized for your system and should run efficiently.")

        # Add model-specific notes based on model name
        if "deepseek" in selected_model:
            st.info("💡 DeepSeek models excel at reasoning tasks and problem-solving.")
        elif "gemma" in selected_model:
            st.info("💡 Gemma models are Google's efficient models with good instruction following.")
        elif "qwen" in selected_model:
            st.info("💡 Qwen models have strong multilingual capabilities.")

    # Option to check Ollama status
    if st.button("Check Ollama Status"):
        try:
            import requests
            try:
                response = requests.get(f"{ollama_base_url}/api/version", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Ollama is running! Version: {data.get('version', 'unknown')}")

                    # Also check for loaded models
                    try:
                        models_response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
                        if models_response.status_code == 200:
                            models_data = models_response.json()
                            model_count = len(models_data.get("models", []))
                            st.info(f"Found {model_count} models available on your Ollama server.")
                        else:
                            st.warning("Could not retrieve model information.")
                    except Exception as e:
                        st.warning(f"Could not check available models: {e}")

                else:
                    st.error(f"Ollama returned status code {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to Ollama. Please check that it's running and the URL is correct.")
            except Exception as e:
                st.error(f"Error checking Ollama status: {e}")
        except ImportError:
            st.error("Requests library not available. Cannot check Ollama status.")

    # Information about Ollama
    with st.expander("About Ollama"):
        st.markdown("""
        **Ollama** lets you run open-source large language models locally on your machine.
        
        To use Ollama with this chat app:
        1. Install Ollama from [ollama.ai](https://ollama.ai)
        2. Ensure the Ollama server is running
        3. Set the correct base URL above (default: http://localhost:11434)
        
        Currently installed models:
        - gemma3:27b (Google's 27B parameter model)
        - qwen3:32b (Alibaba's newer 32B parameter model)
        - qwen:72b (Alibaba's 72B parameter multilingual model)
        - deepseek-r1:70b (70B parameter specialized reasoning model)
        - llama3.3:latest (Meta's Llama 3.3 model)
        - llama4:scout (Meta's newest Llama 4 Scout model)
        
        To install additional models, use: `ollama pull MODEL_NAME`
        """)


def render_conversation_management(
    conversation_storage: ConversationStorage,
    selected_provider: str,
    selected_model: str
) -> None:
    """
    Render the conversation management section in the sidebar.

    Args:
        conversation_storage: Instance of ConversationStorage
        selected_provider: Currently selected provider name
        selected_model: Currently selected model name
    """
    st.header("Conversation Management")

    # Context maintenance option
    maintain_context = st.checkbox(
        "Maintain Conversation Context",
        value=True,
        help="When enabled, the app will send the entire conversation history to the LLM."
    )

    # Store the context maintenance setting in session state
    st.session_state.maintain_context = maintain_context

    # Load existing conversations
    st.subheader("Saved Conversations")
    try:
        # Create storage directory if it doesn't exist
        os.makedirs(os.environ.get("CONVERSATION_STORAGE_DIR", "conversations"), exist_ok=True)

        conversation_list = conversation_storage.list_conversations()

        if conversation_list:
            # Create a selectbox of saved conversations
            conversation_options = ["Current"] + [
                f"{c['title']} ({datetime.fromisoformat(c['updated_at']).strftime('%Y-%m-%d %H:%M')})"
                for c in conversation_list
            ]
            selected_conversation = st.selectbox("Select Conversation", conversation_options)

            # Load the selected conversation
            if selected_conversation != "Current" and conversation_list:
                # Find the selected conversation index
                selected_idx = conversation_options.index(selected_conversation) - 1  # Adjust for "Current"

                if st.button("Load Selected Conversation"):
                    # Get the conversation ID
                    conversation_id = conversation_list[selected_idx]["id"]

                    # Load the conversation
                    st.session_state.conversation_id = conversation_id
                    # Clear the current conversation object to force reload
                    if "conversation_obj" in st.session_state:
                        del st.session_state.conversation_obj

                    # Update messages in session state from the loaded conversation
                    loaded_conversation = conversation_storage.load_conversation(conversation_id)
                    if loaded_conversation:
                        st.session_state.messages = [
                            {"role": "user" if msg.message_type == MessageType.INPUT else "assistant",
                             "content": msg.content}
                            for msg in loaded_conversation.messages
                        ]
                        st.success(f"Loaded conversation: {loaded_conversation.title or conversation_id[:8]}")
                        st.rerun()
        else:
            st.info("No saved conversations found. Start chatting to create one!")
    except Exception as e:
        llm_logger.error(f"Error loading conversation list: {e}", exc_info=True)
        st.error("Could not load saved conversations")

    # Conversation actions
    st.subheader("Conversation Actions")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("New Conversation"):
            # Create a new conversation
            st.session_state.conversation_id = str(uuid.uuid4())
            if "conversation_obj" in st.session_state:
                del st.session_state.conversation_obj

            # Clear messages
            st.session_state.messages = [
                {"role": "assistant",
                 "content": f"Hello! I'm using {selected_provider}'s {selected_model}. How can I help you today?"}
            ]
            st.rerun()

    with col2:
        if st.button("Save Conversation"):
            # Get the current conversation
            from chat.ui.conversation_manager import get_conversation
            current_conversation = get_conversation(conversation_storage)

            # Generate a title if none exists
            if not current_conversation.title:
                current_conversation.title = conversation_storage.generate_conversation_title(current_conversation)

            # Save the conversation
            if conversation_storage.save_conversation(current_conversation):
                st.success(f"Conversation saved: {current_conversation.title or current_conversation.id[:8]}")
            else:
                st.error("Failed to save conversation")


def render_current_conversation_details(conversation_storage: ConversationStorage,
                                        selected_provider: str,
                                        selected_model: str) -> None:
    """
    Render the current conversation details and management options.

    Args:
        conversation_storage: Instance of ConversationStorage
        selected_provider: Currently selected provider name
        selected_model: Currently selected model name
    """
    st.subheader("Current Conversation")

    try:
        # Get current conversation
        from chat.ui.conversation_manager import get_conversation
        conversation = get_conversation(conversation_storage)

        # Edit conversation title
        current_title = conversation.title or conversation_storage.generate_conversation_title(conversation)
        new_title = st.text_input("Conversation Title", value=current_title)

        if new_title != current_title:
            conversation.title = new_title
            try:
                if conversation_storage.save_conversation(conversation):
                    st.success("Title updated")
                else:
                    st.error("Failed to update title")
            except Exception as e:
                llm_logger.error(f"Error updating conversation title: {e}", exc_info=True)
                st.error("Error saving title")

        # Delete conversation button
        if st.button("Delete Current Conversation"):
            try:
                if conversation_storage.delete_conversation(st.session_state.conversation_id):
                    # Create a new conversation
                    st.session_state.conversation_id = str(uuid.uuid4())
                    if "conversation_obj" in st.session_state:
                        del st.session_state.conversation_obj

                    # Clear messages
                    st.session_state.messages = [
                        {"role": "assistant",
                         "content": f"Previous conversation deleted. I'm ready for a new conversation using {selected_provider}'s {selected_model}!"}
                    ]
                    st.success("Conversation deleted")
                    st.rerun()
                else:
                    st.error("Failed to delete conversation")
            except Exception as e:
                llm_logger.error(f"Error deleting conversation: {e}", exc_info=True)
                st.error("Error deleting conversation")

        # Export conversation
        if st.button("Export Conversation"):
            try:
                # Create a download link for the conversation history
                conversation_data = "\n".join([
                    f"[{msg['role']}] ({datetime.now().strftime('%Y-%m-%d %H:%M')}): {msg['content']}"
                    for msg in st.session_state.messages
                ])

                title_slug = current_title.replace(" ", "_").lower()[:30]
                st.download_button(
                    label="Download Conversation",
                    data=conversation_data,
                    file_name=f"{title_slug}_{st.session_state.conversation_id[:8]}.txt",
                    mime="text/plain"
                )
            except Exception as e:
                llm_logger.error(f"Error exporting conversation: {e}", exc_info=True)
                st.error("Error generating export")

        # Display conversation statistics
        st.text(f"ID: {conversation.id[:8]}...")
        st.text(f"Messages: {len(conversation.messages)}")
        st.text(f"Created: {conversation.created_at.strftime('%Y-%m-%d %H:%M')}")
        st.text(f"Updated: {conversation.updated_at.strftime('%Y-%m-%d %H:%M')}")

    except Exception as e:
        llm_logger.error(f"Error in conversation management sidebar: {e}", exc_info=True)
        st.error("Error loading conversation details")
        st.text("Try starting a new conversation.")